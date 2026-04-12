import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

import motile
from motile.costs import EdgeDistance, Appear, Disappear, NodeSelection, Split
from motile.constraints import MaxParents, MaxChildren
from motile.variables import EdgeSelected

from faro.tracking.base import Tracker


# Names returned by compute_custom_costs() and the node attribute used to
# carry each per-tip value into the motile solver.
_CUSTOM_COST_ATTR = {
    "split": "split_readiness",
    "disappear": "disappear_readiness",
}


class TrackerMotile(Tracker):
    """ILP-based tracker using motile for frame-by-frame assignment.

    Each frame, builds a small 2-frame graph:
      - Frame T:   active track tips (from ``df_old``)
      - Frame T+1: new detections (from ``df_new``)
      - Edges:     candidates within ``search_range``

    Previous assignments are never revisited.

    Particle IDs are scoped per-FOV (``fov_state.next_particle_id``). Across
    FOVs the unique identifier is the ``(fov, particle)`` pair, matching
    :class:`~faro.tracking.trackpy.TrackerTrackpy`.

    Subclass and override :meth:`compute_custom_costs` to modulate per-tip
    costs based on track history — e.g. reward divisions when a cell-cycle
    marker is rising, or cheapen track loss when a death marker is high.

    Args:
        search_range: Maximum distance (px) to consider a candidate edge.
        memory: Frames a track can be lost before it is dropped.
        max_children: Outgoing edges per node (2 allows divisions).
        max_parents: Incoming edges per node (2 allows merges).
        appear_cost: Cost for a new track appearing.
        disappear_cost: Cost for an existing track being lost.
        node_selection_cost: Reward (negative) for selecting a node.
        split_cost: Base penalty for a division event.
    """

    def __init__(
        self,
        search_range=50,
        memory=0,
        max_children=2,
        max_parents=1,
        appear_cost=30.0,
        disappear_cost=30.0,
        node_selection_cost=-10.0,
        split_cost=15.0,
    ):
        super().__init__()
        self.search_range = search_range
        self.memory = memory
        self.max_children = max_children
        self.max_parents = max_parents
        self.appear_cost = appear_cost
        self.disappear_cost = disappear_cost
        self.node_selection_cost = node_selection_cost
        self.split_cost = split_cost

    def compute_custom_costs(
        self,
        df_old: pd.DataFrame,
        tip_particles: np.ndarray,
        current_t: int,
    ) -> dict[str, np.ndarray]:
        """Per-tip cost adjustments derived from track history.

        Override this in a subclass to bias specific events using features
        accumulated over past frames. Returns a dict mapping cost name to a
        length-``n_tips`` float array. Negative values *reward* the event;
        positive values *penalize* it. Values are **added** to the base cost
        configured on the constructor.

        Supported keys:

        - ``"split"``     — modulates :class:`motile.costs.Split`
                            (negative rewards divisions).
        - ``"disappear"`` — modulates :class:`motile.costs.Disappear`
                            (negative allows cheap track loss, e.g. death).

        Default: no modulation (empty dict).

        Example — reward divisions when a cell-cycle marker is rising, and
        cheapen track loss when a death marker is high::

            class TrackerMotileCustom(TrackerMotile):
                def compute_custom_costs(self, df_old, tip_particles, current_t):
                    window = 20
                    mask = (
                        (df_old["fov_timestep"] >= current_t - window)
                        & (df_old["particle"].isin(tip_particles))
                    )
                    recent = df_old.loc[mask, ["particle", "cc_marker", "death_marker"]]
                    grouped = {pid: g for pid, g in recent.groupby("particle")}

                    n = len(tip_particles)
                    split = np.zeros(n)
                    disappear = np.zeros(n)
                    for i, pid in enumerate(tip_particles):
                        g = grouped.get(pid)
                        if g is None or len(g) < 3:
                            continue
                        cc = g["cc_marker"].to_numpy()
                        split[i] = -max(0.0, (cc[-1] - cc[0]) * 5 + cc[-1] * 10)
                        disappear[i] = -20.0 * g["death_marker"].iloc[-1]
                    return {"split": split, "disappear": disappear}
        """
        return {}

    def track_cells(
        self, df_old: pd.DataFrame, df_new: pd.DataFrame, fov_state
    ) -> pd.DataFrame:
        if df_new.empty:
            return df_old

        missing = [c for c in ("x", "y", "label") if c not in df_new.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        current_t = fov_state.fov_timestep_counter
        n_det = len(df_new)
        det_pos = df_new[["x", "y"]].to_numpy()

        if df_old.empty:
            df_new["particle"] = _alloc_ids(fov_state, n_det)
            df_new["fov_timestep"] = current_t
            return df_new.reset_index(drop=True)

        tips = _get_active_tips(df_old, current_t, self.memory)
        if tips.empty:
            df_new["particle"] = _alloc_ids(fov_state, n_det)
            df_new["fov_timestep"] = current_t
            return pd.concat([df_old, df_new], ignore_index=True)

        n_tips = len(tips)
        tip_pos = tips[["x", "y"]].to_numpy()
        tip_particles = tips["particle"].to_numpy()

        custom = self.compute_custom_costs(df_old, tip_particles, current_t) or {}
        tip_attrs = {
            attr: np.asarray(custom.get(name, np.zeros(n_tips)), dtype=float)
            for name, attr in _CUSTOM_COST_ATTR.items()
        }

        tree = cKDTree(det_pos)
        neighbors = tree.query_ball_point(tip_pos, self.search_range)

        # Integer node IDs: tips are 0..n_tips-1, detections are offset after.
        det_offset = n_tips
        G = nx.DiGraph()
        G.add_nodes_from(
            (
                i,
                {
                    "t": 0,
                    "pos": tip_pos[i].tolist(),
                    **{attr: float(vals[i]) for attr, vals in tip_attrs.items()},
                },
            )
            for i in range(n_tips)
        )
        G.add_nodes_from(
            (
                det_offset + j,
                {
                    "t": 1,
                    "pos": det_pos[j].tolist(),
                    **{attr: 0.0 for attr in tip_attrs},
                },
            )
            for j in range(n_det)
        )
        G.add_edges_from(
            (i, det_offset + j)
            for i, js in enumerate(neighbors)
            for j in js
        )

        tg = motile.TrackGraph(G, frame_attribute="t")
        solver = motile.Solver(tg)
        solver.add_cost(EdgeDistance(position_attribute="pos", weight=1.0))
        solver.add_cost(NodeSelection(constant=self.node_selection_cost))
        solver.add_cost(Appear(constant=self.appear_cost))
        solver.add_cost(
            Disappear(
                weight=1.0,
                attribute=_CUSTOM_COST_ATTR["disappear"],
                constant=self.disappear_cost,
            )
        )
        if self.max_children > 1:
            solver.add_cost(
                Split(
                    weight=1.0,
                    attribute=_CUSTOM_COST_ATTR["split"],
                    constant=self.split_cost,
                )
            )
        solver.add_constraint(MaxParents(self.max_parents))
        solver.add_constraint(MaxChildren(self.max_children))
        solution = solver.solve()

        edge_sel = solver.get_variables(EdgeSelected)
        assignments: dict[int, list[int]] = {}
        for tip_idx, det_node in tg.edges:
            if solution[edge_sel[(tip_idx, det_node)]] > 0.5:
                assignments.setdefault(tip_idx, []).append(det_node - det_offset)

        particle_ids = np.full(n_det, -1, dtype=np.int64)
        for tip_idx, det_indices in assignments.items():
            if len(det_indices) == 1:
                particle_ids[det_indices[0]] = int(tip_particles[tip_idx])
            else:
                new_ids = _alloc_ids(fov_state, len(det_indices))
                for det_idx, pid in zip(det_indices, new_ids):
                    particle_ids[det_idx] = pid

        unmatched = particle_ids < 0
        if unmatched.any():
            particle_ids[unmatched] = _alloc_ids(fov_state, int(unmatched.sum()))

        df_new["particle"] = particle_ids.astype(np.uint32)
        df_new["fov_timestep"] = current_t

        return pd.concat([df_old, df_new], ignore_index=True)


def _alloc_ids(fov_state, n):
    """Allocate ``n`` fresh particle IDs from ``fov_state.next_particle_id``."""
    start = fov_state.next_particle_id
    fov_state.next_particle_id = start + n
    return np.arange(start, start + n, dtype=np.uint32)


def _get_active_tips(df_old, current_t, memory):
    """Latest observation per particle, within the memory window."""
    min_t = current_t - 1 - memory
    recent = df_old[df_old["fov_timestep"] >= min_t]
    if recent.empty:
        return recent
    idx = recent.groupby("particle")["fov_timestep"].idxmax()
    return recent.loc[idx]
