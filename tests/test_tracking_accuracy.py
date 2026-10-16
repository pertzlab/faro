"""Tracking accuracy against synthetic ground truth.

Drives the full Controller → Microscope → Pipeline → Writer loop using a
fake microscope that renders many cells moving along known trajectories.
Recovers the tracker's particle IDs from the output parquet and compares
them to the GT trajectories, asserting coverage and ID-switch rate.

Complements :mod:`tests.test_pipeline_integration` which only exercises
2 static circles.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from skimage.measure import label as cc_label, regionprops
from useq import MDAEvent

from faro.core.controller import Controller
from faro.core.data_structures import Channel, RTMEvent, SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import SegmentatorBinary
from faro.stimulation.base import StimWithPipeline
from faro.tracking.motile_tracker import TrackerMotile
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import render_disks

IMG_SIZE = 256
N_CELLS = 50
N_FRAMES = 15
CELL_RADIUS = 3
CELL_VALUE = 50000

# (row, col) shift for each imaging channel — channel 0 is the reference
# (what the segmenter sees), channel 1 simulates a small registration
# offset that downstream multichannel consumers might need to tolerate.
CHANNEL_OFFSETS = [(0, 0), (3, 3)]

# Pipeline / tracker knobs. SEARCH_RANGE is the max per-frame displacement
# a tracker is willing to link over, so it must be > MAX_STEP (velocity +
# noise) or correct detections get dropped as unrelated.
SEARCH_RANGE = 10
MEMORY = 3
# A detection within MATCH_RADIUS px of a GT cell counts as that GT cell.
# Generous vs CELL_RADIUS so centroid jitter from thresholding doesn't
# fail matches.
MATCH_RADIUS = 5.0


def _make_gt(n_cells: int, n_frames: int, seed: int = 0) -> list[np.ndarray]:
    """Return per-frame arrays of shape (n_cells, 2) with (row, col) positions.

    Cells start on a perturbed grid, move with a constant velocity plus
    per-frame Gaussian jitter, and reflect off the image margins so they
    stay segmentable.
    """
    rng = np.random.default_rng(seed)
    margin = 20

    # Perturbed grid so cells stay well-separated at t=0 (Otsu/binary
    # segmentation relies on connected components; overlap merges labels).
    side = int(np.ceil(np.sqrt(n_cells)))
    spacing = (IMG_SIZE - 2 * margin) / side
    grid = np.array(
        [
            (margin + (i // side + 0.5) * spacing, margin + (i % side + 0.5) * spacing)
            for i in range(n_cells)
        ]
    )
    pos = grid + rng.uniform(-2, 2, size=grid.shape)
    vel = rng.normal(0, 1.0, size=(n_cells, 2))

    frames = []
    for _ in range(n_frames):
        frames.append(pos.copy())
        pos = pos + vel + rng.normal(0, 0.3, size=pos.shape)
        # Reflect off margins so cells can't leave the FOV.
        for d in (0, 1):
            below = pos[:, d] < margin
            above = pos[:, d] > IMG_SIZE - margin
            vel[below, d] = np.abs(vel[below, d])
            vel[above, d] = -np.abs(vel[above, d])
    return frames


def _render_frame(positions: np.ndarray, offset=(0, 0)) -> np.ndarray:
    return render_disks(
        positions,
        img_size=IMG_SIZE,
        radius=CELL_RADIUS,
        value=CELL_VALUE,
        offset=offset,
    )


class SyntheticCellScene:
    """Scene plugin for :class:`FakeMicroscope` renders N moving cells.

    ``gt[t]`` is an (N, 2) array of (row, col) positions at timepoint t.
    The GT particle ID for cell ``i`` is simply ``i`` across all frames.

    When ``with_slm=True`` the scene declares an SLM so the Controller
    stim branch runs; dispatched masks are recorded in :attr:`slm_events`
    as ``(frame_idx, ndarray)``.
    """

    image_height = IMG_SIZE
    image_width = IMG_SIZE
    channels = ("phase-contrast", "fitc", "stim-405")

    def __init__(
        self,
        n_cells: int = N_CELLS,
        n_frames: int = N_FRAMES,
        seed: int = 0,
        *,
        with_slm: bool = False,
    ):
        self.gt = _make_gt(n_cells, n_frames, seed=seed)
        self.slm_events: list[tuple[int, np.ndarray]] = []
        self.slm_name = "SLM" if with_slm else None
        self.slm_shape = (IMG_SIZE, IMG_SIZE) if with_slm else None

    def render(self, event: MDAEvent) -> np.ndarray:
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        if t >= len(self.gt):
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)
        offset = CHANNEL_OFFSETS[c] if c < len(CHANNEL_OFFSETS) else (0, 0)
        return _render_frame(self.gt[t], offset=offset)

    def on_slm_displayed(self, image: np.ndarray, event: MDAEvent) -> None:
        self.slm_events.append((event.index.get("t", 0), image))


class StimPerCellCenter(StimWithPipeline):
    """Stamp a small disk at each segmented cell's centroid.

    Lets the stim-alignment test compare mask-blob centroids to GT cell
    positions: if the wiring is correct, each disk sits on top of a cell.
    """

    def __init__(self, radius: int = 2):
        self.radius = radius

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        mask = np.zeros_like(labels, dtype=np.uint8)
        yy, xx = np.ogrid[: labels.shape[0], : labels.shape[1]]
        for prop in regionprops(labels):
            cy, cx = prop.centroid
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= self.radius**2
            mask[disk] = 255
        return mask, None


def _make_events(n_frames: int, *, stim_frames=()) -> list[RTMEvent]:
    stim_set = set(stim_frames)
    stim_ch = (Channel(config="stim-405", exposure=100),)
    img_channels = (
        Channel(config="phase-contrast", exposure=50),
        Channel(config="fitc", exposure=50),
    )
    events = []
    for t in range(n_frames):
        has_stim = t in stim_set
        events.append(
            RTMEvent(
                index={"t": t, "p": 0},
                channels=img_channels,
                stim_channels=stim_ch if has_stim else (),
                metadata={},
            )
        )
    return events


def _run_pipeline(
    tracker,
    tmp_dir: str,
    n_frames: int,
    *,
    stimulator=None,
    stim_frames=(),
    stim_mode: str = "current",
) -> FakeMicroscope:
    pipeline = ImageProcessingPipeline(
        storage_path=tmp_dir,
        segmentators=[SegmentationMethod("labels", SegmentatorBinary(), 0, True)],
        tracker=tracker,
        feature_extractor=SimpleFE("labels"),
        stimulator=stimulator,
    )
    scene = SyntheticCellScene(
        n_cells=N_CELLS, n_frames=n_frames, with_slm=stimulator is not None
    )
    mic = FakeMicroscope(scene)
    ctrl = Controller(mic, pipeline)
    ctrl.run_experiment(
        _make_events(n_frames, stim_frames=stim_frames),
        stim_mode=stim_mode,
        validate=False,
    ).wait()
    ctrl._analyzer.wait_idle()
    ctrl.finish_experiment()

    assert not ctrl.background_errors, ctrl.background_errors
    return mic


def _score(mic: FakeMicroscope, df: pd.DataFrame) -> dict:
    """Match tracker detections to GT per frame and score the result.

    Matching: per-frame nearest-neighbor within MATCH_RADIUS. Then for
    each GT trajectory, record which tracker particle IDs covered it and
    count ID-switch transitions.

    The feature extractor writes centroid-0 as ``x`` and centroid-1 as
    ``y`` (see :func:`FeatureExtractor.extract_positions`), so row=x and
    col=y in the dataframe; we match the GT conventions to that.
    """
    gt = mic.scene.gt
    n_frames = len(gt)
    # gt_pid -> ordered list of matched tracker particle IDs (one per covered frame)
    trail_ids: dict[int, list[int]] = {i: [] for i in range(N_CELLS)}
    # gt cell-frames matched to any detection
    matched_cell_frames = 0

    for t in range(n_frames):
        frame = df[df["timestep"] == t]
        if frame.empty:
            continue
        det_xy = frame[["x", "y"]].to_numpy()  # (row, col)
        det_pid = frame["particle"].to_numpy()

        for gt_pid, (gr, gc) in enumerate(gt[t]):
            d2 = (det_xy[:, 0] - gr) ** 2 + (det_xy[:, 1] - gc) ** 2
            j = int(np.argmin(d2))
            if d2[j] <= MATCH_RADIUS**2:
                trail_ids[gt_pid].append(int(det_pid[j]))
                matched_cell_frames += 1

    coverage = matched_cell_frames / (N_CELLS * n_frames)

    switches_per_trail = []
    dominant_fraction = []
    for ids in trail_ids.values():
        if len(ids) < 2:
            continue
        switches = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
        switches_per_trail.append(switches)
        top = Counter(ids).most_common(1)[0][1]
        dominant_fraction.append(top / len(ids))

    return {
        "coverage": coverage,
        "n_gt_trails_tracked": len(switches_per_trail),
        "median_switches": (
            float(np.median(switches_per_trail)) if switches_per_trail else 0.0
        ),
        "mean_dominant_fraction": (
            float(np.mean(dominant_fraction)) if dominant_fraction else 0.0
        ),
    }


@pytest.mark.parametrize(
    "tracker_factory, name",
    [
        (lambda: TrackerTrackpy(search_range=SEARCH_RANGE, memory=MEMORY), "Trackpy"),
        (lambda: TrackerMotile(search_range=SEARCH_RANGE, memory=MEMORY), "Motile"),
    ],
    ids=["Trackpy", "Motile"],
)
def test_tracker_recovers_gt_trajectories(tmp_dir, tracker_factory, name):
    mic = _run_pipeline(tracker_factory(), tmp_dir, n_frames=N_FRAMES)

    parquet = os.path.join(tmp_dir, "tracks", "0_latest.parquet")
    assert os.path.exists(parquet), f"No tracks parquet written at {parquet}"
    df = pd.read_parquet(parquet)

    score = _score(mic, df)
    print(f"[{name}] {score}")

    # Detection sanity: the pipeline should see most GT cell-frames.
    # SegmentatorBinary + well-separated disks is ~perfect, so 90% is
    # generous (leaves headroom for edge cases and future regressions).
    assert (
        score["coverage"] >= 0.90
    ), f"{name}: only {score['coverage']:.1%} of GT cell-frames detected"

    # Tracking quality: of the GT trails that produced ≥2 matched frames,
    # the tracker should link ≥90% of them under a single particle ID.
    assert score["n_gt_trails_tracked"] >= int(
        0.9 * N_CELLS
    ), f"{name}: only {score['n_gt_trails_tracked']}/{N_CELLS} GT trails had ≥2 matches"
    assert (
        score["mean_dominant_fraction"] >= 0.90
    ), f"{name}: dominant tracker-ID fraction {score['mean_dominant_fraction']:.2f} < 0.90"
    assert (
        score["median_switches"] <= 1
    ), f"{name}: median ID switches per GT trail = {score['median_switches']}"


def _mask_blob_centroids(mask: np.ndarray) -> np.ndarray:
    """Return an (n_blobs, 2) array of (row, col) centroids of an SLM mask."""
    cc = cc_label(mask > 0)
    if cc.max() == 0:
        return np.zeros((0, 2))
    return np.array([p.centroid for p in regionprops(cc)])


@pytest.mark.parametrize("stim_mode", ["current", "previous"])
def test_stim_masks_centered_on_cells(tmp_dir, stim_mode):
    """StimPerCellCenter should drop a disk on each cell at every stim frame.

    The mask dispatched at frame ``t`` is built from whichever frame the
    controller picked:

    * ``stim_mode="current"`` — frame ``t``'s own segmentation. Mask
      centroids must line up with GT positions at frame ``t``.
    * ``stim_mode="previous"`` — frame ``t-1``'s segmentation. The cells
      drifted between ``t-1`` and ``t``, so matching against ``gt[t]``
      would surface that drift as a spurious "offset". We therefore
      match against ``gt[t-1]`` — the frame the mask was actually built
      from. Under both modes the alignment is thus equally tight
      (segmentation ↔ computed-from frame is exact up to centroid
      rounding).
    """
    stim_frames = tuple(range(1, N_FRAMES))  # skip t=0: mirror real-expt warm-up
    mic = _run_pipeline(
        TrackerTrackpy(search_range=SEARCH_RANGE, memory=MEMORY),
        tmp_dir,
        n_frames=N_FRAMES,
        stimulator=StimPerCellCenter(radius=2),
        stim_frames=stim_frames,
        stim_mode=stim_mode,
    )

    slm_events = mic.scene.slm_events
    # One SLM event per stim frame.
    assert len(slm_events) == len(
        stim_frames
    ), f"Expected {len(stim_frames)} SLM events, got {len(slm_events)}"

    gt_offset = -1 if stim_mode == "previous" else 0
    distances = []
    mask_counts = []
    for t, mask in slm_events:
        gt = mic.scene.gt[t + gt_offset]
        centroids = _mask_blob_centroids(mask)
        mask_counts.append(len(centroids))
        for cy, cx in centroids:
            d = np.sqrt(((gt[:, 0] - cy) ** 2 + (gt[:, 1] - cx) ** 2).min())
            distances.append(d)

    # Every delivered mask should have ~N_CELLS blobs (one per cell).
    # Allow one off — a cell can momentarily ride the margin wall.
    for t, n in zip([e[0] for e in slm_events], mask_counts):
        assert (
            n >= N_CELLS - 1
        ), f"[{stim_mode}] stim frame t={t}: {n} mask blobs, expected ≥{N_CELLS - 1}"

    distances = np.asarray(distances)
    # Centroid of a thresholded disk sits within ~1 px of the rendered
    # cell centre; allow 2 px median and 4 px worst-case.
    assert np.median(distances) <= 2.0, (
        f"[{stim_mode}] median mask-to-cell distance "
        f"{np.median(distances):.2f}px too large"
    )
    assert np.max(distances) <= 4.0, (
        f"[{stim_mode}] max mask-to-cell distance "
        f"{np.max(distances):.2f}px too large"
    )


def test_previous_mode_mask_is_one_frame_behind_cells(tmp_dir):
    """Regression guard: in ``previous`` mode the dispatched mask reflects
    frame ``t-1``, not frame ``t`` — i.e. measuring against ``gt[t]`` must
    surface roughly one frame of drift. If the pipeline ever accidentally
    routed the "current" frame's mask here, this test's floor would fail.
    """
    stim_frames = tuple(range(1, N_FRAMES))
    mic = _run_pipeline(
        TrackerTrackpy(search_range=SEARCH_RANGE, memory=MEMORY),
        tmp_dir,
        n_frames=N_FRAMES,
        stimulator=StimPerCellCenter(radius=2),
        stim_frames=stim_frames,
        stim_mode="previous",
    )

    # Displacement between the dispatched mask centroids and the
    # current-frame GT positions. We expect ~1 frame of drift per cell
    # (the scene's per-frame velocity has std 1 px on each axis).
    per_cell_drift = []
    for t, mask in mic.scene.slm_events:
        gt = mic.scene.gt[t]
        for cy, cx in _mask_blob_centroids(mask):
            d = np.sqrt(((gt[:, 0] - cy) ** 2 + (gt[:, 1] - cx) ** 2).min())
            per_cell_drift.append(d)

    # Median displacement should be clearly nonzero — otherwise the test
    # would also pass with mode="current" and wouldn't catch a regression.
    median_drift = float(np.median(per_cell_drift))
    assert median_drift >= 0.5, (
        f"Expected ~1 frame of drift between mask and current-frame cells, "
        f"got median {median_drift:.2f}px — mask may not actually be from t-1."
    )
