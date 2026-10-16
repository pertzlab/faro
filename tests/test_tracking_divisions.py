"""Tracking behavior around cell divisions.

* **Tracker-agnostic**: after a 1-to-2 cell split, both daughters are
  tracked through the rest of the run with persistent particle IDs.
  Parametrised across Trackpy and Motile.
* **Trackpy lacks a lineage column**: ``parent_particle`` is absent
  from Trackpy output (trackers may add their own columns).
* **Motile records parent on divisions**: with a strongly negative
  ``split_cost`` the ILP picks the split, and each child's
  ``parent_particle`` points back to the dividing tip.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from useq import MDAEvent

from faro.core.controller import Controller
from faro.core.data_structures import Channel, RTMEvent, SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import SegmentatorBinary
from faro.tracking.motile_tracker import TrackerMotile
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import render_disks


IMG_SIZE = 128
CELL_RADIUS = 4
CELL_VALUE = 50_000
DIVISION_FRAME = 3
N_FRAMES = 6
DAUGHTER_SEPARATION = 12  # px between the two daughters after division


def _render(positions: np.ndarray) -> np.ndarray:
    return render_disks(
        positions, img_size=IMG_SIZE, radius=CELL_RADIUS, value=CELL_VALUE
    )


class _DividingCellScene:
    """One cell for ``DIVISION_FRAME`` frames, then two daughters.

    Before division the parent sits at ``(cy, cx)``. From
    ``DIVISION_FRAME`` onwards the two daughters sit symmetrically
    ``DAUGHTER_SEPARATION / 2`` pixels either side of the parent along
    the column axis and drift outward by 1 px/frame.
    """

    image_height = IMG_SIZE
    image_width = IMG_SIZE
    channels = ("phase-contrast",)

    def __init__(self, cy: int = 64, cx: int = 64):
        self._cy = cy
        self._cx = cx

    def render(self, event: MDAEvent) -> np.ndarray:
        t = event.index.get("t", 0)
        if t < DIVISION_FRAME:
            return _render(np.array([[self._cy, self._cx]]))
        drift = t - DIVISION_FRAME
        half = DAUGHTER_SEPARATION // 2
        return _render(
            np.array(
                [
                    [self._cy, self._cx - half - drift],
                    [self._cy, self._cx + half + drift],
                ]
            )
        )


def _make_events(n_frames: int) -> list[RTMEvent]:
    return [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=(Channel(config="phase-contrast", exposure=50),),
        )
        for t in range(n_frames)
    ]


def _run(tmp_dir: str, tracker) -> pd.DataFrame:
    pipeline = ImageProcessingPipeline(
        storage_path=tmp_dir,
        segmentators=[SegmentationMethod("labels", SegmentatorBinary(), 0, True)],
        tracker=tracker,
        feature_extractor=SimpleFE("labels"),
        stimulator=None,
    )
    scene = _DividingCellScene()
    mic = FakeMicroscope(scene)
    ctrl = Controller(mic, pipeline)
    ctrl.run_experiment(_make_events(N_FRAMES), validate=False).wait()
    ctrl._analyzer.wait_idle()
    ctrl.finish_experiment()
    assert not ctrl.background_errors, ctrl.background_errors
    return pd.read_parquet(os.path.join(tmp_dir, "tracks", "0_latest.parquet"))


# ===========================================================================
# Tracker-agnostic: two daughters survive the split
# ===========================================================================


@pytest.fixture(
    params=[
        pytest.param(TrackerTrackpy, id="Trackpy"),
        pytest.param(TrackerMotile, id="Motile"),
    ],
)
def tracker(request):
    # Generous search_range (> daughter drift per frame) and a memory
    # window so neither tracker hard-rejects the new detections.
    return request.param(search_range=30, memory=3)


def test_division_produces_two_tracked_daughters(tmp_dir, tracker):
    df = _run(tmp_dir, tracker)

    for t in range(DIVISION_FRAME):
        frame = df[df["timestep"] == t]
        assert len(frame) == 1, f"pre-division frame t={t} has {len(frame)} detections"

    daughter_ids_per_frame: list[set[int]] = []
    for t in range(DIVISION_FRAME, N_FRAMES):
        frame = df[df["timestep"] == t]
        assert len(frame) == 2, f"post-division frame t={t} has {len(frame)} detections"
        daughter_ids_per_frame.append(set(frame["particle"].tolist()))

    persistent_ids = set.intersection(*daughter_ids_per_frame)
    assert len(persistent_ids) == 2, (
        f"Expected two particle IDs persisting through all post-division "
        f"frames; got {persistent_ids}. Per-frame: {daughter_ids_per_frame}"
    )


# ===========================================================================
# Tracker-specific: parent_particle lineage column (Motile only)
# ===========================================================================


def test_trackpy_output_has_no_parent_particle_column(tmp_dir):
    """Trackpy doesn't model lineage; the column must be absent entirely
    (not null-filled) so downstream code can tell from schema alone."""
    df = _run(tmp_dir, TrackerTrackpy(search_range=30, memory=3))
    assert "parent_particle" not in df.columns


def test_motile_records_parent_on_division(tmp_dir):
    """With ``split_cost`` strongly negative the ILP picks the 1→2
    division over a "one continues + one appears" assignment. Both
    division children must have ``parent_particle`` pointing back to
    the dividing tip; pre-division rows must be null."""
    df = _run(tmp_dir, TrackerMotile(search_range=30, memory=3, split_cost=-100.0))
    assert "parent_particle" in df.columns

    # Pre-division: no parent info
    pre = df[df["timestep"] < DIVISION_FRAME]
    assert (
        pre["parent_particle"].isna().all()
    ), f"pre-division rows should have <NA> parent, got {pre['parent_particle'].tolist()}"

    # Division frame: both daughters share a single parent ID, and
    # that parent ID matches the particle tracked in the previous frame.
    parent_frame = df[df["timestep"] == DIVISION_FRAME - 1]
    assert len(parent_frame) == 1
    parent_pid = int(parent_frame["particle"].iloc[0])

    div_frame = df[df["timestep"] == DIVISION_FRAME]
    assert len(div_frame) == 2
    parents = div_frame["parent_particle"].tolist()
    assert all(p == parent_pid for p in parents), (
        f"both daughters should trace back to particle {parent_pid}, "
        f"got parents {parents}"
    )
    # Children are freshly-allocated — distinct from each other and from the parent.
    children = set(div_frame["particle"].tolist())
    assert len(children) == 2
    assert parent_pid not in children
