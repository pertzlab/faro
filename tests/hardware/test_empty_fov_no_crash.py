"""Hardware verification: empty-FOV guards survive real cellpose.

Commits ``ad2469c`` / ``180fc17`` added guards in
``extract_and_merge_features`` and ``labels_to_particles`` so the
pipeline doesn't crash when cellpose returns zero labels. The
``TestEmptyFrameEdgeCases`` suite covers this in simulation with
``CircleMicroscope(blank_frames=...)``. Real cellpose on a truly empty
frame is a different code path (bail-early conditions are triggered by
real dataframes rather than an all-zero label image), so we want a
matching hardware contract.

This test is **opt-in**: set ``FARO_HW_TEST_EMPTY_FOV=1`` and park
the stage on a cell-free area of the sample before invoking pytest.
The default hardware suite stays on the focused, cell-covered area
the other tests assume.
"""

from __future__ import annotations

import os

import pytest
from useq import Position, TIntervalLoops

from faro.core.controller import Controller
from faro.core.data_structures import (
    Channel as RTMChannel,
    PowerChannel,
    RTMSequence,
)
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from faro.feature_extraction.simple import SimpleFE
from faro.stimulation.base import StimTopEdgeMeta
from faro.tracking.trackpy import TrackerTrackpy
from tests.hardware.conftest import assert_clean_run


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0
STIM_FRACTION = 0.3


@pytest.mark.hardware
@pytest.mark.skipif(
    os.environ.get("FARO_HW_TEST_EMPTY_FOV") != "1",
    reason="empty-FOV test requires a cell-free stage position; opt in with "
    "FARO_HW_TEST_EMPTY_FOV=1 after parking the stage off-cells",
)
def test_empty_fov_no_crash(
    microscope, scope_config, safe_positions, tmp_path, cellpose_segmentator
) -> None:
    """Cellpose returning 0 labels must not crash the pipeline or writer."""
    cfg = scope_config

    imaging = RTMChannel(
        config=cfg["imaging_channel"],
        exposure=cfg["imaging_exposure"],
        group=cfg["channel_group"],
    )
    stim = PowerChannel(
        config=cfg["stim_channel"],
        exposure=cfg["stim_exposure"],
        group=cfg["channel_group"],
        power=cfg["stim_power"],
    )

    sequence = RTMSequence(
        stage_positions=[
            Position(x=p["x"], y=p["y"], z=p["z"], name=p["name"])
            # One FOV is enough — we're testing guard clauses, not coverage.
            for p in safe_positions[:1]
        ],
        time_plan=TIntervalLoops(
            interval=TIME_BETWEEN_TIMESTEPS_S,
            loops=N_FRAMES,
        ),
        channels=[imaging],
        stim_channels=(stim,),
        stim_frames=frozenset(range(1, N_FRAMES)),
        rtm_metadata={"stim_fraction": STIM_FRACTION},
    )

    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path),
        segmentators=[cellpose_segmentator],
        feature_extractor=SimpleFE("labels"),
        stimulator=StimTopEdgeMeta(),
        tracker=TrackerTrackpy(),
    )
    writer = OmeZarrWriter(storage_path=str(tmp_path), store_stim_images=True)

    controller = Controller(microscope, pipeline, writer=writer)
    try:
        controller.run_experiment(list(sequence), stim_mode="current")
    finally:
        controller.finish_experiment()

    # Empty FOVs are allowed to produce no tracks — guards should just
    # no-op rather than crash. ``assert_clean_run`` still checks that
    # background_errors is empty and the zarr / events.json are valid.
    assert_clean_run(controller, tmp_path, expect_tracks=False)
