"""Hardware verification: production stim path (cellpose → tracks → DMD).

The existing stim-mode / pipeline-lag / timeout tests use ``StimWithImage``
or base ``Stim`` stubs. Production stim goes through ``StimWithPipeline``
— masks depend on segmentation labels that cellpose produces — which is
the path PR #3's deadlock fix (``180fc17``) and FrameDispenser were
written for. This test exercises it end-to-end: real camera, real
cellpose, real tracker, ``StimTopEdgeMeta`` stimulator, real DMD.

Pass condition: no deadlock, ``controller.background_errors`` empty,
every stim frame has a non-empty stored mask.

Gated by ``--scope`` / ``FARO_SCOPE`` like the other hardware tests.
"""

from __future__ import annotations

import numpy as np
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
from tests.hardware.conftest import assert_clean_run, open_stim_channel_array


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0
STIM_FRACTION = 0.3


@pytest.mark.hardware
def test_stim_with_pipeline_cellpose(
    microscope, scope_config, safe_positions, tmp_path, cellpose_segmentator
) -> None:
    """StimWithPipeline + cellpose: no deadlock, masks land on every stim frame."""
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
            for p in safe_positions
        ],
        time_plan=TIntervalLoops(
            interval=TIME_BETWEEN_TIMESTEPS_S,
            loops=N_FRAMES,
        ),
        channels=[imaging],
        stim_channels=(stim,),
        # Skip t=0: StimTopEdgeMeta needs labels from the previous
        # frame's pipeline run, which doesn't exist at t=0.
        stim_frames=frozenset(range(1, N_FRAMES)),
        # StimTopEdgeMeta reads this per-event from metadata.
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

    assert_clean_run(controller, tmp_path, expect_tracks=True)

    # Every stim frame must land a non-empty mask readout. An empty
    # slot (zarr fill_value=0) means the FrameDispenser dropped the
    # frame or the StimWithPipeline handoff deadlocked.
    arr, stim_idx = open_stim_channel_array(tmp_path / "acquisition.ome.zarr")
    for t in range(1, N_FRAMES):
        for p in range(len(safe_positions)):
            stored = np.asarray(arr[t, p, stim_idx])
            assert (stored > 0).any(), (
                f"stim readout at t={t},p={p} is all zeros — StimWithPipeline "
                f"handoff may have stalled or produced no mask"
            )
