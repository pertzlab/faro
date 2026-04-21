"""Hardware verification: pipeline lag + FrameDispenser under real cadence.

PR #3 replaced the pipeline/storage ``SimpleQueue`` with ``FrameDispenser``,
a frame-indexed handoff that lets storage peek at the mask produced by
frame t-1 (previous mode) without consuming it, and walks past skipped
frames on demand. The simulator tests exercise the dispenser under
synthetic timing; this test exercises it under a real camera cadence
where a deliberately-slow pipeline lags acquisition by 2-3 frames.

Pass condition: the run finishes without deadlock, every stim frame
has a non-empty stored mask, and no background errors were recorded.

Gated by ``--scope`` / ``FARO_SCOPE`` like the other hardware tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from useq import Position, TIntervalLoops

from faro.core.controller import Controller
from faro.core.data_structures import (
    PowerChannel,
    RTMSequence,
    Channel as RTMChannel,
)
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from tests.hardware.pertzlab.conftest import (
    DelayedMaskStim,
    assert_clean_run,
    open_stim_channel_array,
)


# Long enough to cover lag recovery; short enough to stay under ~2 min.
N_FRAMES = 12
TIME_BETWEEN_TIMESTEPS_S = 5.0
# Pipeline artificial latency — picked to sit between one and two frame
# intervals so the pipeline consistently lags acquisition by ~2 frames.
SLOW_PIPELINE_DELAY_S = 7.0


@pytest.mark.hardware
def test_pipeline_lag_no_deadlock(
    microscope, scope_config, safe_positions, tmp_path
) -> None:
    """Slow pipeline must not deadlock acquisition; all masks must land."""
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
        stim_frames=frozenset(range(1, N_FRAMES)),
    )

    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path),
        stimulator=DelayedMaskStim(delay_s=SLOW_PIPELINE_DELAY_S),
    )
    writer = OmeZarrWriter(storage_path=str(tmp_path), store_stim_images=True)

    controller = Controller(microscope, pipeline, writer=writer)
    try:
        controller.run_experiment(list(sequence), stim_mode="current")
    finally:
        controller.finish_experiment()

    assert_clean_run(controller, tmp_path, expect_tracks=False)

    arr, stim_idx = open_stim_channel_array(tmp_path / "acquisition.ome.zarr")
    for t in range(1, N_FRAMES):
        for p in range(len(safe_positions)):
            stored = np.asarray(arr[t, p, stim_idx])
            assert (stored > 0).any(), (
                f"stim_mask missing at t={t},p={p} — pipeline lag may have "
                f"caused a dispenser skip or deadlock"
            )
