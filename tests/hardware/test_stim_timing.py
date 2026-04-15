"""Hardware verification: stim-mode timing contract.

Axis-cleanup (PR #3) unified stim-mode semantics so that the stim that
physically fires at frame ``t`` depends only on ``stim_mode``, not on
downstream bookkeeping:

- ``stim_mode="current"``: at frame t, imaging runs → the pipeline
  builds mask_t from image_t → stim fires mask_t. The DMD fires at
  every requested stim frame (including t=0).
- ``stim_mode="previous"``: at frame t, the DMD fires mask_{t-1} (the
  mask produced during the previous timestep) → then imaging runs.
  At t=0 there is no predecessor, so the controller passes
  ``suppress_stim=True`` to ``plan_events`` and no stim event is
  emitted — the DMD is not fired and the zarr slot stays at zeros.

These tests pin that contract on real hardware using a pure-geometry
stimulator that returns a full-FOV mask unconditionally. That way
"did the DMD fire?" is readable directly from the stored stim
readout — bright == DMD fired, baseline == DMD did not. It is not
necessary (and not reliable) to compare stored pixel-exact masks to
computed masks because ``store_stim_images=True`` writes the camera
readout, not the mask, and camera ROI / binning / identity-affine
mapping make the coordinate correspondence approximate.

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
from faro.stimulation.base import Stim
from tests.hardware.conftest import assert_clean_run, open_stim_channel_array


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0


class _WholeFovStim(Stim):
    """Base Stim that fires the entire FOV at every stim frame.

    Using a whole-FOV mask decouples the DMD-fire-or-not question from
    DMD↔camera coordinate alignment, which the synthetic identity
    affine does not solve. The stored readout is uniformly bright
    where fired and uniformly at baseline where not.
    """

    def get_stim_mask(self, metadata: dict):
        h, w = metadata.get("img_shape", (1024, 1024))
        return np.ones((h, w), dtype=np.uint8), None


def _build_sequence(cfg, safe_positions, stim_frames):
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
    return RTMSequence(
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
        stim_frames=frozenset(stim_frames),
    )


def _frame_p99(arr, t, p, c) -> float:
    """99th-percentile camera count for a single (t, p, c) frame."""
    return float(np.percentile(np.asarray(arr[t, p, c]), 99))


@pytest.fixture
def run_whole_fov_stim(microscope, tmp_path):
    """Run an RTMSequence through a Controller+OmeZarrWriter with a
    whole-FOV stimulator, and return (controller, zarr_path).

    Factored out because all three tests in this module run the same
    boilerplate — only the sequence and stim_mode vary.
    """

    def _run(sequence, stim_mode: str):
        pipeline = ImageProcessingPipeline(
            storage_path=str(tmp_path), stimulator=_WholeFovStim()
        )
        writer = OmeZarrWriter(
            storage_path=str(tmp_path), store_stim_images=True
        )
        controller = Controller(microscope, pipeline, writer=writer)
        try:
            controller.run_experiment(list(sequence), stim_mode=stim_mode)
        finally:
            controller.finish_experiment()
        return controller, tmp_path / "acquisition.ome.zarr"

    return _run


@pytest.mark.hardware
def test_current_mode_fires_every_requested_stim_frame(
    scope_config, safe_positions, run_whole_fov_stim, brightness_thresholds
) -> None:
    """Current mode must fire the DMD at every frame in ``stim_frames``."""
    sequence = _build_sequence(
        scope_config, safe_positions, stim_frames=range(0, N_FRAMES)
    )
    controller, zarr_path = run_whole_fov_stim(sequence, stim_mode="current")
    assert_clean_run(controller, zarr_path.parent, expect_tracks=False)

    min_p99 = brightness_thresholds["bright_min_p99"]
    arr, stim_idx = open_stim_channel_array(zarr_path)
    for t in range(N_FRAMES):
        for p in range(len(safe_positions)):
            p99 = _frame_p99(arr, t, p, stim_idx)
            assert p99 >= min_p99, (
                f"current mode: stim readout at t={t},p={p} is too dim "
                f"(p99={p99:.0f} < {min_p99:.0f}); DMD may not have fired"
            )


@pytest.mark.hardware
def test_previous_mode_fires_from_frame_one(
    scope_config, safe_positions, run_whole_fov_stim, brightness_thresholds
) -> None:
    """Previous mode must fire the DMD at frames 1..N-1 (not at 0)."""
    sequence = _build_sequence(
        scope_config, safe_positions, stim_frames=range(0, N_FRAMES)
    )
    controller, zarr_path = run_whole_fov_stim(sequence, stim_mode="previous")
    assert_clean_run(controller, zarr_path.parent, expect_tracks=False)

    min_p99 = brightness_thresholds["bright_min_p99"]
    arr, stim_idx = open_stim_channel_array(zarr_path)
    for t in range(1, N_FRAMES):
        for p in range(len(safe_positions)):
            p99 = _frame_p99(arr, t, p, stim_idx)
            assert p99 >= min_p99, (
                f"previous mode: stim readout at t={t},p={p} is too dim "
                f"(p99={p99:.0f} < {min_p99:.0f}); DMD should have fired "
                f"the t-{1} mask"
            )


@pytest.mark.hardware
def test_previous_mode_frame_zero_fires_nothing(
    scope_config, safe_positions, run_whole_fov_stim, brightness_thresholds
) -> None:
    """Previous mode must suppress the stim event at frame 0."""
    sequence = _build_sequence(
        scope_config, safe_positions, stim_frames=range(0, N_FRAMES)
    )
    controller, zarr_path = run_whole_fov_stim(sequence, stim_mode="previous")
    assert_clean_run(controller, zarr_path.parent, expect_tracks=False)

    ratio = brightness_thresholds["bright_vs_dark_ratio"]
    arr, stim_idx = open_stim_channel_array(zarr_path)
    # Ratio-based so we don't bake in an absolute baseline — frame 0
    # should be close to camera dark (suppressed stim → zarr zeros or
    # dark readout) while frame 1 should show DMD illumination.
    for p in range(len(safe_positions)):
        dark = _frame_p99(arr, 0, p, stim_idx)
        bright = _frame_p99(arr, 1, p, stim_idx)
        assert dark * ratio < bright, (
            f"frame 0 in previous mode at p={p} is not dark: "
            f"t=0 p99={dark:.0f}, t=1 p99={bright:.0f}. "
            f"suppress_stim path may not be engaged."
        )
