"""Hardware verification: stim-mode timing contract.

Axis-cleanup (PR #3) unified stim-mode semantics so that the stored
``stim_mask/{t}`` reflects whatever physically fired at frame t,
independent of mode:

- ``stim_mode="current"``: at frame t, imaging runs → pipeline builds
  mask from image_t → stim fires mask_t. Stored ``stim_mask/{t}`` is
  the frame-t mask.
- ``stim_mode="previous"``: at frame t, stim fires the mask produced
  during frame t-1 → then imaging runs. Stored ``stim_mask/{t}`` is
  the frame-(t-1) mask.
- Frame 0 in previous mode fires no stim (short-circuit in
  ``_build_stim_slm`` when ``t < 0``).

These three tests pin that contract on real hardware, using
``StimLine`` because its output is a pure function of the ``time_step``
metadata — no segmentation/tracking — so the expected mask per frame
is recomputable and comparable to what was stored.

Gated by ``--scope`` / ``FARO_SCOPE`` like the other hardware tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from useq import Position, TIntervalLoops

from faro.core.controller import Controller
from faro.core.data_structures import (
    PowerChannel,
    RTMSequence,
    Channel as RTMChannel,
)
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from faro.stimulation.moving_line_20x import StimLine
from tests.hardware.conftest import assert_clean_run


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0


def _build_sequence_and_stimulator(cfg, safe_positions, microscope):
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

    image_height = microscope.mmc.getImageHeight()
    image_width = microscope.mmc.getImageWidth()

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
        # Stim every frame so the per-frame stored-mask check has
        # something to compare against for t = 1..N-1. In previous
        # mode, frame 0 is expected to fire nothing regardless.
        stim_frames=frozenset(range(0, N_FRAMES)),
    )

    stimulator = StimLine(
        first_stim_frame=0,
        frames_for_1_loop=N_FRAMES,
        stripe_width=image_width // 4,
        n_frames_total=N_FRAMES,
        mask_height=image_height,
        mask_width=image_width,
    )
    return sequence, stimulator


def _open_stim_mask_array(zarr_path):
    """Return the per-FOV stim_mask arrays from the OME-Zarr store.

    Layout is (t, p, c, y, x) with stim channels appended after the
    imaging channels (see OmeZarrWriter). We pull the channel whose
    name starts with ``stim_mask`` for simplicity.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    multiscales = root.attrs["ome"]["multiscales"][0]
    channels = multiscales["metadata"]["omero"]["channels"]
    channel_names = [c["label"] for c in channels]
    stim_idx = next(
        i for i, name in enumerate(channel_names) if name.startswith("stim_mask")
    )
    arr = root[multiscales["datasets"][0]["path"]]
    return arr, stim_idx


@pytest.mark.hardware
def test_current_mode_stored_mask_matches_built_mask(
    microscope, scope_config, safe_positions, tmp_path
) -> None:
    """Stored ``stim_mask/{t}`` must equal StimLine.build(t) in current mode."""
    sequence, stimulator = _build_sequence_and_stimulator(
        scope_config, safe_positions, microscope
    )
    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path), stimulator=stimulator
    )
    writer = OmeZarrWriter(storage_path=str(tmp_path), store_stim_images=True)

    controller = Controller(microscope, pipeline, writer=writer)
    try:
        controller.run_experiment(list(sequence), stim_mode="current")
    finally:
        controller.finish_experiment()

    assert_clean_run(controller, tmp_path, expect_tracks=False)

    zarr_path = tmp_path / "acquisition.ome.zarr"
    arr, stim_idx = _open_stim_mask_array(zarr_path)

    # For each frame t, stored stim_mask should be nonzero wherever
    # StimLine's frame-t mask is nonzero. We compare binarized masks
    # because dtype/scaling between write and read may differ.
    for t in range(N_FRAMES):
        expected = stimulator.spot_mask_linescan(
            frame_count_1_loop=stimulator.frames_for_1_loop,
            time_step=t,
            offset=stimulator.first_stim_frame,
            stripe_width=stimulator.stripe_width,
            height=stimulator.mask_height,
            width=stimulator.mask_width,
        )
        for p in range(len(safe_positions)):
            stored = np.asarray(arr[t, p, stim_idx])
            assert (stored > 0).any(), f"no mask stored at t={t}, p={p}"
            # Allow off-by-a-few-pixels tolerance from writer scaling,
            # but require >90% agreement on the stripe footprint.
            overlap = np.logical_and(stored > 0, expected > 0).sum()
            total = (expected > 0).sum()
            assert overlap / max(total, 1) > 0.9, (
                f"stored mask at t={t},p={p} does not match "
                f"StimLine frame t={t} (current mode contract)"
            )


@pytest.mark.hardware
def test_previous_mode_stored_mask_is_shifted_by_one(
    microscope, scope_config, safe_positions, tmp_path
) -> None:
    """Stored ``stim_mask/{t}`` must equal StimLine.build(t-1) in previous mode."""
    sequence, stimulator = _build_sequence_and_stimulator(
        scope_config, safe_positions, microscope
    )
    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path), stimulator=stimulator
    )
    writer = OmeZarrWriter(storage_path=str(tmp_path), store_stim_images=True)

    controller = Controller(microscope, pipeline, writer=writer)
    try:
        controller.run_experiment(list(sequence), stim_mode="previous")
    finally:
        controller.finish_experiment()

    assert_clean_run(controller, tmp_path, expect_tracks=False)

    zarr_path = tmp_path / "acquisition.ome.zarr"
    arr, stim_idx = _open_stim_mask_array(zarr_path)

    for t in range(1, N_FRAMES):
        expected = stimulator.spot_mask_linescan(
            frame_count_1_loop=stimulator.frames_for_1_loop,
            time_step=t - 1,
            offset=stimulator.first_stim_frame,
            stripe_width=stimulator.stripe_width,
            height=stimulator.mask_height,
            width=stimulator.mask_width,
        )
        for p in range(len(safe_positions)):
            stored = np.asarray(arr[t, p, stim_idx])
            overlap = np.logical_and(stored > 0, expected > 0).sum()
            total = (expected > 0).sum()
            assert overlap / max(total, 1) > 0.9, (
                f"stored mask at t={t},p={p} does not match "
                f"StimLine frame t={t - 1} (previous-mode contract)"
            )


@pytest.mark.hardware
def test_previous_mode_frame_zero_fires_nothing(
    microscope, scope_config, safe_positions, tmp_path
) -> None:
    """Previous mode must short-circuit at frame 0 (stored mask all zeros)."""
    sequence, stimulator = _build_sequence_and_stimulator(
        scope_config, safe_positions, microscope
    )
    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path), stimulator=stimulator
    )
    writer = OmeZarrWriter(storage_path=str(tmp_path), store_stim_images=True)

    controller = Controller(microscope, pipeline, writer=writer)
    try:
        controller.run_experiment(list(sequence), stim_mode="previous")
    finally:
        controller.finish_experiment()

    assert_clean_run(controller, tmp_path, expect_tracks=False)

    zarr_path = tmp_path / "acquisition.ome.zarr"
    arr, stim_idx = _open_stim_mask_array(zarr_path)

    for p in range(len(safe_positions)):
        stored = np.asarray(arr[0, p, stim_idx])
        assert not (stored > 0).any(), (
            f"frame 0 in previous mode fired a nonzero mask at p={p}; "
            f"short-circuit in _build_stim_slm is not engaged"
        )
