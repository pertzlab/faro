"""Hardware smoke test: end-to-end line-stimulation acquisition.

Converted from ``experiments/22_line_stimulation/line_stimulation.ipynb``.
Runs a short multi-FOV / multi-timestep acquisition with a moving-line
DMD pattern. No segmentation, tracking, or feature extraction — this
exercises the geometric ``StimLine`` path on its own. The optocheck
reference channel is acquired on the last frame to validate ref-channel
handling without segmentation.

Why this test exists:
    The notebook is the canonical demo for the "stim independent of
    cell labels" pipeline shape. It's the simplest hardware path —
    image, stim with a precomputed mask, image, repeat — and a useful
    early signal that the camera + DMD + filter wheel + LED + writer
    chain is wired correctly.

Differences from the notebook (intentional, for test ergonomics):
    * No napari / napari-micromanager. FOV positions come from the
      ``safe_positions`` fixture as relative offsets from the stage's
      current XY (max 40 µm), so the stage never makes a large move.
    * No ``fovs.json`` file dependency.
    * Output goes to pytest's ``tmp_path`` (auto-cleaned) instead of
      a hardcoded drive letter.
    * The DMD affine matrix is the synthetic identity from
      ``synthetic_affine`` — no interactive calibration step.
    * Frame count and interval are scaled down (4 frames × 5 s) so
      the whole test takes about half a minute.

The test is gated by ``--scope`` / ``FARO_SCOPE`` and skipped by
default. See ``tests/conftest.py`` for the gating logic.
"""

from __future__ import annotations

import os

import pytest
import zarr
from useq import Position, TIntervalLoops

from faro.core.controller import Controller
from faro.core.data_structures import (
    Channel as RTMChannel,
    PowerChannel,
    RTMSequence,
)
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from faro.stimulation.moving_line_20x import StimLine


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0


@pytest.mark.hardware
def test_line_stimulation_smoke(
    microscope, scope_config, safe_positions, tmp_path
) -> None:
    """End-to-end smoke test: geometric line stim, no segmentation."""

    cfg = scope_config

    imaging = RTMChannel(
        config=cfg["imaging_channel"],
        exposure=cfg["imaging_exposure"],
        group=cfg["channel_group"],
    )
    optocheck = RTMChannel(
        config=cfg["optocheck_channel"],
        exposure=cfg["optocheck_exposure"],
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
        stim_frames=frozenset(range(1, N_FRAMES)),
        ref_channels=(optocheck,),
        ref_frames=frozenset({-1}),
    )

    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path),
        # No segmentator/tracker/FE — pure geometric stim from a
        # closed-form mask, dispatched by Analyzer's metadata-only path.
        stimulator=StimLine(
            first_stim_frame=1,
            frames_for_1_loop=N_FRAMES,
            stripe_width=image_width // 4,
            n_frames_total=N_FRAMES,
            mask_height=image_height,
            mask_width=image_width,
        ),
    )

    writer = OmeZarrWriter(
        storage_path=str(tmp_path),
        store_stim_images=True,
    )

    controller = Controller(microscope, pipeline, writer=writer)
    try:
        controller.run_experiment(list(sequence), stim_mode="current")
    finally:
        controller.finish_experiment()

    # Surface background-thread errors. Experiments intentionally keep
    # running after storage / pipeline errors so a transient hardware
    # glitch doesn't abort a long acquisition, but a hardware test is
    # meaningless if it silently swallows errors.
    assert not controller.background_errors, (
        "Background errors during acquisition:\n"
        + "\n".join(
            f"  [{src}] {etype}: {msg}"
            for src, etype, msg, _ in controller.background_errors
        )
    )

    # ------------------------------------------------------------------
    # Output assertions — minimal "did it produce a napari-loadable
    # store?" smoke checks. No segmentation means no tracks/.
    # ------------------------------------------------------------------
    zarr_path = os.path.join(str(tmp_path), "acquisition.ome.zarr")
    assert os.path.isdir(zarr_path), (
        f"OME-Zarr store not created at {zarr_path}"
    )

    grp = zarr.open_group(zarr_path, mode="r")
    assert "ome" in grp.attrs, (
        "OME metadata missing on root group — store will not load in napari"
    )

    assert (tmp_path / "events.json").is_file(), "events.json not written"
