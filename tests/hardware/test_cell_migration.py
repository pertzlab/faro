"""Hardware smoke test: end-to-end cell migration acquisition.

Converted from ``experiments/21_cell_migration/cell_migration.ipynb``.
Runs a short multi-FOV / multi-timestep acquisition with cellpose
segmentation, trackpy tracking, ``StimPercentageOfCell`` DMD
stimulation, and an optocheck reference channel on the last frame
(exercises the ref-channel write path).

Why this test exists:
    The notebook is the canonical end-to-end demo for the segmentation
    + tracking + per-cell stimulation pipeline. Converting it to a
    pytest target lets us validate the whole stack against real
    hardware in CI-style runs without needing napari or interactive
    FOV selection.

Differences from the notebook (intentional, for test ergonomics):
    * No napari / napari-micromanager. FOV positions come from the
      ``safe_positions`` fixture as relative offsets from the stage's
      current XY (max 40 µm), so the stage never makes a large move.
    * Output goes to pytest's ``tmp_path`` (auto-cleaned) instead of
      a hardcoded drive letter.
    * Segmentation is local (cellpose) instead of remote
      (imaging-server-kit) so the test doesn't depend on a separate
      server being reachable.
    * The DMD affine matrix is the synthetic identity from
      ``synthetic_affine`` — no interactive calibration step.
    * Frame count and interval are scaled down (4 frames × 5 s) so
      the whole test takes about a minute.

The test is gated by ``--scope`` / ``FARO_SCOPE`` and skipped by
default. See ``tests/conftest.py`` for the gating logic.
"""

from __future__ import annotations

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
from faro.feature_extraction.optocheck import OptoCheckFE
from faro.feature_extraction.simple import SimpleFE
from faro.stimulation.percentage_of_cell import StimPercentageOfCell
from faro.tracking.trackpy import TrackerTrackpy
from tests.hardware.conftest import (
    assert_clean_run,
    assert_timestep_ordering,
    load_tracks_df,
)


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0
STIM_CELL_PERCENTAGE = 0.3


@pytest.mark.hardware
def test_cell_migration_smoke(
    microscope, scope_config, safe_positions, tmp_path, cellpose_segmentator
) -> None:
    """End-to-end smoke test: segmentation + tracking + stim + ref."""

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
        # Stim every frame except the first (no prior segmentation yet).
        stim_frames=frozenset(range(1, N_FRAMES)),
        ref_channels=(optocheck,),
        # ``-1`` resolves to the last timepoint via _resolve_frame_set.
        ref_frames=frozenset({-1}),
        # StimPercentageOfCell reads this from per-event metadata.
        rtm_metadata={"stim_cell_percentage": STIM_CELL_PERCENTAGE},
    )

    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path),
        segmentators=[cellpose_segmentator],
        feature_extractor=SimpleFE("labels"),
        feature_extractor_ref=OptoCheckFE(used_mask="labels"),
        stimulator=StimPercentageOfCell(),
        tracker=TrackerTrackpy(),
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

    assert_clean_run(controller, tmp_path, expect_tracks=True)
    assert_timestep_ordering(load_tracks_df(tmp_path))
