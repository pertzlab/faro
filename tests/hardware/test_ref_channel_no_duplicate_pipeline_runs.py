"""Hardware verification: PR #4 (``6d9802a``) + PR #6 (``3c14b85``).

PR #4 fixed a bug where a ref frame at the last timepoint caused the
pipeline to run **twice** for that timepoint — inflating the per-FOV
``fov_timestep`` counter so it drifted ahead of the MDA ``timestep``.

PR #6 added ``TracksDispenser`` so pipeline workers process frames in
acquisition order. Under real cellpose variance a thread-order race
would desync ``fov_timestep`` from ``timestep`` in the parquet.

Both regressions manifest the same way — ``fov_timestep != timestep``
— so a single assertion (``assert_timestep_ordering``) pins both
contracts. Running with a ref channel on the last frame exercises
the PR #4 path; concurrent cellpose workers exercise PR #6.

Gated by ``--scope`` / ``FARO_SCOPE``.
"""

from __future__ import annotations

import pytest
from useq import Position, TIntervalLoops

from faro.core.controller import Controller
from faro.core.data_structures import Channel as RTMChannel, RTMSequence
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from faro.feature_extraction.optocheck import OptoCheckFE
from faro.feature_extraction.simple import SimpleFE
from faro.tracking.trackpy import TrackerTrackpy
from tests.hardware.conftest import (
    assert_clean_run,
    assert_timestep_ordering,
    load_tracks_df,
)


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0


@pytest.mark.hardware
def test_ref_channel_no_duplicate_pipeline_runs(
    microscope, scope_config, safe_positions, tmp_path, cellpose_segmentator
) -> None:
    """Ref channel on last frame must not duplicate pipeline runs."""
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
        ref_channels=(optocheck,),
        # ``-1`` resolves to the last timepoint — the exact timing
        # that PR #4 broke before the fix.
        ref_frames=frozenset({-1}),
    )

    pipeline = ImageProcessingPipeline(
        storage_path=str(tmp_path),
        segmentators=[cellpose_segmentator],
        feature_extractor=SimpleFE("labels"),
        feature_extractor_ref=OptoCheckFE(used_mask="labels"),
        tracker=TrackerTrackpy(),
    )
    writer = OmeZarrWriter(storage_path=str(tmp_path), store_stim_images=False)

    controller = Controller(microscope, pipeline, writer=writer)
    try:
        controller.run_experiment(list(sequence), stim_mode="current")
    finally:
        controller.finish_experiment()

    assert_clean_run(controller, tmp_path, expect_tracks=True)

    df = load_tracks_df(tmp_path)
    assert not df.empty, "no tracks parquet files written — pipeline didn't run"
    assert_timestep_ordering(df)
