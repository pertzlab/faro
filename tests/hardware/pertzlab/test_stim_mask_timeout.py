"""Hardware verification: stim-mask timeout surfaces as background error.

PR #3 changed stim-mask timeouts from a silent fall-through to a typed
``BackgroundError`` recorded on the controller. This test forces the
condition by pairing a deliberately short ``stim_mask_timeout`` on the
Analyzer with a deliberately slow stimulator, then verifies:

- the acquisition still finishes cleanly (log-and-continue),
- at least one ``BackgroundError`` with source=``stim_mask_timeout``
  (or similarly labeled) is recorded on the controller,
- the writer still produced an OME-Zarr store and events.json.

Because ``assert_clean_run`` fails on any background error, this test
inlines its own post-run assertions instead.

Gated by ``--scope`` / ``FARO_SCOPE`` like the other hardware tests.
"""

from __future__ import annotations

import pytest
import zarr
from useq import Position, TIntervalLoops

import faro.core.controller as controller_mod
from faro.core.controller import Controller
from faro.core.data_structures import (
    PowerChannel,
    RTMSequence,
    Channel as RTMChannel,
)
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from tests.hardware.pertzlab.conftest import DelayedMaskStim


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 5.0
# Pipeline sleeps much longer than the analyzer's timeout so the
# controller is guaranteed to time out waiting for the mask. A
# queue-based stimulator (StimWithImage / StimWithPipeline) is
# required — base Stim is called synchronously and has no timeout.
SLOW_PIPELINE_DELAY_S = 10.0
SHORT_TIMEOUT_S = 1.0


@pytest.mark.hardware
def test_stim_mask_timeout_records_background_error(
    microscope, scope_config, safe_positions, tmp_path, monkeypatch
) -> None:
    """Short stim_mask_timeout + slow pipeline must surface a BackgroundError."""
    # Patch the Analyzer default so every Analyzer constructed during
    # this run uses the short timeout. Controller owns Analyzer creation
    # internally, so this is the cleanest injection point.
    original_init = controller_mod.Analyzer.__init__

    def short_timeout_init(self, *args, **kwargs):
        kwargs.setdefault("stim_mask_timeout", SHORT_TIMEOUT_S)
        # If the caller passed stim_mask_timeout explicitly, still
        # force the short value — we need it to trip.
        kwargs["stim_mask_timeout"] = SHORT_TIMEOUT_S
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(controller_mod.Analyzer, "__init__", short_timeout_init)

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

    # The experiment must have finished (log-and-continue) but recorded
    # at least one timeout error.
    assert controller.background_errors, (
        "expected at least one BackgroundError from stim_mask timeout, "
        "got none — timeout is being silently swallowed again"
    )
    # Controller records the timeout under source="stim_mask"
    # (see faro.core.controller._record_background_error call).
    timeout_errors = [
        e for e in controller.background_errors if e.source == "stim_mask"
    ]
    assert timeout_errors, (
        f"expected a stim_mask BackgroundError; got sources "
        f"{[e.source for e in controller.background_errors]}"
    )

    # And the run still produced a valid store despite the timeout.
    zarr_path = tmp_path / "acquisition.ome.zarr"
    assert zarr_path.is_dir(), "OME-Zarr store missing after timeout run"
    grp = zarr.open_group(str(zarr_path), mode="r")
    assert "ome" in grp.attrs, "OME metadata missing — writer shutdown incomplete"
    assert (tmp_path / "events.json").is_file(), "events.json missing"
