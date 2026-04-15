"""Hardware verification: ``SKIP_WAIT_DEVICES`` actually saves wall time.

``Moench.SKIP_WAIT_DEVICES = ("Mosaic3",)`` (commit ``da1354c``) exists
to bypass Mosaic3's perpetually-stuck ``Busy()`` flag, which otherwise
eats a 5 s timeout per MDA event. This test confirms the default skip
list is at least 30 % faster than running with no skip list.

``MoenchMDAEngine._wait_for_system_excluding_xy`` reads
``getattr(mic, "SKIP_WAIT_DEVICES", ())`` on every event (see
``faro/microscope/pertzlab/moench.py:376``) so monkeypatching the class
attribute mid-session takes effect on the next ``run_experiment`` —
no microscope re-init needed.

Gated by ``--scope`` / ``FARO_SCOPE``; only applicable on scopes that
define ``SKIP_WAIT_DEVICES``.
"""

from __future__ import annotations

import time

import pytest
from useq import Position, TIntervalLoops

from faro.core.controller import Controller
from faro.core.data_structures import Channel as RTMChannel, RTMSequence
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from tests.hardware.conftest import assert_clean_run


N_FRAMES = 4
TIME_BETWEEN_TIMESTEPS_S = 2.0  # short so the wait overhead dominates
# Default skip list must save at least this fraction of wall time.
# Wait overhead is ~5 s × N_FRAMES when the Mosaic3 times out each
# event, which dwarfs a 2 s interval, so the gap is huge in practice.
MIN_SPEEDUP = 1.3


def _build_sequence(cfg, safe_positions):
    imaging = RTMChannel(
        config=cfg["imaging_channel"],
        exposure=cfg["imaging_exposure"],
        group=cfg["channel_group"],
    )
    return RTMSequence(
        stage_positions=[
            Position(x=p["x"], y=p["y"], z=p["z"], name=p["name"])
            # One FOV keeps wait overhead the dominant cost.
            for p in safe_positions[:1]
        ],
        time_plan=TIntervalLoops(
            interval=TIME_BETWEEN_TIMESTEPS_S,
            loops=N_FRAMES,
        ),
        channels=[imaging],
    )


def _time_acquisition(microscope, sequence, tmp_path) -> float:
    pipeline = ImageProcessingPipeline(storage_path=str(tmp_path))
    writer = OmeZarrWriter(storage_path=str(tmp_path), store_stim_images=False)
    controller = Controller(microscope, pipeline, writer=writer)
    start = time.monotonic()
    try:
        controller.run_experiment(list(sequence), stim_mode="current")
    finally:
        controller.finish_experiment()
    elapsed = time.monotonic() - start
    assert_clean_run(controller, tmp_path, expect_tracks=False)
    return elapsed


@pytest.mark.hardware
def test_skip_wait_devices_speedup(
    microscope, scope_name, scope_config, safe_positions, tmp_path, monkeypatch
) -> None:
    """Default ``SKIP_WAIT_DEVICES`` must be >=30 % faster than empty skip list."""
    mic_cls = type(microscope)
    if not getattr(mic_cls, "SKIP_WAIT_DEVICES", None):
        pytest.skip(f"{scope_name!r} scope does not declare SKIP_WAIT_DEVICES")

    default_skip = mic_cls.SKIP_WAIT_DEVICES
    sequence = _build_sequence(scope_config, safe_positions)

    # Run 1: no skip list (monkeypatched). This is the slow path —
    # Mosaic3 Busy() times out on every MDA event.
    monkeypatch.setattr(mic_cls, "SKIP_WAIT_DEVICES", ())
    slow_dir = tmp_path / "noskip"
    slow_dir.mkdir()
    slow_elapsed = _time_acquisition(microscope, sequence, slow_dir)

    # Run 2: restore the default skip list.
    monkeypatch.setattr(mic_cls, "SKIP_WAIT_DEVICES", default_skip)
    fast_dir = tmp_path / "skip"
    fast_dir.mkdir()
    fast_elapsed = _time_acquisition(microscope, sequence, fast_dir)

    ratio = slow_elapsed / max(fast_elapsed, 1e-3)
    print(
        f"SKIP_WAIT_DEVICES speedup: "
        f"noskip={slow_elapsed:.1f}s  skip={fast_elapsed:.1f}s  "
        f"ratio={ratio:.2f}x  (required >= {MIN_SPEEDUP:.2f}x)"
    )
    assert ratio >= MIN_SPEEDUP, (
        f"SKIP_WAIT_DEVICES={default_skip!r} didn't meaningfully speed up "
        f"acquisition — noskip={slow_elapsed:.1f}s vs skip={fast_elapsed:.1f}s "
        f"(ratio {ratio:.2f}x < required {MIN_SPEEDUP:.2f}x). "
        f"Either the skip list is wrong for this scope or the Mosaic3 "
        f"Busy() bug is no longer triggering."
    )
