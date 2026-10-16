"""Tests for interactive pause (queue-drain) and fixed-duration WaitEvents.

Two features pinned here:

* **Interactive pause** drains the engine's backpressure window into a
  held buffer and refills on resume. The invariant: pausing/resuming a
  run changes *when* frames are acquired, never *what* — same frame
  count, same indices, byte-identical OME-Zarr output.

* **WaitEvents** insert a timed gap between phases. They claim no t/p
  index (so downstream indices are unchanged vs. no wait), emit no
  MDAEvents (so they add no frames), and shift subsequent events'
  ``min_start_time`` later by at least their duration.

Pause is driven from the main thread by polling ``handle.status()`` so
the tests are deterministic; events are spaced via ``min_start_time`` so
the real MDA engine is slow enough to observe the paused state.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import zarr

from faro.core.controller import Controller
from faro.core.data_structures import RTMEvent, WaitEvent, combine, wait
from faro.core.writers import OmeZarrWriter
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_events, make_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spaced(events: list[RTMEvent], dt: float) -> list[RTMEvent]:
    """Stamp ``min_start_time = i*dt`` so the MDA engine paces itself."""
    return [
        e.model_copy(update={"min_start_time": i * dt}) for i, e in enumerate(events)
    ]


def _wait_until(predicate, *, timeout: float = 5.0, poll: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def _trackpy():
    return TrackerTrackpy(search_range=50, memory=3)


def _raw_tiff_names(path: str) -> list[str]:
    raw_dir = os.path.join(path, "raw")
    if not os.path.isdir(raw_dir):
        return []
    return sorted(f for f in os.listdir(raw_dir) if f.endswith(".tiff"))


def _run_capture_zarr(path: str, events, *, pause: bool) -> np.ndarray:
    """Run ``events`` into an OME-Zarr store, optionally pausing once
    mid-run, then return the raw array (t, c, y, x)."""
    os.makedirs(path, exist_ok=True)
    writer = OmeZarrWriter(path, store_stim_images=False)
    pipeline = make_pipeline(path, tracker=_trackpy(), with_stim=False)
    ctrl = Controller(FakeMicroscope(CircleScene()), pipeline, writer=writer)

    handle = ctrl.run_experiment(events, validate=False)
    if pause:
        assert _wait_until(
            lambda: handle.status().n_events_consumed >= 2
        ), "run finished before it could be paused — widen spacing"
        handle.pause()
        assert _wait_until(
            lambda: handle.status().state == "paused"
        ), "feed loop never reported the paused state"
        handle.resume()
    handle.wait()
    ctrl._analyzer.wait_idle()
    ctrl._analyzer.shutdown(wait=True)  # closes the writer

    root = zarr.open_group(os.path.join(path, "acquisition.ome.zarr"), mode="r")
    return np.asarray(root["0"])


# ---------------------------------------------------------------------------
# WaitEvent semantics (combine-level, no acquisition)
# ---------------------------------------------------------------------------


class TestWaitEventSemantics:
    def test_wait_constructs_waitevent(self):
        w = wait(5.0)
        assert isinstance(w, WaitEvent)
        assert w.duration_s == 5.0

    def test_wait_emits_no_mda_events(self):
        assert wait(5.0).plan_events() == []

    def test_wait_claims_no_index_and_drops_no_frames(self):
        """combine with/without a wait yields identical frame indices —
        the wait must not consume a t-slot."""
        a, b = make_events(3), make_events(3)
        no_wait = combine(a, b, axis="t")
        with_wait = combine(make_events(3), wait(10.0), make_events(3), axis="t")

        frames_no = [e.index for e in no_wait if not isinstance(e, WaitEvent)]
        frames_with = [e.index for e in with_wait if not isinstance(e, WaitEvent)]
        assert frames_no == frames_with
        assert sum(isinstance(e, WaitEvent) for e in with_wait) == 1

    def test_wait_shifts_subsequent_min_start_time(self):
        duration = 10.0
        no_wait = combine(make_events(3), make_events(3), axis="t")
        with_wait = combine(make_events(3), wait(duration), make_events(3), axis="t")

        # The second phase begins at the first event with t == 3.
        def first_second_phase_time(events):
            return next(
                e.min_start_time
                for e in events
                if not isinstance(e, WaitEvent) and e.index.get("t") == 3
            )

        shift = first_second_phase_time(with_wait) - first_second_phase_time(no_wait)
        assert shift >= duration


# ---------------------------------------------------------------------------
# WaitEvent in a real run — adds time, not frames
# ---------------------------------------------------------------------------


class TestWaitEventRun:
    def test_wait_adds_no_frames(self, tmp_dir):
        """A wait between two 2-frame phases yields exactly 4 raw frames."""
        events = combine(make_events(2), wait(0.05), make_events(2), axis="t")
        pipeline = make_pipeline(tmp_dir, tracker=_trackpy(), with_stim=False)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)
        handle = ctrl.run_experiment(events, validate=False)
        handle.wait()
        ctrl._analyzer.wait_idle()
        ctrl._analyzer.shutdown(wait=True)
        assert len(_raw_tiff_names(tmp_dir)) == 4

    def test_wait_passes_through_waiting_state(self, tmp_dir):
        events = combine(make_events(2), wait(0.3), make_events(2), axis="t")
        pipeline = make_pipeline(tmp_dir, tracker=_trackpy(), with_stim=False)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)

        seen_waiting = []
        handle = ctrl.run_experiment(events, validate=False)
        handle.statusChanged.connect(
            lambda s: (
                seen_waiting.append(s.state == "waiting")
                if s.state == "waiting"
                else None
            )
        )
        handle.wait()
        ctrl.finish_experiment()
        assert any(seen_waiting), "run never entered the 'waiting' state"


# ---------------------------------------------------------------------------
# Interactive pause — output equivalence + state machine
# ---------------------------------------------------------------------------


class TestInteractivePause:
    def test_pause_resume_preserves_frame_count(self, tmp_dir):
        events = _spaced(make_events(8), dt=0.1)
        pipeline = make_pipeline(tmp_dir, tracker=_trackpy(), with_stim=False)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)

        handle = ctrl.run_experiment(events, validate=False)
        assert _wait_until(lambda: handle.status().n_events_consumed >= 2)
        handle.pause()
        assert _wait_until(lambda: handle.status().state == "paused")
        handle.resume()
        final = handle.wait()
        ctrl._analyzer.wait_idle()
        ctrl._analyzer.shutdown(wait=True)

        assert final.state == "done"
        assert len(_raw_tiff_names(tmp_dir)) == 8
        assert final.n_events_acquired == 8

    def test_pause_produces_identical_zarr(self, tmp_dir):
        """Paused and unpaused runs of the same events must yield
        byte-identical OME-Zarr raw data."""
        events = _spaced(make_events(8), dt=0.1)
        raw_no_pause = _run_capture_zarr(
            os.path.join(tmp_dir, "nopause"), events, pause=False
        )
        raw_paused = _run_capture_zarr(
            os.path.join(tmp_dir, "paused"), events, pause=True
        )
        np.testing.assert_array_equal(raw_no_pause, raw_paused)

    def test_cancel_during_pause_exits_cleanly(self, tmp_dir):
        events = _spaced(make_events(8), dt=0.1)
        pipeline = make_pipeline(tmp_dir, tracker=_trackpy(), with_stim=False)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)

        handle = ctrl.run_experiment(events, validate=False)
        assert _wait_until(lambda: handle.status().n_events_consumed >= 2)
        handle.pause()
        assert _wait_until(lambda: handle.status().state == "paused")
        handle.cancel()
        assert _wait_until(lambda: not handle.is_running(), timeout=10)
        ctrl.finish_experiment()
        # The worker drains and exits cleanly to a terminal "done" state;
        # cancellation is proven by stopping early (not all 8 acquired)
        # with no fatal error rather than by the transient "cancelling".
        final = handle.status()
        assert final.fatal_error is None
        assert final.n_events_acquired < 8
