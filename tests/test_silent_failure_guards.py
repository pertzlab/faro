"""Guards against runs that fail without anyone noticing.

On 2026-08-07 a 46-hour plate stored nothing and reported success: the
notebook assigned ``ctrl.pipeline`` and ``ctrl.writer``, which Python accepted
as new attributes that the Controller never read.
"""

from __future__ import annotations

import os
import time

import pytest

import faro.core.controller as controller_module
from faro.core.controller import KEEP_ERRORS_PER_KIND, Analyzer, Controller
from faro.core.writers import TiffWriter
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_events, make_pipeline


class TestControllerAttributes:
    def test_pipeline_and_writer_are_readable(self, tmp_dir):
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        writer = TiffWriter(tmp_dir)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline, writer=writer)
        assert ctrl.pipeline is pipeline
        assert ctrl.writer is writer

    def test_assigning_pipeline_or_writer_raises(self, tmp_dir):
        ctrl = Controller(FakeMicroscope(CircleScene()), None)
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        with pytest.raises(AttributeError, match="new Controller"):
            ctrl.pipeline = pipeline
        with pytest.raises(AttributeError, match="new Controller"):
            ctrl.writer = TiffWriter(tmp_dir)
        assert ctrl.pipeline is None and ctrl.writer is None


class TestRunThatStoresNothing:
    def test_run_without_writer_and_pipeline_is_refused(self):
        ctrl = Controller(FakeMicroscope(CircleScene()), None)
        with pytest.raises(RuntimeError, match="store nothing"):
            ctrl.run_experiment(make_events(2), validate=False)
        with pytest.raises(RuntimeError, match="store nothing"):
            ctrl.load_experiment(make_events(2), validate=False)

    def test_run_with_only_a_writer_is_allowed(self, tmp_dir):
        ctrl = Controller(
            FakeMicroscope(CircleScene()), None, writer=TiffWriter(tmp_dir)
        )
        ctrl.run_experiment(make_events(2), validate=False).wait()
        ctrl.finish_experiment()
        assert ctrl.background_errors == []
        assert sorted(os.listdir(os.path.join(tmp_dir, "raw"))) == [
            "000_00000.tiff",
            "000_00001.tiff",
        ]


class TestErrorCapping:
    def test_only_the_first_errors_of_a_kind_are_kept(self):
        analyzer = Analyzer()  # no pipeline, no writer
        try:
            for i in range(250):
                try:
                    raise ValueError(f"bug {i}")
                except ValueError as e:
                    analyzer._record_background_error("pipeline", e)
            try:
                raise KeyError("other")
            except KeyError as e:
                analyzer._record_background_error("storage", e)

            kept = [(e.source, e.exc_type) for e in analyzer.background_errors]
            assert kept.count(("pipeline", "ValueError")) == KEEP_ERRORS_PER_KIND
            assert kept.count(("storage", "KeyError")) == 1
            assert analyzer.error_summary() == {
                ("pipeline", "ValueError"): 250,
                ("storage", "KeyError"): 1,
            }
        finally:
            analyzer.shutdown(wait=False)

    def test_controller_collects_the_counts(self, tmp_dir, monkeypatch):
        # 8 failing frames, with a lower limit than the default.
        monkeypatch.setattr(controller_module, "KEEP_ERRORS_PER_KIND", 5)
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))

        def broken_run(*args, **kwargs):
            raise ValueError("pipeline bug")

        monkeypatch.setattr(pipeline, "run", broken_run)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)
        ctrl.run_experiment(make_events(8), validate=False).wait()
        ctrl.finish_experiment()
        assert ctrl.error_summary()[("pipeline", "ValueError")] == 8
        assert len(ctrl.background_errors) == 5


class TestWaitIdle:
    def test_waits_for_a_frame_that_is_still_being_stored(self, tmp_dir, monkeypatch):
        """wait_idle returned while the storage worker still wrote the last frame.

        The queue was empty, but the frame was not handed to the pipeline yet,
        so a caller could read the tracks before the last frame was in them.
        """
        store = Analyzer._do_store

        def slow_store(self, *args, **kwargs):
            time.sleep(0.3)
            store(self, *args, **kwargs)

        monkeypatch.setattr(Analyzer, "_do_store", slow_store)
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)
        ctrl.run_experiment(make_events(3), validate=False).wait()
        assert ctrl._analyzer.wait_idle(timeout=10)
        assert ctrl._analyzer.stored_images == 3
        ctrl.finish_experiment()


class TestFrameCallbacks:
    def test_mda_that_does_not_start_leaves_no_callback(self, tmp_dir, monkeypatch):
        # A callback left connected would get every frame of the next run twice.
        mic = FakeMicroscope(CircleScene())
        ctrl = Controller(mic, None, writer=TiffWriter(tmp_dir))
        frame_ready = mic.mmc.mda.events.frameReady
        n_before = len(frame_ready)

        def busy(event_iter):
            raise ValueError(
                "Cannot start an MDA while the previous MDA is still running."
            )

        monkeypatch.setattr(mic, "run_mda", busy)
        try:
            with pytest.raises(ValueError, match="still running"):
                ctrl.run_experiment(make_events(2), validate=False).wait()
        finally:
            ctrl.finish_experiment()
        assert len(frame_ready) == n_before

        monkeypatch.undo()
        try:
            handle = ctrl.run_experiment(make_events(2), validate=False)
            handle.wait()
        finally:
            ctrl.finish_experiment()
        assert handle.status().n_frames_received == 2

    def test_connect_error_is_not_hidden(self, tmp_dir, monkeypatch):
        # On the Qt signal backend connect_frame raises before it connects,
        # and the Qt disconnect then raises for a callback that is not
        # connected.
        mic = FakeMicroscope(CircleScene())
        ctrl = Controller(mic, None, writer=TiffWriter(tmp_dir))

        def qt_connect(callback):
            raise RuntimeError("pymmcore-plus is using the Qt MDA signal backend")

        def qt_disconnect(callback):
            raise TypeError("'method' object is not connected")

        monkeypatch.setattr(mic, "connect_frame", qt_connect)
        monkeypatch.setattr(mic, "disconnect_frame", qt_disconnect)
        try:
            with pytest.raises(RuntimeError, match="Qt MDA signal backend"):
                ctrl.run_experiment(make_events(2), validate=False).wait()
        finally:
            ctrl.finish_experiment()
        assert ctrl._active_handle is None

    def test_incomplete_timepoint_is_dropped(self, tmp_dir, monkeypatch):
        mic = FakeMicroscope(CircleScene())
        ctrl = Controller(mic, None, writer=TiffWriter(tmp_dir))

        # The first snap (t=0, c=0) fails. pymmcore-plus logs it and goes on,
        # so only c=1 of t=0 arrives.
        real_snap = mic.mmc.snapImage
        n_snaps = 0

        def snap_fails_once(*args, **kwargs):
            nonlocal n_snaps
            n_snaps += 1
            if n_snaps == 1:
                raise RuntimeError("camera timeout")
            return real_snap(*args, **kwargs)

        monkeypatch.setattr(mic.mmc, "snapImage", snap_fails_once)
        try:
            # A lost frame must not stop the run: wait() raises on a fatal error.
            events = make_events(2, n_channels=2)
            ctrl.run_experiment(events, validate=False).wait()
            assert ctrl._frame_buffers == {}
        finally:
            ctrl.finish_experiment()
