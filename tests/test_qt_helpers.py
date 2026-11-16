"""``faro.core.qt`` helpers and the Qt-event drain at the end of an MDA thread.

The helpers must do nothing without Qt. With a Qt application, an event
that a GUI slot posts on the MDA thread must be delivered on that thread
before it ends (else Windows can deadlock at thread exit, see
``drain_posted_qt_events``). The Qt test runs in a subprocess, so its
``QCoreApplication`` does not change the other tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from faro.core.qt import (
    deliver_gui_slots_on_main_thread,
    drain_posted_qt_events,
    pump_qt_events,
)

REPO = Path(__file__).resolve().parents[1]


def test_helpers_do_nothing_without_qtpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "qtpy", None)  # import fails
    pump_qt_events()
    drain_posted_qt_events()
    assert deliver_gui_slots_on_main_thread(object()) is False


def test_helpers_do_nothing_without_app():
    QtCore = pytest.importorskip("qtpy.QtCore")
    if QtCore.QCoreApplication.instance() is not None:
        pytest.skip("a Qt application already exists in this process")
    pump_qt_events()
    drain_posted_qt_events()
    assert deliver_gui_slots_on_main_thread(object()) is False


_DRAIN_SCRIPT = textwrap.dedent(
    """
    import threading

    from qtpy.QtCore import QCoreApplication, QEvent, QObject
    from useq import MDAEvent

    from tests.fake_microscope import FakeMicroscope
    from tests.fixtures import CircleScene

    app = QCoreApplication([])
    delivered = []

    class Receiver(QObject):
        def event(self, ev):
            if ev.type() == QEvent.Type.User:
                t = threading.current_thread()
                delivered.append((t.ident, t.is_alive()))
                return True
            return super().event(ev)

    receivers = []
    mda_thread_ids = []

    def on_started(*_):
        # like a GUI slot under the psygnal backend: runs on the MDA thread
        mda_thread_ids.append(threading.get_ident())
        r = Receiver()  # lives on the MDA thread
        receivers.append(r)
        QCoreApplication.postEvent(r, QEvent(QEvent.Type.User))

    mic = FakeMicroscope(CircleScene())
    mic.mmc.mda.events.sequenceStarted.connect(on_started)
    thread = mic.run_mda([MDAEvent()])
    thread.join(timeout=60)
    assert not thread.is_alive(), "MDA thread did not end"
    assert mda_thread_ids and mda_thread_ids[0] != threading.get_ident()
    assert delivered == [(mda_thread_ids[0], True)], delivered
    print("OK")
"""
)


def test_run_mda_drains_posted_qt_events():
    pytest.importorskip("qtpy.QtCore")
    result = subprocess.run(
        [sys.executable, "-c", _DRAIN_SCRIPT],
        cwd=REPO,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr


_ROUTE_SCRIPT = textwrap.dedent(
    """
    import logging
    import threading
    import time

    from psygnal import Signal
    from qtpy.QtCore import QCoreApplication, QEvent, QObject

    from faro.core.qt import deliver_gui_slots_on_main_thread
    from tests.fake_microscope import FakeMicroscope
    from tests.fixtures import CircleScene

    app = QCoreApplication([])
    core = FakeMicroscope(CircleScene()).mmc
    assert deliver_gui_slots_on_main_thread(core)
    MAIN = threading.main_thread()
    calls = {k: [] for k in ("widget", "plain", "snap", "mda", "after", "dead", "other")}
    messages = []

    class Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logging.getLogger("faro.core.qt").addHandler(Handler())

    def on_main():
        return threading.current_thread() is MAIN

    class Widget(QObject):
        def __init__(self, key="widget"):
            super().__init__()
            self.key = key

        def on_prop(self, dev, prop, value):
            if dev == "Dev":  # the fake core also sends events of its own
                calls[self.key].append(on_main())

        def on_px(self, *a):
            self.objectName()  # touches Qt, like a widget slot; raises once deleted
            calls[self.key].append(on_main())

        def on_other(self, *a):
            calls["other"].append(on_main())

        def on_snap(self, *a):
            calls["snap"].append(on_main())

        def on_mda(self, *a):
            calls["mda"].append(on_main())

        def boom(self, *a):
            raise ValueError("boom in a GUI slot")

        def after(self, *a):
            calls["after"].append(on_main())

    def wait_for(cond, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end and not cond():
            app.processEvents()
            time.sleep(0.005)
        return cond()

    def from_thread(fn):
        t = threading.Thread(target=fn)
        t.start()
        t.join()

    w = Widget()
    core.events.propertyChanged.connect(w.on_prop)
    core.events.propertyChanged.connect(
        lambda dev, *a: calls["plain"].append(on_main()) if dev == "Dev" else None
    )
    core.events.imageSnapped.connect(w.on_snap)
    core.mda.events.sequenceFinished.connect(w.on_mda)

    # emitted on the main thread: delivered at once
    core.events.propertyChanged.emit("Dev", "Prop", "1")
    assert calls["widget"] == [True] and calls["plain"] == [True], calls

    # emitted on another thread: the Qt slot waits for the main thread, a
    # plain function and imageSnapped run on the emitting thread
    def emit_all():
        core.events.propertyChanged.emit("Dev", "Prop", "2")
        core.events.imageSnapped.emit("Camera")
        core.mda.events.sequenceFinished.emit(None)

    from_thread(emit_all)
    assert calls["plain"] == [True, False], calls
    assert calls["snap"] == [False], calls
    assert calls["widget"] == [True], calls
    assert wait_for(lambda: calls["widget"] == [True, True] and calls["mda"] == [True]), calls

    # a queued slot that raises is logged; the next slot still runs
    core.events.configSet.connect(w.boom)
    core.events.configSet.connect(w.after)
    from_thread(lambda: core.events.configSet.emit("Group", "Preset"))
    assert wait_for(lambda: calls["after"] == [True]), calls
    assert any("GUI slot raised" in m for m in messages), messages

    # a slot whose widget was deleted before delivery is skipped
    dead = Widget("dead")
    core.events.pixelSizeChanged.connect(dead.on_px)
    from_thread(lambda: core.events.pixelSizeChanged.emit(0.5))
    dead.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    n_errors = sum("GUI slot raised" in m for m in messages)
    core.events.pixelSizeChanged.connect(w.after)
    from_thread(lambda: core.events.pixelSizeChanged.emit(0.6))
    assert wait_for(lambda: calls["after"] == [True, True]), calls
    assert calls["dead"] == [] and sum("GUI slot raised" in m for m in messages) == n_errors

    # psygnal signals that do not belong to the core are not changed
    class Other:
        changed = Signal(int)

    other = Other()
    keep = Widget("other")
    other.changed.connect(keep.on_other)
    from_thread(lambda: other.changed.emit(1))
    assert False in calls["other"], calls
    print("OK")
"""
)


def test_gui_slots_run_on_main_thread():
    pytest.importorskip("qtpy.QtCore")
    result = subprocess.run(
        [sys.executable, "-c", _ROUTE_SCRIPT],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout + result.stderr
    )
