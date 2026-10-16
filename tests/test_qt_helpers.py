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

from faro.core.qt import drain_posted_qt_events, pump_qt_events

REPO = Path(__file__).resolve().parents[1]


def test_helpers_do_nothing_without_qtpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "qtpy", None)  # import fails
    pump_qt_events()
    drain_posted_qt_events()


def test_helpers_do_nothing_without_app():
    QtCore = pytest.importorskip("qtpy.QtCore")
    if QtCore.QCoreApplication.instance() is not None:
        pytest.skip("a Qt application already exists in this process")
    pump_qt_events()
    drain_posted_qt_events()


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
