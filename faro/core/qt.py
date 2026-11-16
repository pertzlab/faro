"""Small Qt helpers that do nothing when no Qt application is running.

faro runs headless as well as next to napari, so Qt is optional. The
helpers import Qt lazily and return at once without a ``QCoreApplication``.
"""

import logging
import threading
from contextlib import suppress

logger = logging.getLogger(__name__)


def pump_qt_events() -> None:
    """Process pending Qt events if a Qt app is running; no-op otherwise.

    Call it in a wait loop on the main thread, so napari keeps repainting
    while a worker thread does the work.
    """
    try:
        from qtpy.QtCore import QCoreApplication
    except Exception:
        return
    app = QCoreApplication.instance()
    if app is not None:
        app.processEvents()


def drain_posted_qt_events() -> None:
    """Deliver Qt events posted to the calling thread; no-op without Qt.

    GUI handlers running on an acquisition thread (psygnal backend) can
    create QObjects there, so queued Qt events accumulate on that thread
    with no event loop to deliver them. Windows delivers the leftovers
    during thread shutdown while holding the loader lock; blocking on the
    GIL there deadlocks the process against a painting main thread.
    Delivering the events here, where the GIL is taken normally, leaves
    nothing for the shutdown path. DeferredDelete events need their own
    pass because ``sendPostedEvents`` excludes them by default.

    Call it last on a worker thread, just before the thread ends.
    """
    try:
        from qtpy.QtCore import QCoreApplication, QEvent
    except Exception:
        return
    if QCoreApplication.instance() is None:
        return
    with suppress(Exception):
        QCoreApplication.sendPostedEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


# Signal instances whose Qt-object slots run on the main thread, by id. The
# instances are kept alive here, so an id cannot be reused by another signal.
_ROUTED_SIGNALS: dict[int, object] = {}
_CONNECT_PATCHED = False
_QUEUE_TIMER = None
# Signals whose slots keep running on the emitting thread. napari-micromanager
# reads the camera image in its imageSnapped slot; run later on the main
# thread, it reads the buffer after the next snap: a wrong preview, and a
# second read on cameras that hand out each frame only once (PVCAM, Andor).
_DIRECT_SIGNALS = frozenset({"imageSnapped"})


def deliver_gui_slots_on_main_thread(core, interval_ms: int = 5) -> bool:
    """Run the core-event slots of Qt objects on the main thread.

    faro forces pymmcore-plus's psygnal backend: every core and MDA event is
    delivered on the thread that emits it, so faro's acquisition callbacks
    run on the MDA thread without waiting for the GUI. But the widgets of
    pymmcore-widgets and napari-micromanager assume the Qt backend, which
    delivers on the GUI thread. With psygnal, their slots run on the MMCore
    notification thread and the MDA thread. Examples: the preset combo
    blocks its signals from two threads and stays blocked; a PropertyWidget
    writes old values back to the core; the MDA widget changes its position
    table off the GUI thread.

    After this call, a slot that belongs to a ``QObject`` and is connected
    to one of this core's signals runs on the main thread: at once when the
    event comes from the main thread, else queued and delivered by a timer
    every ``interval_ms``. Other slots, such as faro's callbacks, still run
    on the emitting thread. Only connections made after the call are
    changed, so call it before building the widgets. Exceptions in a queued
    slot are logged, and a slot whose widget was deleted is skipped.

    The approach follows ``vmteach.gui._deliver_gui_slots_on_main_thread``.

    Call it on the main thread, with a Qt application. Returns True if the
    slots of ``core`` are routed; False without Qt, off the main thread, or
    when ``core`` has no psygnal signals (e.g. Qt signals, which deliver on
    the GUI thread anyway).
    """
    try:
        from psygnal import SignalInstance
        from qtpy.QtCore import QCoreApplication, QObject
    except Exception:
        return False
    if QCoreApplication.instance() is None:
        return False
    if threading.current_thread() is not threading.main_thread():
        logger.warning("deliver_gui_slots_on_main_thread must run on the main thread.")
        return False

    n_signals = 0
    for group in (core.events, core.mda.events):
        for name in getattr(type(group), "_psygnal_signals", {}):
            if name in _DIRECT_SIGNALS:
                continue
            sig = getattr(group, name, None)
            if isinstance(sig, SignalInstance):
                _ROUTED_SIGNALS[id(sig)] = sig
                n_signals += 1
    if not n_signals:
        return False
    _start_queue_timer(interval_ms)
    _patch_connect(SignalInstance, QObject)
    return True


def _patch_connect(SignalInstance, QObject) -> None:
    """Make ``connect`` pass ``thread="main"`` for Qt slots of routed signals."""
    global _CONNECT_PATCHED
    if _CONNECT_PATCHED:
        return
    original = SignalInstance.connect

    def connect(self, slot=None, *, thread=None, **kwargs):
        if (
            thread is None
            and slot is not None
            and _ROUTED_SIGNALS.get(id(self)) is self
            and isinstance(getattr(slot, "__self__", None), QObject)
        ):
            thread = "main"
        return original(self, slot, thread=thread, **kwargs)

    connect.__doc__ = original.__doc__
    SignalInstance.connect = connect
    _CONNECT_PATCHED = True


def _start_queue_timer(interval_ms: int) -> None:
    """Deliver the slots queued for the main thread, every ``interval_ms``."""
    global _QUEUE_TIMER
    if _QUEUE_TIMER is not None:
        return
    from psygnal import EmitLoopError, emit_queued
    from qtpy.QtCore import QCoreApplication, QTimer

    def drain() -> None:
        # emit_queued stops at the first failing slot; go on with the rest.
        while True:
            try:
                emit_queued()
                return
            except EmitLoopError as exc:
                cause = exc.__cause__
                if isinstance(cause, ReferenceError) or (
                    isinstance(cause, RuntimeError) and "has been deleted" in str(cause)
                ):
                    continue  # the widget is gone, like a direct emission skips it
                logger.error(
                    "A GUI slot raised while handling a core event.", exc_info=cause
                )

    _QUEUE_TIMER = QTimer(QCoreApplication.instance())
    _QUEUE_TIMER.timeout.connect(drain)
    _QUEUE_TIMER.start(interval_ms)
