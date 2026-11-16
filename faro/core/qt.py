"""Small Qt helpers that do nothing when no Qt application is running.

faro runs headless as well as next to napari, so Qt is optional. The
helpers import Qt lazily and return at once without a ``QCoreApplication``.
"""

import logging
import os
import threading
import traceback
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
_PROPERTY_SIGNAL_PATCHED = False
_QUEUE_TIMER = None
# Set while a devicePropertyChanged(...).connect call for a Qt object runs.
_FORCE_MAIN = threading.local()
# Signals whose slots keep running on the emitting thread. napari-micromanager
# reads the camera image in its imageSnapped slot; run later on the main
# thread, it reads the buffer after the next snap: a wrong preview, and a
# second read on cameras that hand out each frame only once (PVCAM, Andor).
_DIRECT_SIGNALS = frozenset({"imageSnapped"})
# How PyQt and PySide report a call into a deleted Qt object.
_DELETED_MESSAGES = ("has been deleted", "already deleted")


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
    event comes from the main thread, else queued and delivered by
    :func:`start_main_thread_queue` every ``interval_ms``. This includes
    ``core.events.devicePropertyChanged(...).connect(slot)``. Other slots,
    such as faro's callbacks, still run on the emitting thread. Only
    connections made after the call are changed, so call it before building
    the widgets. Not covered: slots that are plain functions, lambdas or
    ``functools.partial`` objects, and group-level connections.

    The approach follows ``vmteach.gui._deliver_gui_slots_on_main_thread``.
    Do not use both on the same core.

    Call it on the main thread, with a Qt application. Returns True if the
    slots of ``core`` are routed; False without Qt, off the main thread, or
    when ``core`` has no psygnal signals (e.g. Qt signals, which deliver on
    the GUI thread anyway, or a remote core).
    """
    try:
        from psygnal import SignalGroup, SignalInstance
        from qtpy.QtCore import QCoreApplication, QObject
    except Exception:
        return False
    if QCoreApplication.instance() is None:
        return False
    if threading.current_thread() is not threading.main_thread():
        logger.warning("deliver_gui_slots_on_main_thread must run on the main thread.")
        return False

    mda = getattr(core, "mda", None)
    groups = [getattr(core, "events", None), getattr(mda, "events", None)]
    groups = [g for g in groups if isinstance(g, SignalGroup)]
    if not groups:
        return False
    n_signals = 0
    for group in groups:
        for name in group:
            if name in _DIRECT_SIGNALS:
                continue
            sig = group[name]
            if isinstance(sig, SignalInstance):
                _ROUTED_SIGNALS[id(sig)] = sig
                n_signals += 1
    if not n_signals:
        logger.warning(
            "No signals found on the core's psygnal events; GUI slots still "
            "run on the emitting thread."
        )
        return False
    start_main_thread_queue(interval_ms)
    _patch_connect(SignalInstance, QObject)
    _patch_property_signal(QObject)
    return True


def start_main_thread_queue(interval_ms: int = 5) -> None:
    """Deliver the callbacks psygnal queued for the main thread.

    psygnal queues a callback connected with ``thread="main"`` when the
    signal is emitted on another thread. A QTimer on the main thread delivers
    them every ``interval_ms``. An exception in a callback is logged, and a
    callback of a deleted Qt object is skipped. Use this instead of
    ``psygnal.qt.start_emitting_from_queue``, whose timer drains the same
    queue but re-raises exceptions into Qt (in a plain PyQt app that aborts
    the process); its main-thread timer is stopped here.

    Call it on the main thread, with a Qt application. It is idempotent; after
    the QApplication was replaced, it starts a new timer.
    """
    global _QUEUE_TIMER
    try:
        from qtpy.QtCore import QCoreApplication, QTimer
    except Exception:
        return
    app = QCoreApplication.instance()
    if app is None:
        return
    if threading.current_thread() is not threading.main_thread():
        logger.warning("start_main_thread_queue must run on the main thread.")
        return
    if _timer_alive(_QUEUE_TIMER, app):
        return
    with suppress(Exception):
        from psygnal.qt import stop_emitting_from_queue

        stop_emitting_from_queue()
    timer = QTimer(app)
    timer.timeout.connect(_drain_main_thread_queue)
    timer.start(interval_ms)
    _QUEUE_TIMER = timer


def _timer_alive(timer, app) -> bool:
    if timer is None:
        return False
    try:
        return timer.parent() is app and timer.isActive()
    except RuntimeError:  # deleted together with an earlier QApplication
        return False


def _drain_main_thread_queue() -> None:
    from psygnal import EmitLoopError, emit_queued

    # emit_queued stops at the first failing callback; go on with the rest.
    while True:
        try:
            emit_queued()
            return
        except EmitLoopError as exc:
            cause = exc.__cause__ or exc
            if _is_deleted_receiver(cause):
                continue  # the widget is gone, like a direct emission skips it
            logger.error(
                "A GUI slot raised while handling a core event (%s).",
                _where(cause),
                exc_info=cause,
            )


def _is_deleted_receiver(exc: BaseException) -> bool:
    if isinstance(exc, ReferenceError):
        return True
    return isinstance(exc, RuntimeError) and any(
        m in str(exc) for m in _DELETED_MESSAGES
    )


def _where(exc: BaseException) -> str:
    """Name the slot that raised and the line that raised, outside psygnal."""
    frames = [
        f
        for f in traceback.extract_tb(exc.__traceback__)
        if "psygnal" not in f.filename.replace("\\", "/").split("/")
    ]
    if not frames:
        return type(exc).__name__
    first, last = frames[0], frames[-1]
    where = f"{first.name} in {os.path.basename(first.filename)}:{first.lineno}"
    if last is not first:
        where += (
            f", raised in {last.name} ({os.path.basename(last.filename)}:{last.lineno})"
        )
    return where


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
            and (
                getattr(_FORCE_MAIN, "on", False)
                or isinstance(getattr(slot, "__self__", None), QObject)
            )
        ):
            thread = "main"
        return original(self, slot, thread=thread, **kwargs)

    connect.__doc__ = original.__doc__
    SignalInstance.connect = connect
    _CONNECT_PATCHED = True


def _patch_property_signal(QObject) -> None:
    """Route ``devicePropertyChanged(...).connect(slot)`` for Qt slots too.

    pymmcore-plus connects a wrapper function to ``propertyChanged`` there,
    so the connection has no ``__self__`` to check. Used, for example, by
    the camera exposure widget of pymmcore-widgets.
    """
    global _PROPERTY_SIGNAL_PATCHED
    if _PROPERTY_SIGNAL_PATCHED:
        return
    try:
        from pymmcore_plus.core.events._prop_event_mixin import _PropertySignal
    except Exception:
        logger.debug("pymmcore-plus has no _PropertySignal; not patched.")
        return
    original = _PropertySignal.connect

    def connect(self, callback):
        if not isinstance(getattr(callback, "__self__", None), QObject):
            return original(self, callback)
        _FORCE_MAIN.on = True
        try:
            return original(self, callback)
        finally:
            _FORCE_MAIN.on = False

    connect.__doc__ = original.__doc__
    _PropertySignal.connect = connect
    _PROPERTY_SIGNAL_PATCHED = True
