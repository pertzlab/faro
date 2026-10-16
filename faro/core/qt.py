"""Small Qt helpers that do nothing when no Qt application is running.

faro runs headless as well as next to napari, so Qt is optional. Both
helpers import Qt lazily and return at once without a ``QCoreApplication``.
"""

from contextlib import suppress


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
