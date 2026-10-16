"""Status reporting + cancellation handle for asynchronous experiment runs.

``Controller.run_experiment`` returns a ``RunHandle``. The handle owns:

* the worker thread supervising the MDA,
* a cooperative cancellation event,
* an immutable ``RunStatus`` snapshot that the controller's threads update
  via :meth:`RunHandle.update`,
* a ``statusChanged`` :mod:`psygnal` signal that UI widgets (and any other
  observer) can subscribe to for live updates.

The handle is thread-safe: callers and the worker can both read ``status()``
and the worker can call ``update(...)`` without coordination from the
caller. Status updates emit ``statusChanged`` synchronously; with ``qtpy``
loaded :mod:`psygnal` routes the emission to listeners' threads through
Qt's queued connections, so a napari widget connected on the main thread
sees a queued slot call from the worker without extra plumbing.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal

from psygnal import Signal

if TYPE_CHECKING:
    pass


RunState = Literal[
    "pending", "running", "pausing", "paused", "waiting", "cancelling", "done", "error"
]


@dataclass(frozen=True)
class RunStatus:
    """Immutable snapshot of an experiment run."""

    state: RunState = "pending"
    # Latest RTMEvent index the event stream committed to the MDA.
    current_event_index: dict[str, int] | None = None
    current_fov: int | None = None
    # Counts. Note the distinct units:
    #  - *_events_* counts RTMEvents (one logical timepoint+FOV; expands
    #    into several MDAEvents -- one per imaging/ref channel + stim).
    #  - n_frames_received counts MDAEvents (individual channel snaps).
    # Widgets must compare like-with-like: progress is n_events_acquired
    # / n_events_total, NOT n_frames_received / n_events_total.
    n_events_total: int = 0  # how many RTMEvents the run was started with
    n_events_consumed: int = 0  # RTMEvents pulled by the event stream so far
    n_events_acquired: int = (
        0  # RTMEvents whose first frame has arrived (WaitEvents bump on completion)
    )
    n_frames_received: int = 0  # MDAEvent frames acknowledged via frameReady
    # Timing.
    started_at: float | None = (
        None  # time.monotonic() when the first frame began acquiring
    )
    finished_at: float | None = None  # time.monotonic() when the worker exited
    last_frame_wallclock: float | None = None
    # Lag: how late the *current RTMEvent* started acquiring vs its
    # scheduled min_start_time, in ms. Measured once per RTMEvent (on its
    # first frame), not per channel-frame. Positive == behind schedule.
    lag_ms: float | None = None
    # Seconds left on an active WaitEvent countdown. Non-None only while
    # state == "waiting"; cleared back to None when the wait ends.
    wait_remaining_s: float | None = None
    # Pipeline / storage backpressure visibility (best-effort).
    pipeline_inflight: int = 0
    storage_queue_depth: int = 0
    # A run can finish "successfully" while some stim events fired the
    # all-off fallback mask; this counter makes that impossible to miss.
    # The warning log carries the (t, p) locations.
    n_stim_fallbacks: int = 0
    # Errors. ``background_errors`` accumulates analyzer-side issues; ``fatal_error``
    # is set when the worker itself raises.
    background_errors: tuple[Any, ...] = field(default_factory=tuple)
    fatal_error: BaseException | None = None


class RunHandle:
    """Handle returned by ``Controller.run_experiment`` / ``continue_experiment``.

    Use ``wait()`` to block until the run finishes, ``cancel()`` to request a
    graceful stop, ``status()`` for a snapshot, or subscribe to
    ``statusChanged`` for live updates. ``statusChanged`` is a per-instance
    :mod:`psygnal` signal that emits the latest ``RunStatus`` after every
    update.
    """

    # psygnal class-level Signal is a descriptor; access via ``handle.statusChanged``
    # gives an instance-bound signal. Different handles have independent signals.
    statusChanged = Signal(RunStatus)

    def __init__(
        self,
        n_events_total: int = 0,
        events: list | None = None,
        *,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._status = RunStatus(state="pending", n_events_total=n_events_total)
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Hook run synchronously on the caller's thread the first time
        # cancel() is called, before it returns. The controller uses it
        # to wake an event stream parked in a stim-mask wait so cancellation
        # is prompt rather than bounded by the stim-mask timeout.
        self._on_cancel = on_cancel
        # Optional snapshot of the (sorted) RTMEvents this handle is driving.
        # Widgets use this to render per-event visualisations (e.g. an event
        # strip + FOV map) that need to know the full run plan up front.
        # ``None`` keeps backward-compat with callers that construct
        # RunHandle directly without an events list.
        self.events: list | None = list(events) if events is not None else None
        # True when the cancel came from outside faro, e.g. the stop
        # button of a napari-micromanager MDA widget calling
        # mmc.mda.cancel() directly. The controller sets this before it
        # forwards the cancel to this handle, so callers can tell the
        # two apart. The experiment stays open either way; call
        # finish_experiment() to tear it down.
        self.externally_cancelled: bool = False

    # -- public API ---------------------------------------------------------

    def status(self) -> RunStatus:
        """Return the current immutable status snapshot."""
        with self._lock:
            return self._status

    def wait(self, timeout: float | None = None) -> RunStatus:
        """Block until the worker thread finishes (or ``timeout`` elapses).

        Returns the final ``RunStatus``. Re-raises ``fatal_error`` if the
        worker crashed -- mirroring the previous synchronous-run behaviour.
        """
        if self._thread is not None:
            self._thread.join(timeout)
        status = self.status()
        if status.fatal_error is not None:
            raise status.fatal_error
        return status

    def cancel(self) -> None:
        """Request graceful cancellation. Idempotent. Does not block.

        Sets the cancel event the event stream polls between events; on the
        next poll the loop stops feeding new events, asks the MDA engine to
        abort the in-flight event, and exits. Use ``wait()`` afterwards to
        block until the worker actually stops.

        The event stream can also be parked deep inside a blocking stim-mask
        wait, where it cannot poll the cancel event. The ``on_cancel``
        hook (set by the controller) is invoked synchronously here to
        wake that wait, so cancellation takes effect immediately instead
        of after the stim-mask timeout.
        """
        first_cancel = not self._cancel_event.is_set()
        self._cancel_event.set()
        # A cancel during pause must also release the event stream's pause-wait.
        self._pause_event.clear()
        if first_cancel and self._on_cancel is not None:
            try:
                self._on_cancel()
            except Exception:
                traceback.print_exc()
        with self._lock:
            if self._status.state in ("running", "pausing", "paused"):
                self._status = replace(self._status, state="cancelling")
                new_status = self._status
            else:
                return
        self.statusChanged.emit(new_status)

    def pause(self) -> None:
        """Request a graceful pause. Idempotent. Does not block.

        Sets the pause event the event stream polls before pulling each new
        RTMEvent. The loop finishes feeding the current event, then halts
        at the next iteration -- the MDA engine drains whatever is already
        queued, and no further events are fed until :meth:`resume`. State
        goes ``running -> pausing`` here; the event stream flips it to
        ``paused`` once it actually stops.
        """
        if self._pause_event.is_set():
            return
        self._pause_event.set()
        with self._lock:
            if self._status.state == "running":
                self._status = replace(self._status, state="pausing")
                new_status = self._status
            else:
                return
        self.statusChanged.emit(new_status)

    def resume(self) -> None:
        """Resume a paused run. Idempotent. Does not block."""
        if not self._pause_event.is_set():
            return
        self._pause_event.clear()
        with self._lock:
            if self._status.state in ("pausing", "paused"):
                self._status = replace(self._status, state="running")
                new_status = self._status
            else:
                return
        self.statusChanged.emit(new_status)

    def is_running(self) -> bool:
        """True if the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def is_paused(self) -> bool:
        """True if a pause has been requested (pausing or paused)."""
        return self._pause_event.is_set()

    # -- worker-side helpers ------------------------------------------------

    @property
    def cancel_event(self) -> threading.Event:
        """The cooperative-cancel event the event stream polls. Worker-side."""
        return self._cancel_event

    @property
    def pause_event(self) -> threading.Event:
        """The cooperative-pause event the event stream polls. Worker-side."""
        return self._pause_event

    def update(self, **updates: Any) -> RunStatus:
        """Atomically apply ``updates`` to the status snapshot and emit.

        Called from worker / pipeline / storage threads. ``statusChanged``
        listeners on the main thread see queued-connection delivery via
        psygnal's Qt integration.
        """
        with self._lock:
            self._status = replace(self._status, **updates)
            new_status = self._status
        self.statusChanged.emit(new_status)
        return new_status
