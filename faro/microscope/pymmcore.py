import atexit
import contextlib
import threading
import weakref

from faro.core.qt import drain_posted_qt_events
from faro.microscope.base import AbstractMicroscope


class PyMMCoreMicroscope(AbstractMicroscope):
    """Intermediate base for all pymmcore-plus-based microscopes.

    Subclasses must set ``self.mmc`` to a ``CMMCorePlus`` instance.

    Implements the :class:`AbstractMicroscope` MDA interface by delegating
    to ``self.mmc`` (run_mda, frameReady signal, cancel, etc.).

    Power properties (mapping channel config -> light-source device/property)
    are declared explicitly per microscope via ``POWER_PROPERTIES``. A
    ``PowerChannel`` whose config is not mapped raises in
    :meth:`resolve_power` rather than silently dropping the requested power.

    On construction, an atexit hook is registered that cancels any running
    MDA and unloads every Micro-Manager device when the interpreter shuts
    down. Without this, device adapters can stay bound to the dying Python
    process and prevent the next process from acquiring the same
    configuration. Subclasses with extra teardown (background threads,
    serial ports, etc.) should override :meth:`_teardown_hardware` rather
    than re-register their own hooks.
    """

    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    POWER_PROPERTIES: dict[str, tuple[str, str]] = {}

    def __init__(self):
        super().__init__()
        self.mmc = None  # subclasses must set this
        self._current_group: str | None = None

        # Register cleanup via a weakref so the hook doesn't pin the
        # microscope instance. self.mmc is checked at fire time because
        # subclasses set it after super().__init__() returns.
        weak_self = weakref.ref(self)

        def _atexit_teardown() -> None:
            scope = weak_self()
            if scope is None:
                return
            scope._teardown_hardware()

        atexit.register(_atexit_teardown)
        self._atexit_teardown = _atexit_teardown

    def _teardown_hardware(self) -> None:
        """Release all hardware held by this microscope.

        Called from the atexit hook registered in :meth:`__init__`, and
        also reusable as an explicit teardown step from subclasses that
        expose a public ``shutdown`` API. Override in subclasses to add
        extra cleanup (stopping background threads, closing serial ports,
        etc.) — call ``super()._teardown_hardware()`` last so device
        unload happens after subclass-owned threads have stopped.

        Suppresses exceptions so a flaky device can't prevent the rest
        of the teardown (or, in the atexit path, the interpreter's
        finalization) from running.
        """
        if self.mmc is None:
            return
        with contextlib.suppress(Exception):
            self.mmc.mda.cancel()
        with contextlib.suppress(Exception):
            self.mmc.unloadAllDevices()

    # ------------------------------------------------------------------
    # MDA interface implementation
    # ------------------------------------------------------------------

    def run_mda(self, event_iter):
        """Run the MDA on a thread that drains its posted Qt events before exiting.

        Like ``CMMCorePlus.run_mda``, but the acquisition thread calls
        :func:`~faro.core.qt.drain_posted_qt_events` after the run. Under the
        psygnal backend, GUI slots run on this thread and can leave Qt events
        on it; on Windows these deadlock the process at thread exit (see the
        helper's docstring).
        """
        if self.mmc.mda.is_running():
            raise ValueError(
                "Cannot start an MDA while the previous MDA is still running."
            )

        def _run_and_drain() -> None:
            try:
                self.mmc.mda.run(event_iter)
            finally:
                drain_posted_qt_events()

        thread = threading.Thread(target=_run_and_drain)
        thread.start()
        return thread

    def connect_frame(self, callback):
        self._check_signal_backend()
        self.mmc.mda.events.frameReady.connect(callback)

    def disconnect_frame(self, callback):
        self.mmc.mda.events.frameReady.disconnect(callback)

    def cancel_mda(self):
        self.mmc.mda.cancel()

    def connect_sequence_canceled(self, callback):
        self.mmc.mda.events.sequenceCanceled.connect(callback)

    def disconnect_sequence_canceled(self, callback):
        self.mmc.mda.events.sequenceCanceled.disconnect(callback)

    def resolve_group(self, config_name: str) -> str:
        """Return the channel group for *config_name*, auto-detecting if needed."""
        if self._current_group is None:
            self._current_group = self.mmc.getChannelGroup()
        if self._current_group:
            return self._current_group
        # getChannelGroup() was empty — find a group containing this preset
        for group in self.mmc.getAvailableConfigGroups():
            if config_name in self.mmc.getAvailableConfigs(group):
                self._current_group = group
                return group
        return ""

    def resolve_power(self, channel):
        """Return (device, property, power) for a PowerChannel, or None.

        Returns None only when the channel carries no power (a plain Channel,
        or a PowerChannel with ``power`` unset). When ``power`` IS set but no
        device/property mapping can be resolved, this raises instead of
        silently returning None — a silent None means the requested power is
        never pushed to the hardware and the light source stays at whatever
        value it happened to hold, which is a near-invisible failure.
        """
        power = getattr(channel, "power", None)
        if power is None:
            return None
        mapping = self.get_power_properties().get(channel.config)
        if mapping is None:
            known = sorted(self.get_power_properties())
            raise ValueError(
                f"PowerChannel {channel.config!r} sets power={power}, but no "
                f"power-property mapping resolves for it. It was not "
                f"auto-detected from the config and is not listed in "
                f"{type(self).__name__}.POWER_PROPERTIES. The requested power "
                f"would be silently ignored. Add an explicit mapping, e.g.\n"
                f"    POWER_PROPERTIES = {{..., {channel.config!r}: "
                f"('<device>', '<Color>_Level')}}\n"
                f"Currently mapped channels: {known or '(none)'}."
            )
        device_name, property_name = mapping
        return (device_name, property_name, power)

    # ------------------------------------------------------------------
    # Power property management
    # ------------------------------------------------------------------

    def get_power_properties(self) -> dict[str, tuple[str, str]]:
        """Return the microscope's declared config -> (device, property) map.

        Mappings are declared explicitly on the subclass via
        ``POWER_PROPERTIES``. There is no auto-detection: the LED-selection
        wiring differs per config (color-named presets vs numeric TTL states),
        so inferring it silently caused requested powers to be dropped without
        warning.
        """
        return dict(self.POWER_PROPERTIES)

    # ------------------------------------------------------------------
    # Internal: signal-backend safety net
    # ------------------------------------------------------------------

    def _check_signal_backend(self) -> None:
        """Fail loud if pymmcore-plus is using the Qt MDA signal backend.

        faro's async pipeline runs frameReady on the engine thread; the Qt
        backend routes it through queued delivery to the main thread instead,
        and the controller's pipeline silently never sees the frames. faro
        sets ``PYMM_SIGNALS_BACKEND='psygnal'`` from ``faro/__init__.py`` so
        any MDARunner constructed after that point is psygnal-backed -- but
        if a CMMCorePlus.mda was already accessed before ``import faro``
        (typically because ``import napari_micromanager`` ran first and its
        transitive ``import pymmcore_widgets`` set the env to ``'qt'``), the
        runner is locked Qt-backed and faro's override is too late. Detect
        that here so the user gets a clear remediation rather than silent
        frame loss.

        Accessing ``self.mmc.mda`` lazily constructs the runner if it hasn't
        been already, so this also doubles as a forcing function: a runner
        built right here picks up the now-correct env.
        """
        if self.mmc is None:
            return
        sig_type = type(self.mmc.mda.events).__name__
        if sig_type == "QMDASignaler":
            raise RuntimeError(
                "pymmcore-plus is using the Qt MDA signal backend "
                f"({sig_type}), but faro's async controller pipeline requires "
                "the synchronous psygnal backend so frameReady runs on the "
                "engine thread regardless of the main thread's state. With "
                "Qt-backed signals frame callbacks are queued to the main "
                "thread and the pipeline never sees a frame.\n\n"
                "Cause: a pymmcore-plus MDA runner was constructed before "
                "PYMM_SIGNALS_BACKEND was set to 'psygnal'. faro/__init__.py "
                "sets this as early as it can, but napari-micromanager "
                "(via pymmcore_widgets) sets it to 'qt' first whenever it "
                "imports before any faro import.\n\n"
                "Fix: ensure `import faro` (or any `from faro...`) runs "
                "before `import napari_micromanager` (or `import "
                "pymmcore_widgets`). Or, as the very first line of your "
                "notebook/script (above every import):\n\n"
                "    import os; os.environ['PYMM_SIGNALS_BACKEND'] = 'psygnal'"
            )

    def validate_hardware(self, events) -> bool:
        # Materialize once so both the base and util checks can iterate.
        events = list(events)
        ok = super().validate_hardware(events)
        if self.mmc is None:
            return ok  # nothing else to validate against
        from faro.core.utils import validate_hardware

        ok_mmc = validate_hardware(
            events, self.mmc, power_properties=self.get_power_properties()
        )
        return ok and ok_mmc
