from rtm_pymmcore.microscope.base import AbstractMicroscope


class PyMMCoreMicroscope(AbstractMicroscope):
    """Intermediate base for all pymmcore-plus-based microscopes.

    Subclasses must set ``self.mmc`` to a ``CMMCorePlus`` instance.

    Power properties (mapping channel config → light-source device/property)
    are auto-detected from the loaded Micro-Manager config.  Subclasses may
    set ``POWER_PROPERTIES`` to override or supplement the auto-detected
    values.
    """

    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    POWER_PROPERTIES: dict[str, tuple[str, str]] = {}

    def __init__(self):
        super().__init__()
        self.mmc = None  # subclasses must set this
        self._detected_power_properties: dict[str, tuple[str, str]] | None = None

    def detect_power_properties(self, group=None) -> dict[str, tuple[str, str]]:
        """Auto-detect power properties from the loaded Micro-Manager config.

        Scans for devices with ``*_Level`` properties (e.g. Spectra, LedDMD)
        and matches channel config presets to their LED color.

        Call this after ``mmc.loadSystemConfiguration()`` to populate the
        mapping.  Results are cached; call with ``group`` to restrict the scan.
        Manual ``POWER_PROPERTIES`` always take priority over auto-detected ones.
        """
        if self.mmc is None:
            return {}
        from rtm_pymmcore.core.utils import detect_power_properties
        detected = detect_power_properties(self.mmc, group=group)
        self._detected_power_properties = detected
        return detected

    def get_power_properties(self) -> dict[str, tuple[str, str]]:
        """Return merged power properties (auto-detected + manual overrides)."""
        detected = self._detected_power_properties or {}
        # Manual POWER_PROPERTIES override auto-detected ones
        return {**detected, **self.POWER_PROPERTIES}

    def validate_hardware(self, events) -> bool:
        if self.mmc is None:
            return True  # nothing to validate against
        # Auto-detect on first use if not yet done
        if self._detected_power_properties is None:
            self.detect_power_properties()
        from rtm_pymmcore.core.utils import validate_hardware
        return validate_hardware(events, self.mmc, power_properties=self.get_power_properties())
