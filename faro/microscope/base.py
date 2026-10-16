from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterator
from threading import Thread

# Silence the FutureWarning from napari-micromanager importing deprecated pymmcore_plus.mda.handlers
warnings.filterwarnings(
    "ignore",
    message="The 'pymmcore_plus.mda.handlers' module is deprecated",
    category=FutureWarning,
)

# PYMM_SIGNALS_BACKEND is hard-set to 'psygnal' in faro/__init__.py before any
# submodule (including this one) is imported. A setdefault here would be a
# no-op anyway: pymmcore_widgets pre-empts to 'qt' whenever it imports first.

import numpy as np
from useq import MDAEvent

from faro.core.data_structures import Channel
from faro.core.dmd import DMD


def as_channel(channel) -> Channel:
    """Coerce a config name or a channel dict to a ``Channel``; pass Channels through."""
    if isinstance(channel, Channel):
        return channel
    if isinstance(channel, str):
        return Channel(config=channel)
    if isinstance(channel, dict):
        return Channel(**channel)
    raise TypeError(
        f"expected a Channel, a channel dict or a config name, got {type(channel).__name__}"
    )


class AbstractMicroscope:
    """Base class defining the microscope interface.

    The Controller depends only on this interface — it never touches
    pymmcore-plus directly.  Subclasses implement the four MDA callables
    plus optional ``resolve_group`` / ``resolve_power`` for channel
    resolution.
    """

    os.environ["QT_LOGGING_RULES"] = (
        "*.debug=false; *.warning=false"  # Fix to suppress PyQT warnings from napari-micromanager when running in a Jupyter notebook
    )

    dmd = None  # optional DMD device
    use_autofocus_event = False  # optional autofocus
    dmd_needs_to_be_waken = False  # optional DMD wake

    def __init__(self):
        self.dmd = None

    def init_scope(self):
        """Initialize the microscope scope."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    # ------------------------------------------------------------------
    # MDA interface — used by Controller
    # ------------------------------------------------------------------

    def run_mda(self, event_iter: Iterator[MDAEvent]) -> Thread:
        """Start MDA acquisition. Returns thread/handle."""
        raise NotImplementedError

    def connect_frame(self, callback: Callable[[np.ndarray, MDAEvent], None]) -> None:
        """Connect frameReady callback: callback(img, event)."""
        raise NotImplementedError

    def disconnect_frame(
        self, callback: Callable[[np.ndarray, MDAEvent], None]
    ) -> None:
        """Disconnect frameReady callback."""
        raise NotImplementedError

    def cancel_mda(self) -> None:
        """Cancel running MDA."""
        raise NotImplementedError

    def connect_sequence_canceled(self, callback: Callable[..., None]) -> None:
        """Connect a callback that fires when the MDA runner is cancelled.

        Fires for every runner cancel: faro's own :meth:`cancel_mda` and
        external ones, like the stop button of a napari-micromanager MDA
        widget. The Controller uses it to turn external cancels into
        ``handle.cancel()``. The base implementation does nothing.
        """

    def disconnect_sequence_canceled(self, callback: Callable[..., None]) -> None:
        """Disconnect a callback added with :meth:`connect_sequence_canceled`."""

    def resolve_group(self, config_name) -> str:
        """Return channel group for config name. Optional override."""
        return ""

    def resolve_power(self, channel):
        """Return (device, property, power) or None. Optional override."""
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_hardware(self, events) -> bool:
        """Validate events against hardware capabilities.

        Base implementation only runs the DMD calibration check; subclasses
        compose extra checks (channel configs, exposure limits, device
        properties, etc.) by calling ``super().validate_hardware(events)``.
        """
        return self._validate_dmd_calibration(events)

    def _validate_dmd_calibration(self, events) -> bool:
        """Warn if events contain stim but the DMD isn't calibrated.

        Skipped when the microscope has no DMD or the events have no
        stim channels — non-DMD setups and non-stim experiments don't
        need it. Without this check, the failure surfaces deep in the
        first stim event as ``DMD.affine_transform`` raising
        ``ValueError("DMD not calibrated...")``.
        """
        if self.dmd is None:
            return True
        if getattr(self.dmd, "affine", None) is not None:
            return True
        if not any(getattr(ev, "stim_channels", ()) for ev in events):
            return True
        warnings.warn(
            "DMD not calibrated (affine matrix is None) but events contain "
            "stim channels. Run mic.calibrate_dmd() before starting the "
            "experiment.",
            UserWarning,
        )
        return False

    # ------------------------------------------------------------------
    # DMD
    # ------------------------------------------------------------------

    def calibrate_dmd(self, calibration_channel):
        """Calibrate the DMD. Always runs the calibration when called.

        ``calibration_channel`` is a ``Channel``/``PowerChannel``, a channel
        dict (``{"config": ..., "exposure": ...}``) or a config name. Does
        nothing on a scope without a DMD.
        """
        if isinstance(self.dmd, DMD):
            self.dmd.calibrate(as_channel(calibration_channel))

    def post_experiment(self):
        """Post-process the experiment. Optional override."""
        pass

    def shutdown(self):
        """Release all hardware resources held by this microscope.

        Unlike :meth:`post_experiment` (which runs *between* experiments
        and may keep devices warm for the next run), ``shutdown`` signals
        that this microscope instance is being discarded. Subclasses
        should stop any background threads (DMD wakeup loops, etc.) and
        unload devices so that COM ports / SLM handles are released and
        the Python process can exit cleanly.

        Base implementation is a no-op; subclasses override as needed.
        """
        pass
