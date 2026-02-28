from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from threading import Thread

import numpy as np
from useq import MDAEvent

from rtm_pymmcore.core.dmd import DMD


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

    dmd = None                      # optional DMD device
    use_autofocus_event = False     # optional autofocus
    dmd_needs_to_be_waken = False   # optional DMD wake

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

    def disconnect_frame(self, callback: Callable[[np.ndarray, MDAEvent], None]) -> None:
        """Disconnect frameReady callback."""
        raise NotImplementedError

    def cancel_mda(self) -> None:
        """Cancel running MDA."""
        raise NotImplementedError

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

        Base implementation is a no-op (returns True). Subclasses override
        to check channel configs, exposure limits, device properties, etc.
        """
        return True

    # ------------------------------------------------------------------
    # DMD
    # ------------------------------------------------------------------

    def calibrate_dmd(self):
        "Calibrate the DMD if it is not already calibrated." ""
        if isinstance(self.dmd, DMD) and self.dmd.affine is None:
            self.dmd.calibrate()

    def post_experiment(self):
        """Post-process the experiment. Optional override."""
        pass
