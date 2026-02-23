import os
from rtm_pymmcore.core.dmd import DMD
from queue import Queue


class AbstractMicroscope:
    """Base class for Microscope Init"""

    os.environ["QT_LOGGING_RULES"] = (
        "*.debug=false; *.warning=false"  # Fix to suppress PyQT warnings from napari-micromanager when running in a Jupyter notebook
    )

    def __init__(self):
        self.dmd = None
        self.queue = Queue()
        self.pipeline = None
        self.analyzer = None
        self.controller = None

    def init_scope(self):
        """Initialize the microscope scope."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def validate_events(self, events) -> bool:
        """Validate events against pipeline components and hardware.

        Combines :meth:`pipeline.validate_pipeline` (signatures + required
        metadata) with :meth:`validate_hardware` (channel configs + exposure/
        power limits).

        Returns True if **all** checks pass, False otherwise.
        """
        ok = True
        if self.pipeline is not None:
            ok = self.pipeline.validate_pipeline(events) and ok
        ok = self.validate_hardware(events) and ok
        return ok

    def validate_hardware(self, events) -> bool:
        """Validate events against hardware capabilities.

        Base implementation is a no-op (returns True). Subclasses override
        to check channel configs, exposure limits, device properties, etc.
        """
        return True

    def run_experiment(self, events=None, *, df_acquire=None, stim_mode="current"):
        """Run the experiment."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def calibrate_dmd(self):
        "Calibrate the DMD if it is not already calibrated." ""
        if isinstance(self.dmd, DMD) and self.dmd.affine is None:
            self.dmd.calibrate()

    def post_experiment(self):
        """Post-process the experiment."""
        raise NotImplementedError("This method should be implemented in a subclass.")
