import os
from rtm_pymmcore.dmd import DMD
from queue import Queue
import pymmcore_plus
import psutil


class AbstractMicroscope:
    """Base class for Microscope Init"""

    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    os.environ["QT_LOGGING_RULES"] = (
        "*.debug=false; *.warning=false"  # Fix to suppress PyQT warnings from napari-micromanager when running in a Jupyter notebook
    )

    def __init__(self):
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
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

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def calibrate_dmd(self):
        "Calibrate the DMD if it is not already calibrated." ""
        if isinstance(self.dmd, DMD) and self.dmd.affine is None:
            self.dmd.calibrate()

    def post_experiment(self):
        """Post-process the experiment."""
        raise NotImplementedError("This method should be implemented in a subclass.")
