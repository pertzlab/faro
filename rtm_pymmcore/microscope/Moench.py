import pymmcore_plus
from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope
from rtm_pymmcore.controller import Controller, Analyzer
from rtm_pymmcore.dmd import DMD
from useq._mda_event import SLMImage
from useq import MDAEvent
import os
import time
import threading
import locale

os.environ["PYMM_PARALLEL_INIT"] = "0"


class KeepDMDAlive:
    def __init__(self, mmc):
        self.mmc = mmc
        self.thread = None
        self.last_wakeup = 0
        self.is_running = False

    def wakeup_dmd(self):
        self.mmc.setSLMExposure(self.mmc.getSLMDevice(), 200000.0)
        self.mmc.setSLMPixelsTo(self.mmc.getSLMDevice(), 255)
        self.mmc.displaySLMImage(self.mmc.getSLMDevice())

    def run(self):
        # Set locale to C/POSIX to ensure period as decimal separator
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            # If 'C' is not available, try 'en_US.UTF-8' or 'en_US'
            for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
                try:
                    locale.setlocale(locale.LC_NUMERIC, loc)
                    break
                except locale.Error:
                    continue

        self.is_running = True
        self.last_wakeup = 0
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_wakeup > 60:  # Wake up every minute
                self.wakeup_dmd()
                self.last_wakeup = current_time
            time.sleep(5)

    def stop(self):
        # Set locale to C/POSIX to ensure period as decimal separator
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            # If 'C' is not available, try 'en_US.UTF-8' or 'en_US'
            for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
                try:
                    locale.setlocale(locale.LC_NUMERIC, loc)
                    break
                except locale.Error:
                    continue

        self.is_running = False
        self.thread.join()
        time.sleep(5)
        self.mmc.setSLMExposure(self.mmc.getSLMDevice(), 100)
        self.mmc.displaySLMImage(self.mmc.getSLMDevice())


class Moench(AbstractMicroscope):
    MICROMANAGER_PATH = "C:\\micromanager_rtm_pymmcore\\Micro-Manager-2.0_api74"
    MICROMANAGER_CONFIG = "C:\\micromanager_rtm_pymmcore\\pertzlab_mic_configs\\micromanager\\Moench\\TiMoench_w_TTL_PrimeBSI.cfg"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = True
    DMD_NEEDS_TO_BE_WAKEN = True
    DMD_CHANNEL_GROUP = "TTL_ERK"
    DMD_CALIBRATION_PROFILE = {
        "channel_group": "TTL_ERK",
        "channel_config": "CyanStim",
        "device_name": "Spectra",
        "property_name": "Cyan_Level",
        "power": 10,
    }
    ROI_X = 0
    ROI_Y = 60
    ROI_WIDTH = 800
    ROI_HEIGHT = 800
    SET_ROI_REQUIRED = True

    def __init__(self, affine_calibration_matrix=None):
        super().__init__()
        pymmcore_plus.configure_logging(
            stderr_level="CRITICAL",
            file_level="CRITICAL",
            file="loglog.txt",
        )
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus(mm_path=self.MICROMANAGER_PATH)
        self.slm_dev = None
        self.slm_width = None
        self.slm_height = None

        self.affine_calibration_matrix = affine_calibration_matrix
        self.wakeup_dmd = None
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")
        self.slm_dev = self.mmc.getSLMDevice()
        self.slm_width = self.mmc.getSLMWidth(self.slm_dev)
        self.slm_height = self.mmc.getSLMHeight(self.slm_dev)
        self.dmd = DMD(
            self.mmc,
            self.DMD_CALIBRATION_PROFILE,
            affine_matrix=self.affine_calibration_matrix,
        )
        self.wakeup_dmd = KeepDMDAlive(self.mmc)
        self.wakeup_dmd.run()

        self.image_height = self.mmc.getImageHeight()
        self.image_width = self.mmc.getImageWidth()

    def calibrate_dmd(
        self,
        verbous=False,
        n_points=15,
        radius=4,
        exposure=25,
        marker_style="x",
        calibration_points_DMD=None,
    ):
        "Calibrate the DMD if it is not already calibrated." ""
        if self.dmd is not None and self.dmd.affine is None:
            self.wakeup_dmd.stop()
            self.dmd.calibrate(
                verbous=verbous,
                n_points=n_points,
                radius=radius,
                exposure=exposure,
                marker_style=marker_style,
                calibration_points_DMD=calibration_points_DMD,
            )
            self.wakeup_dmd.run()

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        self.wakeup_dmd.stop()
        time.sleep(2)
        self.analyzer = Analyzer(self.pipeline)
        self.controller = Controller(
            self.analyzer,
            self.mmc,
            self.queue,
            use_autofocus_event=self.USE_AUTOFOCUS_EVENT,
            dmd=self.dmd,
            dmd_needs_to_be_waken=self.DMD_NEEDS_TO_BE_WAKEN,
        )
        pymmcore_plus.configure_logging(stderr_level="WARNING")
        self.controller.run(df_acquire)

    def set_roi(self):
        self.mmc.clearROI()
        self.mmc.setROI(self.ROI_X, self.ROI_Y, self.ROI_WIDTH, self.ROI_HEIGHT)

    def post_experiment(self):
        """Post-process the experiment."""
        self.wakeup_dmd.run()
