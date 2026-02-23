import threading
import time
import requests
from rtm_pymmcore.microscope.pymmcore import PyMMCoreMicroscope
import pymmcore_plus
from useq._mda_event import SLMImage
from useq import MDAEvent
from rtm_pymmcore.core.dmd import DMD
from rtm_pymmcore.core.controller import Controller, Analyzer


class WakeUpLaser:
    def __init__(self, lumencore_ip="192.168.201.200"):
        self.ip = lumencore_ip
        self.last_wakeup = 0
        self.is_running = False
        self.thread = None

    def wakeup_laser(self):
        url = f"http://{self.ip}/service/?command=WAKEUP"
        requests.get(url, timeout=5)

    def run(self, wait_for_warmup=True):
        self.is_running = True
        self.thread = threading.Thread(target=self._keep_alive)
        self.thread.start()
        if wait_for_warmup:
            time.sleep(15)

    def _keep_alive(self):
        while self.is_running:
            if time.time() - self.last_wakeup > 60:
                self.wakeup_laser()
                self.last_wakeup = time.time()
            time.sleep(3)

    def stop(self):
        self.is_running = False
        self.thread.join()


class Niesen(PyMMCoreMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    MICROMANAGER_CONFIG = "E:\\pertzlab_mic_configs\\micromanager\\Niesen\\Ti2CicercoConfig_w_DMD_w_TTL.cfg"
    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False
    DMD_CHANNEL_GROUP = "WF_DMD"
    POWER_PROPERTIES = {
        "CyanStim": ("LedDMD", "Cyan_Level"),
    }
    DMD_CALIBRATION_PROFILE = {
        "channel_group": "WF_DMD",
        "channel_config": "CyanStim",
        "device_name": "LedDMD",
        "property_name": "Cyan_Level",
        "power": 100,
    }

    def __init__(self, affine_calibration_matrix=None, fast_init=False):
        super().__init__()
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()
        self.wl = WakeUpLaser()
        self.wl.wakeup_laser()
        if not fast_init:
            time.sleep(10)
        self.init_scope()
        self.dmd = DMD(
            self.mmc,
            self.DMD_CALIBRATION_PROFILE,
            affine_matrix=affine_calibration_matrix,
        )
        self.slm_dev = None
        self.slm_width = None
        self.slm_height = None

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.wl.wakeup_laser()
        self.mmc.setConfig(groupName="System", configName="Startup")
        self.slm_dev = self.mmc.getSLMDevice()
        self.slm_width = self.mmc.getSLMWidth(self.slm_dev)
        self.slm_height = self.mmc.getSLMHeight(self.slm_dev)
        self.mmc.setSLMPixelsTo(self.slm_dev, 255)
        self.mmc.displaySLMImage(self.slm_dev)
        self.mmc.setChannelGroup(channelGroup=self.DMD_CHANNEL_GROUP)

    def calibrate_dmd(
        self,
        verbose=False,
        n_points=15,
        radius=4,
        exposure=25,
        marker_style="x",
        calibration_points_DMD=None,
    ):
        "Calibrate the DMD if it is not already calibrated." ""
        if self.dmd is not None and self.dmd.affine is None:
            self.dmd.calibrate(
                verbose=verbose,
                n_points=n_points,
                radius=radius,
                exposure=exposure,
                marker_style=marker_style,
                calibration_points_DMD=calibration_points_DMD,
            )

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        pymmcore_plus.configure_logging(stderr_level="WARNING")
        self.wl.run(wait_for_warmup=False)
        self.analyzer = Analyzer(self.pipeline)
        self.controller = Controller(
            self.analyzer, self.mmc, self.queue, self.USE_AUTOFOCUS_EVENT,
            dmd=self.dmd, power_properties=self.get_power_properties(),
        )
        self.controller.run(df_acquire)

    def post_experiment(self):
        """Post-process the experiment."""
        self.wl.stop()
