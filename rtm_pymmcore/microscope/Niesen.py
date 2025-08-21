import threading
import time
import pymmcore_plus
import requests

from useq._mda_event import SLMImage
from useq import MDAEvent
from abstract_microscope import AbstractMicroscope
from rtm_pymmcore.dmd import DMD


class WakeUpLaser:
    def __init__(self, lumencore_ip="192.168.201.200"):
        self.ip = lumencore_ip
        self.last_wakeup = 0
        self.is_running = False
        self.thread = None

    def wakeup_laser(self):
        url = f"http://{self.ip}/service/?command=WAKEUP"
        requests.get(url, timeout=5)

    def run(self, wait_for_warmup=False):
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


class Niesen(AbstractMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    MICROMANAGER_CONFIG = "E:\\pertzlab_mic_configs\\micromanager\\Niesen\\Ti2CicercoConfig_w_DMD_w_TTL.cfg"
    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False
    DMD_CHANNEL_GROUP = "WF_DMD"
    DMD_CALIBRATION_PROFILE = {
        "channel_group": "WF_DMD",
        "channel_config": "CyanStim",
        "device_name": "LedDMD",
        "property_name": "Cyan_Level",
        "power": 100,
    }

    def __init__(self, affine_calibration_matrix=None):
        super().__init__()
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()
        self.wl = WakeUpLaser()
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
        event_slm_on = MDAEvent(slm_image=SLMImage(data=True))
        self.mmc.mda.run([event_slm_on])
        self.mmc.setROI(150, 150, 1900, 1900)
        self.mmc.setChannelGroup(channelGroup=self.DMD_CHANNEL_GROUP)

    def calibrate_dmd(self):
        "Calibrate the DMD if it is not already calibrated." ""
        if self.dmd is not None and self.dmd.affine is None:
            self.dmd.calibrate()

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        pymmcore_plus.configure_logging(stderr_level="WARNING")
        self.wl.run(wait_for_warmup=True)

    def post_experiment(self):
        """Post-process the experiment."""
        self.wl.stop()
