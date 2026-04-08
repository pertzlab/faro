import pymmcore_plus
from faro.microscope.pymmcore import PyMMCoreMicroscope
import os


class MMDemo(PyMMCoreMicroscope):
    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False

    def __init__(
        self,
        micromanager_path="C:\\Program Files\\Micro-Manager-2.0",
    ):
        super().__init__()
        self.micromanager_path = micromanager_path
        pymmcore_plus.use_micromanager(self.micromanager_path)
        self.micromanager_config = os.path.join(
            self.micromanager_path, "MMConfig_demo.cfg"
        )

        self.mmc = pymmcore_plus.CMMCorePlus()
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.micromanager_config)
        self.mmc.setChannelGroup(channelGroup=self.CHANNEL_GROUP)

    def post_experiment(self):
        pass
