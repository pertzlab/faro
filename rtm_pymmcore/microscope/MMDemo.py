import pymmcore_plus
from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope
from rtm_pymmcore.controller import ControllerSimulated, Analyzer
import os


class MMDemo(AbstractMicroscope):
    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False

    def __init__(
        self,
        old_data_project_path: str,
        micromanager_path="C:\\Program Files\\Micro-Manager-2.0",
    ):
        super().__init__()
        self.micromanager_path = micromanager_path
        pymmcore_plus.use_micromanager(self.micromanager_path)
        self.micromanager_config = os.path.join(
            self.micromanager_path, "MMConfig_demo.cfg"
        )

        self.mmc = pymmcore_plus.CMMCorePlus()
        self.old_data_project_path = old_data_project_path
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.micromanager_config)
        self.mmc.setChannelGroup(channelGroup=self.CHANNEL_GROUP)

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        self.analyzer = Analyzer(self.pipeline)
        self.controller = ControllerSimulated(
            self.analyzer,
            self.mmc,
            self.queue,
            self.USE_AUTOFOCUS_EVENT,
            project_path=self.old_data_project_path,
        )
        pymmcore_plus.configure_logging(stderr_level="WARNING")
        self.controller.run(df_acquire=df_acquire)

    def post_experiment(self):
        """Post-process the experiment."""
        pass
