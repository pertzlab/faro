import pymmcore_plus
from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope
from rtm_pymmcore.controller import Controller, Analyzer


class Jungfrau(AbstractMicroscope):
    MICROMANAGER_PATH = "D:\\Program Files\\Micro-Manager-2.0_n"
    MICROMANAGER_CONFIG = "E:\\pertzlab_mic_configs\\micromanager\\\Jungfrau\\TiFluoroJungfrau_w_TTL_DIGITALIO.cfg"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = True

    def __init__(self):
        super().__init__()
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        self.analyzer = Analyzer(self.pipeline)
        self.controller = Controller(
            self.analyzer, self.mmc, self.queue, self.USE_AUTOFOCUS_EVENT
        )
        pymmcore_plus.configure_logging(stderr_level="WARNING")
        self.controller.run(df_acquire)

    def post_experiment(self):
        """Post-process the experiment."""
