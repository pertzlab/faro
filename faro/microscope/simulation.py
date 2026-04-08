from faro.microscope.pymmcore import PyMMCoreMicroscope


class SimDMD:
    """Lightweight SLM wrapper for simulated microscopes.

    Camera and SLM share the same coordinate space so
    affine_transform is the identity (no calibration needed).
    """

    def __init__(self, name: str):
        self.name = name
        self.affine = True  # always "calibrated"

    def affine_transform(self, mask):
        return mask


class UniMMCoreSimulation(PyMMCoreMicroscope):

    def __init__(
        self,
        mmc, # pymmcore_plus CMMCorePlus instance
    ):
        super().__init__()
        self.mmc = mmc

    def init_scope(self):
        # Detect SLM device for optogenetic stimulation
        try:
            slm_name = self.mmc.getSLMDevice()
            if slm_name:
                self.dmd = SimDMD(slm_name)
        except Exception:
            pass

    def post_experiment(self):
        pass
