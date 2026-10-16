from faro.core.dmd import DMD
from faro.microscope.pymmcore import PyMMCoreMicroscope


class UniMMCoreSimulation(PyMMCoreMicroscope):
    """Simulated microscope behind a pymmcore-plus ``UniMMCore``.

    Works with any pure-Python core that exposes a camera, a channel group
    and optionally an SLM, e.g. ``vmteach.load_microscope(...)``. The SLM is
    wrapped in the same :class:`faro.core.dmd.DMD` as on the real scopes, so
    the calibration routine, the affine transform and the stim event path
    are exercised exactly as in an experiment: call ``calibrate_dmd`` before
    running stim events (the simulated projector is aligned by default, and
    ``sim.slm_affine`` models a misaligned one).
    """

    def __init__(
        self,
        mmc,  # pymmcore_plus CMMCorePlus instance
    ):
        super().__init__()
        self.mmc = mmc

    def init_scope(self):
        # Wrap the SLM device, if any, in faro's DMD (uncalibrated until
        # calibrate_dmd runs, like on hardware).
        try:
            slm_name = self.mmc.getSLMDevice()
        except Exception:
            slm_name = None
        if slm_name:
            self.dmd = DMD(self.mmc, resolve_power=self.resolve_power)

    def post_experiment(self):
        pass
