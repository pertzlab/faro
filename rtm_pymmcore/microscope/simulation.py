import threading

from rtm_pymmcore.microscope.base import AbstractMicroscope
from rtm_pymmcore.core.controller import Controller, ControllerSimulated, Analyzer


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


class UniMMCoreSimulation(AbstractMicroscope):

    def __init__(
        self,
        mmc, # pymmcore_plus CMMCorePlus instance
    ):
        super().__init__()
        self._experiment_thread: threading.Thread | None = None
        self.mmc = mmc
        self._patch_getROI()

    def _patch_getROI(self):
        """Work around pymmcore-plus bug: UniMMCore.getROI() has inverted
        condition for Python cameras, causing RuntimeError on super().getROI().
        Fall back to image dimensions when the native call fails."""
        original = self.mmc.getROI
        mmc = self.mmc

        def _safe_getROI(*args):
            try:
                return original(*args)
            except (RuntimeError, NotImplementedError):
                return [0, 0, mmc.getImageWidth(), mmc.getImageHeight()]

        self.mmc.getROI = _safe_getROI

    def run_experiment(self, events=None, *, df_acquire=None, stim_mode="current"):
        """Run the experiment in a background thread.

        Args:
            events: Iterable of RTMEvent (primary path).
            df_acquire: Legacy DataFrame (backwards compat). If both are None
                        and a positional arg is a DataFrame, treat it as df_acquire.
            stim_mode: ``"current"`` or ``"previous"`` — see Controller.run().

        Keeps the Qt event loop free so napari can display frames and
        the proxy signal relay can dispatch frameReady / sequenceFinished.
        """
        import pandas as pd

        # Backwards compat: positional DataFrame
        if events is not None and isinstance(events, pd.DataFrame):
            df_acquire = events
            events = None

        # Wait for any previous experiment to finish (keep Qt alive)
        if self._experiment_thread is not None and self._experiment_thread.is_alive():
            self.post_experiment()

        self.analyzer = Analyzer(self.pipeline)

        # Detect SLM device for optogenetic stimulation
        dmd = None
        try:
            slm_name = self.mmc.getSLMDevice()
            if slm_name:
                dmd = SimDMD(slm_name)
        except Exception:
            pass

        self.controller = Controller(
            self.analyzer, self.mmc, self.queue, dmd=dmd
        )

        run_kwargs = {"stim_mode": stim_mode}
        if events is not None:
            run_kwargs["events"] = events
        elif df_acquire is not None:
            run_kwargs["df_acquire"] = df_acquire

        self._experiment_thread = threading.Thread(
            target=self.controller.run,
            kwargs=run_kwargs,
            daemon=True,
        )
        self._experiment_thread.start()

    def post_experiment(self):
        """Wait for the experiment to finish.

        Unlike a plain thread.join(), this periodically processes Qt events
        so the viewer stays responsive (signal dispatches, canvas repaints).
        """
        if self._experiment_thread is None:
            return
        try:
            from qtpy.QtWidgets import QApplication
            app = QApplication.instance()
        except ImportError:
            app = None

        while self._experiment_thread.is_alive():
            if app is not None:
                app.processEvents()
            self._experiment_thread.join(timeout=0.05)
