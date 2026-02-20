import threading

from rtm_pymmcore.microscope.base import AbstractMicroscope
from rtm_pymmcore.core.controller import Controller, ControllerSimulated, Analyzer

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

    def run_experiment(self, df_acquire):
        """Run the experiment in a background thread.

        Keeps the Qt event loop free so napari can display frames and
        the proxy signal relay can dispatch frameReady / sequenceFinished.
        """
        # Wait for any previous experiment to finish (keep Qt alive)
        if self._experiment_thread is not None and self._experiment_thread.is_alive():
            self.post_experiment()

        self.analyzer = Analyzer(self.pipeline)
        self.controller = Controller(
            self.analyzer, self.mmc, self.queue
        )

        self._experiment_thread = threading.Thread(
            target=self.controller.run,
            kwargs={"df_acquire": df_acquire},
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
