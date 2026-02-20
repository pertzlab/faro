import threading

from rtm_pymmcore.microscope.base import AbstractMicroscope
from rtm_pymmcore.core.controller import Controller, ControllerSimulated, Analyzer
from pymmcore_proxy import connect


class PymmcoreProxyMic(AbstractMicroscope):

    def __init__(
        self,
        url, # URL of the pymmcore-proxy server, e.g. "http://localhost:5600"
        old_data_project_path: str = None,
    ):
        super().__init__()
        self.url = url
        self._experiment_thread: threading.Thread | None = None
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc = connect(url=self.url)

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
