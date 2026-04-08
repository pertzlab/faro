from faro.microscope.pymmcore import PyMMCoreMicroscope
from pymmcore_proxy import connect


class PymmcoreProxyMic(PyMMCoreMicroscope):

    def __init__(
        self,
        url, # URL of the pymmcore-proxy server, e.g. "http://localhost:5600"
    ):
        super().__init__()
        self.url = url
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc = connect(url=self.url)

    def post_experiment(self):
        pass
