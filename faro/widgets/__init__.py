"""Optional UI widgets for visualising / controlling faro runs.

These imports are guarded so that headless deployments don't pay the Qt
import cost just from ``import faro``.
"""

from faro.widgets.experiment_status import ExperimentStatusWidget

__all__ = ["ExperimentStatusWidget"]
