import pandas as pd


class Tracker:
    """Base class for tracking algorithms. Subclasses must implement track_cells() method."""

    required_metadata: set[str] = set()

    def track_cells(self, df_old, df_new, fov_state) -> pd.DataFrame:
        raise NotImplementedError("Subclass must implement track_cells() method.")
