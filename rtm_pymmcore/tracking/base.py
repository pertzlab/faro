import pandas as pd


class Tracker:
    """Base class for tracking algorithms. Subclasses must implement track_cells() method."""

    def track_cells(self) -> pd.DataFrame:
        raise NotImplementedError("Subclass must implement track_cells() method.")
