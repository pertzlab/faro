import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


class FeatureExtractor:
    """
    Base class for all feature extractors. Specific implementations should inherit
    from this class and override the extract_features method.
    """

    def extract_positions(self, labels: dict) -> pd.DataFrame:
        """Extract (label, x, y) positions from the segmentation mask.

        Uses self.used_mask to select which label image to use.

        Args:
            labels: dict mapping mask name to label image.

        Returns:
            DataFrame with columns: label, x, y
        """
        table = regionprops_table(
            labels[self.used_mask], properties=["label", "centroid"]
        )
        df = pd.DataFrame.from_dict(table)
        df = df.rename({"centroid-0": "x", "centroid-1": "y"}, axis="columns")
        return df

    def extract_features(self, labels: dict, image: np.ndarray, df_tracked=None, metadata=None):
        raise NotImplementedError("Subclasses should implement this!")
