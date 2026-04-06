import numpy as np
import pandas as pd


class FeatureExtractorRef:
    """
    Base class for reference-acquisition feature extractors.
    Specific implementations should inherit from this class and override
    the extract_features method.
    """

    def __init__(self, used_mask, multi_timepoint=False):
        self.used_mask = used_mask
        self.multi_timepoint = multi_timepoint

    def extract_features(
        self,
        segmentation_results: dict,
        image: np.ndarray,
        df_tracked: pd.DataFrame,
        metadata: dict,
    ) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this!")


# Backwards-compat alias
FeatureExtractorOptoCheck = FeatureExtractorRef
