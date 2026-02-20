import numpy as np


class FeatureExtractor:
    """
    Base class for all segmentators. Specific implementations should inherit
    from this class and override this method.
    """

    def extract_features(self, labels: dict, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")
