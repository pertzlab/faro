"""Components specific to this experiment.

Put the segmentator, feature extractor or stimulator that only this
experiment needs here and import them in the notebook's pipeline cell:

    from custom import MySegmentator, MyFeatureExtractor, MyStimulator

Keep them out of notebook cells so the re-analysis notebook can import the
same classes and so they can be tested. Once a class is needed by a second
experiment, move it into faro and pin the commit that has it.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table

from faro.feature_extraction.base import FeatureExtractor
from faro.segmentation.base import Segmentator
from faro.stimulation.base import StimWithPipeline


class MySegmentator(Segmentator):
    """TODO: return a label image, 0 = background, one integer per cell."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: segment the image")


class MyFeatureExtractor(FeatureExtractor):
    """One row per cell. ``labels`` is a dict keyed by segmentation name."""

    def __init__(self, used_mask: str = "labels", channel: int = 0):
        self.used_mask = used_mask
        self.channel = channel
        super().__init__()

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        table = regionprops_table(
            labels[self.used_mask],
            intensity_image=image[self.channel],
            properties=["label", "area", "mean_intensity"],
        )
        return pd.DataFrame(table), None


class MyStimulator(StimWithPipeline):
    """Return a mask in camera pixels, 1 = light on.

    Receives the label images and the tracks, so it can treat cells
    differently by ``particle`` id or by anything the extractor measured.
    """

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        mask = np.zeros(labels.shape, dtype=np.uint8)
        for prop in regionprops(labels):
            mask[labels == prop.label] = 1  # TODO: your rule
        return mask, None
