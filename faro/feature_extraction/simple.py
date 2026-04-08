import numpy as np
from skimage.measure import label

import skimage
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
import pandas as pd
from .base import FeatureExtractor


"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class SimpleFE(FeatureExtractor):
    def __init__(self, used_mask):
        self.used_mask = used_mask
        super().__init__()

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        table = skimage.measure.regionprops_table(
            labels[self.used_mask], properties=["label", "area"]
        )
        table = pd.DataFrame.from_dict(table)
        return table, None
