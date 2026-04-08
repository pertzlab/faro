import numpy as np
from faro.segmentation.base import Segmentator
import skimage
import csbdeep
from stardist.models import StarDist2D
import matplotlib.pyplot as plt

"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class SegmentatorStardist(Segmentator):

    def __init__(
        self,
        model: str = "2D_versatile_fluo",
        model_path: str = None,
        norm_min: float = 1,
        norm_max: float = 99,
        min_size: int = 0,
        prob_thresh=None,
    ):
        """
        Initialize the SegmentatorStardist object.

        Parameters:
        model_path (str): The path to the pre-trained model (or name of pretrained network). Defaults to '2D_versatile_fluo'
        norm_min (float): The minimum value for normalization. Defaults to 1.
        norm_max (float): The maximum value for normalization. Defaults to 99.
        min_size (int): The minimal object size. Defaults to 30. If 0, no filtering is performed.

        """
        if model_path is None:
            self.model = StarDist2D.from_pretrained(model)
        else:
            self.model = StarDist2D(None, name=model, basedir=model_path)

        self.norm_min = norm_min
        self.norm_max = norm_max
        self.min_size = min_size  # minimal object size
        self.prob_thresh = prob_thresh

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Run the stardist model on data and do post-processing (remove small cells)
        """
        img_normed = csbdeep.utils.normalize(image, self.norm_min, self.norm_max)
        if self.prob_thresh is None:
            labels, details = self.model.predict_instances(img_normed)
        else:
            labels, details = self.model.predict_instances(
                img_normed, prob_thresh=self.prob_thresh
            )

        if self.min_size > 0:
            # remove cells below threshold
            labels = skimage.morphology.remove_small_objects(
                labels, min_size=self.min_size, connectivity=1
            )
        return labels
