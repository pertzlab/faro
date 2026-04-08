import numpy as np
from .base import Segmentator
import skimage
from napari_convpaint import conv_paint, conv_paint_utils


import matplotlib.pyplot as plt

"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class SegmentatorConvpaint(Segmentator):

    def __init__(self, model_path: str, min_cell_size=0, fill_holes_smaller_than=0):

        self.random_forest, self.model, self.model_param, self.model_state = (
            conv_paint.load_model(model_path)
        )
        self.min_cell_size = min_cell_size
        self.fill_holes_smaller_than = fill_holes_smaller_than

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Run the stardist model on data and do post-processing (remove small cells)
        """

        mean, std = conv_paint_utils.compute_image_stats(image)
        img_normed = conv_paint_utils.normalize_image(image, mean, std)
        labels = self.model.predict_image(
            img_normed, self.random_forest, self.model_param
        )
        labels = labels - 1
        labels = labels.astype(bool)
        if self.fill_holes_smaller_than > 0:
            labels = skimage.morphology.remove_small_holes(
                labels, area_threshold=self.fill_holes_smaller_than
            )
        labels = skimage.morphology.label(labels)
        if self.min_cell_size > 0:
            labels = skimage.morphology.remove_small_objects(
                labels, min_size=self.min_cell_size
            )
        return labels
