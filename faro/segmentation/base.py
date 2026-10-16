import inspect

import numpy as np
from skimage.measure import label

import skimage
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects as _sk_remove_small_objects
import pandas as pd


"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


# scikit-image renamed the ``remove_small_objects`` size argument in 0.26:
# ``min_size`` (drops objects with area < N) was deprecated in favour of
# ``max_size`` (drops objects with area <= max_size). Detect which the
# installed version accepts so callers work on both.
_RSO_ACCEPTS_MAX_SIZE = (
    "max_size" in inspect.signature(_sk_remove_small_objects).parameters
)


def remove_small_objects(label_image, min_size, connectivity=1):
    """Remove connected components smaller than ``min_size`` pixels.

    Version-tolerant shim over scikit-image's API change. Through 0.25,
    ``remove_small_objects(min_size=N)`` drops objects with area ``< N``. In
    0.26 ``min_size`` was deprecated in favour of ``max_size``, which drops
    objects with area ``<= max_size``. Passing ``max_size = min_size - 1``
    reproduces the original "area < min_size" threshold exactly, so every
    backend keeps identical behaviour on either version.
    """
    if _RSO_ACCEPTS_MAX_SIZE:
        return _sk_remove_small_objects(
            label_image, max_size=min_size - 1, connectivity=connectivity
        )
    return _sk_remove_small_objects(
        label_image, min_size=min_size, connectivity=connectivity
    )


class Segmentator:
    """
    Base class for all segmentators. Specific implementations should inherit
    from this class and override this method.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters:
        image (np.ndarray): The image to segment.

        Returns:
        np.ndarray: The segmented image.
        """
        raise NotImplementedError("Subclasses should implement this!")


class SegmentatorBinary(Segmentator):
    """
    Binary segmentator.

    This class implements a simple binary segmentation. It segments an image
    by setting all non-zero pixels to 1 and all zero pixels to 0.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        binary_image = image > 0
        label_image = label(binary_image)
        return label_image


class DummySegmentator(Segmentator):
    """
    Dummy segmentator.

    This class implements a dummy segmentator that returns a label image where
    the mask is the whole input image.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        return np.ones_like(image)


class OtsuSegmentator(Segmentator):
    """
    Otsu segmentator.

    This class implements a simple Otsu segmentation. It segments an image
    using Otsu's method to find the optimal threshold.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        from skimage.filters import threshold_otsu, gaussian
        from skimage.measure import label
        from skimage import filters

        image_gaussian = gaussian(image, sigma=1)
        thresh = threshold_otsu(image_gaussian)
        binary_image = image_gaussian > thresh
        label_image = label(binary_image)
        return label_image
