import numpy as np
from .base_segmentation import Segmentator
import skimage
import imaging_server_kit as sk
import time

"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class SegmentatorImagingServerKit(Segmentator):

    def __init__(
        self, server: str, algorithm: str, model_param: dict = None, min_size: int = 0
    ):

        self.algorithm = algorithm
        self.model_param = model_param
        self.client = sk.Client(server)
        self.min_size = min_size

    def segment(self, img: np.ndarray) -> np.ndarray:
        """
        Run the an imagekit model on data and do post-processing (remove small cells)
        """
        max_attempts = 5
        for attempt in range(0, max_attempts):
            try:
                if self.model_param is None:
                    labels = self.client.run(img, algorithm=self.algorithm)[0].data
                else:
                    labels = self.client.run(
                        img, algorithm=self.algorithm, **self.model_param
                    )[0].data
            except Exception as e:
                print("Failed attempt ", attempt)
                if attempt == max_attempts:
                    print("Give up to connect to segmentation server")
                else:
                    time.sleep(0.5)
            else:
                break

        if self.min_size > 0:
            # remove cells below threshold
            labels = skimage.morphology.remove_small_objects(
                labels, min_size=self.min_size, connectivity=1
            )
        return labels
