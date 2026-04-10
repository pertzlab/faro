import warnings

warnings.filterwarnings("ignore", message="Sparse invariant checks")

import os
import time

import numpy as np
from faro.segmentation.base import Segmentator
import skimage
from cellpose import models

_TIMING = bool(os.environ.get("FARO_CELLPOSE_TIMING"))


class CellposeV4(Segmentator):

    def __init__(
        self,
        custom_model_path=None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 50,
        gpu: bool = True,
    ):

        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self._segment_calls = 0

        t0 = time.perf_counter()
        if custom_model_path is None:
            self.model = models.CellposeModel(gpu=gpu)
        else:
            self.model = models.CellposeModel(
                pretrained_model=custom_model_path, gpu=gpu
            )
        if _TIMING:
            print(f"[timing] CellposeV4.__init__ model load: {time.perf_counter() - t0:.1f}s")

    def segment(self, image: np.ndarray) -> np.ndarray:
        self._segment_calls += 1
        t0 = time.perf_counter()

        masks, flows, styles = self.model.eval(
            image,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )

        if _TIMING and self._segment_calls <= 5:
            print(f"[timing] cellpose.segment call={self._segment_calls} t={time.perf_counter() - t0:.1f}s")

        if self.min_size > 0:
            # remove cells below threshold
            masks = skimage.morphology.remove_small_objects(
                masks, min_size=self.min_size, connectivity=1
            )
        return masks
