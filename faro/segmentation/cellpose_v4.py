import warnings

warnings.filterwarnings("ignore", message="Sparse invariant checks")

import numpy as np
from faro.segmentation.base import Segmentator
import skimage
from cellpose import models


class CellposeV4(Segmentator):

    def __init__(
        self,
        custom_model_path=None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 50,
    ):

        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size

        if custom_model_path is None:
            self.model = models.CellposeModel(gpu=True)
        else:
            self.model = models.CellposeModel(
                pretrained_model=custom_model_path, gpu=True
            )

    def segment(self, image: np.ndarray) -> np.ndarray:

        masks, flows, styles = self.model.eval(
            image,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )

        if self.min_size > 0:
            # remove cells below threshold
            masks = skimage.morphology.remove_small_objects(
                masks, min_size=self.min_size, connectivity=1
            )
        return masks
