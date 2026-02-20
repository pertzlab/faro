import numpy as np
from rtm_pymmcore.segmentation.base import Segmentator
import skimage
from cellpose import models


class SegmentorCellpose(Segmentator):

    def __init__(
        self,
        model: str = "cyto3",
        is_custom_model: bool = False,
        diameter=None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        dual_channels: bool = False,
        min_size: int = 50,
    ):

        self.is_custom_model = is_custom_model
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size

        if not is_custom_model:
            self.model = models.Cellpose(model_type=model, gpu=True)
        else:
            self.model = models.CellposeModel(model_type=model, gpu=True)
        if dual_channels:
            self.channels = [1, 2]
        else:
            self.channels = [0, 0]

    def segment(self, img: np.ndarray) -> np.ndarray:

        if self.is_custom_model:
            masks, flows, styles = self.model.eval(
                img,
                channels=self.channels,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                diameter=self.diameter,
            )
        else:
            # Use the default Cellpose model for segmentation
            masks, flows, styles, diams = self.model.eval(
                img,
                channels=self.channels,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                diameter=self.diameter,
            )

        if self.min_size > 0:
            # remove cells below threshold
            masks = skimage.morphology.remove_small_objects(
                masks, min_size=self.min_size, connectivity=1
            )
        return masks
