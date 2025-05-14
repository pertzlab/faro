import numpy as np
from rtm_pymmcore.segmentation.base_segmentation import Segmentator
import skimage
from cellpose import models

class CellposeV4(Segmentator):

    def __init__(
        self,
        diameter=None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 50,
    ):

        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size

        self.model = models.CellposeModel(gpu=True)


    def segment(self, img: np.ndarray) -> np.ndarray:

        masks, flows, styles = self.model.eval(
                img,
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
