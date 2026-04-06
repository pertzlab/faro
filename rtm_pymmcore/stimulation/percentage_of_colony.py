import os
from .base_stimulation import Stim
import numpy as np
from napari_convpaint.conv_paint_model import ConvpaintModel
from scipy import ndimage as ndi
import skimage
from skimage.morphology import disk
import matplotlib.pyplot as plt


class StimColonyPercentage(Stim):
    """
    Segment the colony using Convpaint and stimulate a left-to-right fraction
    of it based on percentage_stimulated.
    """

    def __init__(
        self,
        path_to_model: str,
        model_name: str,
        channel_for_segmentation: int,
        percentage_stimulated: float = 0.5,  # 0..1 fraction by default
    ):
        super().__init__()
        self.path_to_model = path_to_model
        self.model_name = model_name
        self.percentage_stimulated = float(percentage_stimulated)
        self.cpm = ConvpaintModel(model_path=os.path.join(path_to_model, model_name))
        self.use_labels = False
        self.channel_for_segmentation = channel_for_segmentation

    def segment_and_cleanup(self, img) -> np.ndarray:
        img = skimage.exposure.adjust_gamma(img, gamma=0.1)
        seg = self.cpm.segment(img)
        # 1) convert to binary mask
        if seg.dtype == bool:
            spot_mask = seg
        else:
            seg_int = seg.astype(np.int64, copy=False)
            labels, counts = np.unique(seg_int, return_counts=True)

            # drop background label 0
            nz = labels != 1
            if not np.any(nz):
                raise ValueError("Segmentation contains only background (0).")

            largest_label = labels[nz][np.argmax(counts[nz])]
            spot_mask = seg_int == largest_label

        # 2) keep largest connected component
        lab, n = ndi.label(spot_mask)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            spot_mask = lab == sizes.argmax()

        # 3) fill holes
        spot_mask = ndi.binary_fill_holes(spot_mask)

        # 4) dilate by 10 px
        spot_mask = ndi.binary_dilation(spot_mask, structure=disk(10))

        return spot_mask.astype(bool)

    def get_stim_mask(self, label_images: dict, metadata: dict, img: np.array = None):
        img = img[self.channel_for_segmentation]
        mask = self.segment_and_cleanup(img)
        percentage_stimulated_metadata = metadata.get("percentage_stimulated", None)
        if (
            percentage_stimulated_metadata is None
            and self.percentage_stimulated is not None
        ):
            percentage_stimulated = self.percentage_stimulated
        else:
            percentage_stimulated = float(percentage_stimulated_metadata)

        # mask: 2D boolean array
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            raise ValueError("Mask is empty.")

        x_cut = np.quantile(xs, percentage_stimulated)
        spot_mask = mask & (np.arange(mask.shape[1])[None, :] <= x_cut)

        return spot_mask, None
