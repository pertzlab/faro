from .base import StimWithPipeline
import numpy as np


class CenterCircle(StimWithPipeline):
    """
    Stimulate a circle in the center of the field of view. (basic example for testing)
    """

    def get_stim_mask(
        self, label_images: dict, metadata: dict, img: np.ndarray = None,
        tracks=None,
    ) -> tuple[np.ndarray, object]:

        height = label_images['labels'].shape[0]
        width = label_images['labels'].shape[1]

        spot_mask = np.zeros_like(label_images['labels'], dtype=np.uint8)
        center_y = height // 2
        center_x = width // 2
        radius = min(height, width) // 8
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        spot_mask[mask] = 255

        return spot_mask.astype("uint8"), None
