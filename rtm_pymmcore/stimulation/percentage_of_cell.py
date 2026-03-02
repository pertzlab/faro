from .base import StimWithPipeline
import numpy as np
import skimage
import math
from skimage.morphology import disk

# import skimage binary_dilation under a local name and provide a scipy fallback
from skimage.morphology import binary_dilation as skimage_binary_dilation
from scipy.ndimage import binary_dilation as scipy_binary_dilation


class StimPercentageOfCell(StimWithPipeline):
    """
    Stimulate a percentage of the cell.

    This class implements a stimulation that stimulates a percentage of the cell.
    The percentage can be parametrized.
    """

    required_metadata = {"stim_cell_percentage"}

    def above_line(self, i, j, x2, y2, x3, y3):
        v1 = (x2 - x3, y2 - y3)
        v2 = (x2 - i, y2 - j)
        xp = v1[0] * v2[1] - v1[1] * v2[0]
        return xp > 0

    def get_stim_mask(
        self, label_images, metadata: dict = None, img: np.ndarray = None,
        tracks=None,
    ) -> np.ndarray:
        label_image = label_images["labels"]
        h, w = label_image.shape
        light_map = np.zeros_like(label_image, dtype=bool)
        props = skimage.measure.regionprops(label_image)
        if metadata is None:
            metadata = {}
        percentage_of_stim = metadata.get("stim_cell_percentage", 0.3)

        selem = disk(5)

        try:
            extent = 0.5 - percentage_of_stim

            for prop in props:
                label = prop.label

                # bounding box with padding to reduce computation to a small window
                minr, minc, maxr, maxc = prop.bbox
                pad = 6  # a bit larger than selem radius
                r0 = max(0, minr - pad)
                r1 = min(h, maxr + pad)
                c0 = max(0, minc - pad)
                c1 = min(w, maxc + pad)

                # subimages
                sub_labels = label_image[r0:r1, c0:c1]
                single_label_sub = sub_labels == label

                if not single_label_sub.any():
                    continue

                # get centroid and geometry in absolute coordinates
                y0, x0 = prop.centroid
                orientation = prop.orientation

                # point on major axis where cutoff starts
                x2 = x0 - math.sin(orientation) * extent * prop.major_axis_length
                y2 = y0 - math.cos(orientation) * extent * prop.major_axis_length

                # second point to define line segment (use minor_axis_length/2)
                length = 0.5 * prop.minor_axis_length
                x3 = x2 + (length * math.cos(-orientation))
                y3 = y2 + (length * math.sin(-orientation))

                # prepare grid for subwindow in absolute coordinates
                ys = np.arange(r0, r1)
                xs = np.arange(c0, c1)
                x_coords_sub, y_coords_sub = np.meshgrid(xs, ys)
                # x_coords_sub, y_coords_sub currently are (rows, cols) with x across cols, y across rows

                # compute cross product (vectorized) to create cutoff mask in subwindow
                v1_x = x3 - x2
                v1_y = y3 - y2
                v2_x = x3 - x_coords_sub
                v2_y = y3 - y_coords_sub
                cross_product = v1_x * v2_y - v1_y * v2_x
                cutoff_mask_sub = cross_product > 0

                # expand the labeled region locally in a version-compatible way
                try:
                    # newest skimage uses 'footprint'
                    expanded_sub = skimage_binary_dilation(
                        single_label_sub, footprint=selem
                    )
                except TypeError:
                    try:
                        # older skimage used 'selem'
                        expanded_sub = skimage_binary_dilation(
                            single_label_sub, selem=selem
                        )
                    except TypeError:
                        # fallback to scipy implementation (uses 'structure')
                        expanded_sub = scipy_binary_dilation(
                            single_label_sub, structure=selem
                        )

                stim_mask_sub = np.logical_and(cutoff_mask_sub, expanded_sub)

                # write back to global map
                light_map[r0:r1, c0:c1] = np.logical_or(
                    light_map[r0:r1, c0:c1], stim_mask_sub
                )

            return light_map.astype("uint8"), None
        except Exception as e:
            print(e)
            return np.zeros_like(label_image), None
