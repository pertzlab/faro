from .base_stimulation import Stim
import numpy as np
import skimage
import math
from skimage.morphology import disk

# import skimage binary_dilation under a local name and provide a scipy fallback
from skimage.morphology import binary_dilation as skimage_binary_dilation
from scipy.ndimage import binary_dilation as scipy_binary_dilation


class PercentageDistanceEdgeStimulation(Stim):
    """
    Stimulate a percentage of the cell.

    This class implements a stimulation that stimulates a percentage of the cell area
    plus a configurable distance from the edge. The orientation is stored in df_tracked
    to maintain consistent directionality across frames.

    Parameters via metadata:
    - stim_cell_percentage (float): Percentage of cell area to stimulate (default: 0.3 = 30%)
    - stim_edge_distance (int): Distance from edge in pixels for stimulation expansion (default: 5)
    """

    def get_stim_mask(
        self,
        label_images,
        metadata: dict = None,
        img: np.array = None,
        df_tracked: np.ndarray = None,
    ) -> np.ndarray:
        label_image = label_images["labels"]
        h, w = label_image.shape
        light_map = np.zeros_like(label_image, dtype=bool)
        props = skimage.measure.regionprops(label_image)
        if metadata is None:
            metadata = {}

        # Get parameters from metadata
        percentage_of_stim = metadata.get("stim_cell_percentage", 0.3)
        edge_distance = metadata.get(
            "stim_edge_distance", 5
        )  # distance from edge in pixels

        selem = disk(int(edge_distance))

        # Get current FOV info (particles are unique per FOV)
        current_fov = metadata.get("fov", 0)

        try:
            for prop in props:
                label = prop.label

                # bounding box with padding to reduce computation to a small window
                minr, minc, maxr, maxc = prop.bbox
                pad = int(edge_distance) + 1
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

                # Try to get stored orientation from df_tracked, otherwise use current
                orientation = prop.orientation
                particle_id = None

                if df_tracked is not None and len(df_tracked) > 0:
                    # Find the particle ID for this label in the current FOV
                    label_mask = (df_tracked["label"] == label) & (
                        df_tracked["fov"] == current_fov
                    )

                    if label_mask.any():
                        # Get the most recent entry for this label to find its particle ID
                        particle_id = df_tracked.loc[label_mask, "particle"].iloc[-1]

                        # Check if this particle already has a stored orientation (within this FOV)
                        particle_mask = (df_tracked["particle"] == particle_id) & (
                            df_tracked["fov"] == current_fov
                        )
                        if "stim_orientation" in df_tracked.columns:
                            stored_orientations = df_tracked.loc[
                                particle_mask, "stim_orientation"
                            ].dropna()
                            if len(stored_orientations) > 0:
                                # Use the first stored orientation for this particle
                                orientation = stored_orientations.iloc[0]

                        # Store orientation for this particle if not already stored
                        if "stim_orientation" not in df_tracked.columns:
                            df_tracked["stim_orientation"] = np.nan

                        # Update the orientation for all entries of this particle in this FOV
                        df_tracked.loc[particle_mask, "stim_orientation"] = orientation

                # Calculate the extent along major axis based on percentage_of_stim
                # percentage_of_stim = 0.3 means stimulate 30% of cell area on one side
                # This translates to a linear extent along the major axis
                # For ellipse-like shapes: area_fraction ≈ linear_fraction
                extent = 0.5 - percentage_of_stim

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
