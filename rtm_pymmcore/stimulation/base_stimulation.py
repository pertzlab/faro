import numpy as np
import numpy.typing as npt
from skimage.draw import disk
from skimage.measure import regionprops


# TODO return also labels_stim, list in which the stimulated cells are marked
class Stim:
    """
    Base class for all stimulators. Specific implementations should inherit
    from this class and override the get_stim_mask method.
    """

    def get_stim_mask(
        self, label_images: dict, metadata: dict, img: np.ndarray
    ) -> npt.NDArray[np.uint8]:
        """
        Parameters:
        label_image (np.ndarray): The label image to stimulate.

        Returns:
        np.ndarray: The stimulation mask.
        list: A list of labels that were stimulated.
        """
        raise NotImplementedError("Subclasses should implement this!")


class StimWholeFOV(Stim):
    """
    Stimulate the whole FOV.
    """

    def get_stim_mask(
        self, label_images: dict, metadata: dict = None, img: np.array = None
    ) -> npt.NDArray[np.uint8]:
        return np.ones((img.shape[-2], img.shape[-1]), dtype=np.uint8), None


class StimNothing(Stim):
    """Use when you don't want to stimulate. Returns empty stimulation mask."""

    def get_stim_mask(
        self, label_image: np.ndarray, metadata: dict = None, img: np.array = None
    ) -> npt.NDArray[np.uint8]:
        return np.zeros_like(label_image), [1, 2, 3, 4]  # some dummy values


class StimCircle(Stim):
    """
    Circle stimulator.

    This class implements a simple circle stimulation. It creates a stimulation mask
    by drawing a circle at the centroid of each labeled region in the label image.
    The radius of the circle and the x/y offset of the centroid can be parametrized.
    """

    def get_stim_mask(
        self, label_images: np.ndarray, metadata: dict, img: np.ndarray = None
    ) -> npt.NDArray[np.uint8]:

        offset_x = metadata.get("offset_x", 0)
        offset_y = metadata.get("offset_y", 0)
        radius = metadata.get("radius", 25)

        label_image = label_images["labels"]
        stim_mask = np.zeros_like(label_image, dtype=np.uint8)
        props = regionprops(label_image)
        labels_stim = []
        for prop in props:
            centroid = (prop.centroid[0] + offset_y, prop.centroid[1] + offset_x)
            rr, cc = disk(centroid, radius=radius)
            # Ensure the coordinates are within the image bounds
            rr = np.clip(rr, 0, stim_mask.shape[0] - 1)
            cc = np.clip(cc, 0, stim_mask.shape[1] - 1) 
            stim_mask[rr, cc] = 255
            labels_stim.append(prop.label)
        return stim_mask, labels_stim
