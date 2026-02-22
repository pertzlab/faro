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

    required_metadata: set[str] = set()

    def __init__(self):
        self.use_labels = True
        self.use_imgs = True

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

    def __init__(self):
        self.use_labels = False
        self.use_imgs = False

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
