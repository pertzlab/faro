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
        self,
        label_images: dict,
        metadata: dict,
        img: np.ndarray,
        df_tracked: np.ndarray = None,
    ) -> npt.NDArray[np.uint8]:
        """
        Parameters:
        label_images (dict): Dictionary of label images.
        metadata (dict): Metadata dictionary.
        img (np.ndarray): The raw image.
        df_tracked (np.ndarray): DataFrame with tracked cells (optional).

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
        self,
        label_images: dict,
        metadata: dict = None,
        img: np.array = None,
        df_tracked: np.ndarray = None,
    ) -> npt.NDArray[np.uint8]:
        return np.ones((img.shape[-2], img.shape[-1]), dtype=np.uint8), None


class StimNothing(Stim):
    """Use when you don't want to stimulate. Returns empty stimulation mask."""

    def get_stim_mask(
        self,
        label_images: dict,
        metadata: dict = None,
        img: np.array = None,
        df_tracked: np.ndarray = None,
    ) -> npt.NDArray[np.uint8]:
        label_image = list(label_images.values())[0] if label_images else None
        if label_image is not None:
            return np.zeros_like(label_image), [1, 2, 3, 4]  # some dummy values
        return np.zeros((1, 1), dtype=np.uint8), []
