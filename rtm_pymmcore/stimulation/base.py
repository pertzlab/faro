from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.measure import regionprops


class Stim(ABC):
    """Stimulator that needs no pipeline data (only metadata).

    Subclasses receive ``metadata`` which includes ``"img_shape"``
    (height, width) so they can create masks of the correct size.
    """

    required_metadata: set[str] = set()

    @abstractmethod
    def get_stim_mask(
        self,
        metadata: dict,
    ) -> tuple[npt.NDArray[np.uint8], object]: ...


class StimWithImage(Stim):
    """Stimulator that needs the raw image (but not segmentation labels)."""

    @abstractmethod
    def get_stim_mask(
        self,
        metadata: dict,
        img: np.ndarray,
    ) -> tuple[npt.NDArray[np.uint8], object]: ...


class StimWithPipeline(Stim):
    """Stimulator that needs segmentation labels (and optionally image/tracks)."""

    @abstractmethod
    def get_stim_mask(
        self,
        label_images: dict,
        metadata: dict,
        img: np.ndarray = None,
        tracks: "pd.DataFrame | None" = None,
    ) -> tuple[npt.NDArray[np.uint8], object]: ...


class StimWholeFOV(Stim):
    """Stimulate the whole FOV."""

    def get_stim_mask(self, metadata: dict) -> tuple[npt.NDArray[np.uint8], object]:
        return True, None


class StimTopEdgeMeta(StimWithPipeline):
    """Illuminate the top *fraction* of each cell's y-extent.

    Unlike ``StimTopEdge`` (in the notebook), the fraction is read from
    ``metadata["stim_fraction"]`` at runtime, so it can vary per-event.

    Declares ``required_metadata = {"stim_fraction"}`` so that
    ``pipeline.validate_pipeline()`` flags events that forget to set it.
    """

    required_metadata: set[str] = {"stim_fraction"}

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        from skimage.morphology import disk as _disk, dilation

        labels = label_images["labels"]
        stim_mask = np.zeros(labels.shape, dtype=np.uint8)
        fraction = metadata["stim_fraction"]
        selem = _disk(3)

        for prop in regionprops(labels):
            minr, minc, maxr, maxc = prop.bbox
            y_cutoff = minr + fraction * (maxr - minr)

            cell_mask = labels == prop.label
            rows, cols = np.where(cell_mask)
            top_pixels = rows < y_cutoff
            if not top_pixels.any():
                continue

            local = np.zeros_like(labels, dtype=np.uint8)
            local[rows[top_pixels], cols[top_pixels]] = 1
            local = dilation(local, footprint=selem)
            stim_mask = np.maximum(stim_mask, local)

        return stim_mask, None


class StimNothing(Stim):
    """Use when you don't want to stimulate. Returns empty stimulation mask."""

    def get_stim_mask(self, metadata: dict) -> tuple[npt.NDArray[np.uint8], object]:
        h, w = metadata["img_shape"]
        return np.zeros((h, w), dtype=np.uint8), [1, 2, 3, 4]  # some dummy values
