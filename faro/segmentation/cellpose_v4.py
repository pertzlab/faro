import threading
import warnings

warnings.filterwarnings("ignore", message="Sparse invariant checks")

import numpy as np
from faro.segmentation.base import Segmentator, remove_small_objects
from cellpose import models


class CellposeV4(Segmentator):

    def __init__(
        self,
        custom_model_path=None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 50,
        gpu: bool = True,
        gamma: float = 1.0,
    ):

        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.gamma = gamma

        if custom_model_path is None:
            self.model = models.CellposeModel(gpu=gpu)
        else:
            self.model = models.CellposeModel(
                pretrained_model=custom_model_path, gpu=gpu
            )
        # Analyzer.executor runs up to ``max_workers`` pipeline tasks at
        # once — under FOV batching that is one task per FOV in a batch
        # firing nearly simultaneously. ``CellposeModel.eval`` is not
        # thread-safe (shared internal buffers + a single CUDA context
        # under PyTorch), and overlapping calls have been observed to
        # silently return empty/corrupt masks for an entire batch of
        # FOVs while the next batch works. Serialize ``.eval`` here so
        # at most one segmentation runs at a time. Tracking, feature
        # extraction, and storage stay parallel across FOVs.
        self._eval_lock = threading.Lock()

    def segment(self, image: np.ndarray) -> np.ndarray:
        # ``image`` is either a single 2D channel (Y, X) or a multi-channel
        # stack (C, Y, X) when SegmentationMethod.use_channel is a list.
        if self.gamma != 1.0:
            # Gamma is applied to the raw counts of every channel before
            # Cellpose normalizes. Cellpose's default normalize=True rescales
            # the 1st-99th percentile to [0, 1] *per channel* (models.py
            # docstring), and that rescale is linear, so x**gamma followed by
            # per-channel percentile normalization is scale-invariant per
            # channel: a brighter channel and a dimmer channel get the same
            # contrast curve. So a single gamma is correct across channels.
            image = image**self.gamma

        # For a (C, Y, X) stack tell Cellpose the channel axis so it
        # normalizes each channel independently and packs them as the model's
        # input channels (Cellpose-SAM is channel-order invariant and uses the
        # first 3, zero-filling the rest). A 2D image needs no channel_axis.
        eval_kwargs = dict(
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )
        if image.ndim == 3:
            eval_kwargs["channel_axis"] = 0

        with self._eval_lock:
            masks, flows, styles = self.model.eval(image, **eval_kwargs)

        if self.min_size > 0:
            # remove cells below threshold
            masks = remove_small_objects(masks, min_size=self.min_size, connectivity=1)
        return masks
