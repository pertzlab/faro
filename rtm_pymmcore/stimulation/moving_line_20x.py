from .base_stimulation import Stim
import numpy as np
import skimage
import math


class StimLine(Stim):
    """
    Stimulate a line across the field of view.

    This class implements a stimulation that stimulates a line across the field of view.
    The line can be parametrized.
    """

    def __init__(
        self,
        first_stim_frame=0,
        frames_for_1_loop=120,
        stripe_width=278,
        n_frames_total=360,
        mask_height=1024,
        mask_width=1024,
        use_labels=False,
        use_imgs=False,
    ):
        super().__init__()
        self.first_stim_frame = first_stim_frame
        self.frames_for_1_loop = frames_for_1_loop
        self.stripe_width = stripe_width
        self.n_frames_total = n_frames_total
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.use_labels = use_labels

    def spot_mask_linescan(
        self,
        frame_count_1_loop,
        time_step,
        offset,
        stripe_width,
        height=1024,
        width=1024,
        direction="right",
    ):
        spot_mask = np.zeros((height, width))
        stim_pattern_frame = (time_step - offset) % frame_count_1_loop
        total_travel_distance = width + stripe_width
        relative_frame = stim_pattern_frame / frame_count_1_loop  # goes from 0 to 1

        if direction == "left":
            relative_frame = 1 - relative_frame

        # Stripe moves from fully outside right edge to halfway past the left edge
        center_pos = width + stripe_width / 2 - total_travel_distance * relative_frame
        start = center_pos - stripe_width / 2
        end = center_pos + stripe_width / 2

        # Clip stripe to within image bounds
        clipped_start = max(int(start), 0)
        clipped_end = min(int(end), width)

        if clipped_end > clipped_start:
            spot_mask[:, clipped_start:clipped_end] = 1

        return spot_mask.astype(bool)

    def get_stim_mask(
        self, label_images: dict, metadata: dict, img: np.array = None
    ) -> np.ndarray:

        time_step = metadata.get("timestep", 0)
        is_a_stim_timestep = metadata.get("stim", False)
        # guard against empty dict: next(iter(...)) would raise StopIteration
        if label_images and len(label_images) > 0:
            label_image = label_images[next(iter(label_images))]
            height, width = label_image.shape
        else:
            height = self.mask_height
            width = self.mask_width

        spot_mask = np.zeros((height, width))

        if is_a_stim_timestep:
            if (time_step) < (self.frames_for_1_loop / 2 + self.first_stim_frame):
                offset = self.first_stim_frame
                spot_mask = self.spot_mask_linescan(
                    self.frames_for_1_loop,
                    time_step,
                    offset,
                    self.stripe_width,
                    height=height,
                    width=width,
                    direction="left",
                )
            else:
                offset = self.first_stim_frame
                spot_mask = self.spot_mask_linescan(
                    self.frames_for_1_loop,
                    time_step,
                    offset,
                    self.stripe_width,
                    height=height,
                    width=width,
                    direction="left",
                )
                offset = self.first_stim_frame + (self.frames_for_1_loop / 2)
                spot_mask2 = self.spot_mask_linescan(
                    self.frames_for_1_loop,
                    time_step,
                    offset,
                    self.stripe_width,
                    height=height,
                    width=width,
                    direction="left",
                )
                spot_mask = np.max([spot_mask, spot_mask2], axis=0)
        return spot_mask.astype("uint8"), None
