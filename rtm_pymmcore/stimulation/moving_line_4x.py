from .base import Stim
import numpy as np


class StimLine4x(Stim):
    """
    Stimulate a line across the field of view.

    This class implements a stimulation that stimulates a line across the field of view.
    The line can be parametrized.
    """

    def __init__(
        self,
        first_stim_frame=0,
        frames_for_1_loop=120,
        stripe_width=55,  # approx. 180um
        n_frames_total=360,
        mask_height=1024,
        mask_width=1024,
        stim_height=205,
        stim_width=205,
    ):
        self.first_stim_frame = first_stim_frame
        self.frames_for_1_loop = frames_for_1_loop
        self.stripe_width = stripe_width
        self.n_frames_total = n_frames_total
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.stim_height = stim_height
        self.stim_width = stim_width

    def spot_mask_linescan(
        self,
        frame_count_1_loop,
        time_step,
        offset,
        stripe_width=55,
        height=1024,
        width=1024,
        stim_h=205,
        stim_w=205,
        direction="right",
    ):
        # full output (this is what you will imshow / use as mask)
        spot_mask = np.zeros((height, width), dtype=np.uint8)

        # ---- build the moving stripe in the small stimulus area ----
        stim = np.zeros((stim_h, stim_w), dtype=np.uint8)

        stim_pattern_frame = (time_step - offset) % frame_count_1_loop
        relative_frame = stim_pattern_frame / frame_count_1_loop  # 0..1

        if direction == "left":
            relative_frame = 1 - relative_frame

        total_travel_distance = stim_w + stripe_width
        center_pos = stim_w + stripe_width / 2 - total_travel_distance * relative_frame
        start = center_pos - stripe_width / 2
        end = center_pos + stripe_width / 2

        # Clip stripe to within stimulus bounds
        clipped_start = max(int(start), 0)
        clipped_end = min(int(end), stim_w)

        if clipped_end > clipped_start:
            stim[:, clipped_start:clipped_end] = 1

        # ---- paste stimulus into the canvas ----
        y0 = 400
        x0 = 400
        spot_mask[y0 : y0 + stim_h, x0 : x0 + stim_w] = stim

        return spot_mask.astype(bool)

    def get_stim_mask(self, metadata: dict) -> np.ndarray:

        time_step = metadata.get("timestep", 0)
        is_a_stim_timestep = metadata.get("stim", False)
        height = metadata.get("img_shape", (self.mask_height, self.mask_width))[0]
        width = metadata.get("img_shape", (self.mask_height, self.mask_width))[1]

        spot_mask = np.zeros((height, width))

        if is_a_stim_timestep:
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
            if time_step >= (self.frames_for_1_loop / 2 + self.first_stim_frame):
                offset2 = self.first_stim_frame + (self.frames_for_1_loop / 2)
                spot_mask2 = self.spot_mask_linescan(
                    self.frames_for_1_loop,
                    time_step,
                    offset2,
                    self.stripe_width,
                    height=height,
                    width=width,
                    direction="left",
                )
                spot_mask = np.max([spot_mask, spot_mask2], axis=0)
        return spot_mask.astype("uint8"), None
