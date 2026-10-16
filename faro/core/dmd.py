import skimage
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy

# from .acquisition import acq
from pymmcore_plus import CMMCorePlus
from useq import PropertyTuple
from faro.core._useq_compat import SLMImage
from useq import MDAEvent
import random


class DMD:
    """all methods that relate to the control of the DMD
    img is in camera space (2048px*2048px / 1024px*1024px / ... )
    mask is in dmd space (600px*800px)

    """

    def __init__(
        self,
        mmc: CMMCorePlus,
        resolve_power=None,
        affine_matrix=None,
        test_mode: bool = False,
    ):
        """Args:
        mmc: core object from CMMCorePlus()
        resolve_power: callable(channel) -> (device, property, power) or None,
            typically ``microscope.resolve_power``. Used by :meth:`calibrate`
            to resolve the calibration channel's light-source power from the
            microscope's ``POWER_PROPERTIES`` (rather than hardcoding the
            device/property). If None (or it returns None) calibration carries
            no power override and the line stays at its current level.
        test_mode: try the function without a DMD set up in uManager. Defaults to False.
        """
        # Load all dmd properties from micro-manager
        self.mmc = mmc
        self.test_mode = test_mode
        self.affine = None
        self._resolve_power = resolve_power

        if affine_matrix is not None:
            self.affine = affine_matrix

        if test_mode == False:
            self.name = self.mmc.getSLMDevice()
            self.height = self.mmc.getSLMHeight(self.name)
            self.width = self.mmc.getSLMWidth(self.name)
            self.bppx = self.mmc.getSLMBytesPerPixel(self.name)
            self.exposure_time = self.mmc.getSLMExposure(self.name)
            self.sample_mask_on = np.full((self.height, self.width), 255).astype(
                np.uint8
            )
            self.sample_mask_off = np.zeros((self.height, self.width)).astype(np.uint8)
            # The pattern shown during live view. The microscope's keep-alive
            # loop re-displays this every refresh, so whatever all_on() /
            # checker_board() / all_off() last set persists instead of being
            # forced back to all-on.
            self.livemode_image = self.all_on_img()

    def _calibration_channel_dict(self, channel) -> dict:
        """MDAEvent ``channel`` dict for the calibration *channel*."""
        ch_dict = {"config": channel.config}
        group = getattr(channel, "group", None)
        if group:
            ch_dict["group"] = group
        return ch_dict

    def _calibration_properties(self, channel, power=None):
        """MDAEvent ``properties`` for the calibration *channel*, or None.

        Resolves (device, property, power) via the microscope's
        ``resolve_power`` so the device/property come from the single
        ``POWER_PROPERTIES`` source of truth. ``power`` overrides the channel's
        power (e.g. ``0`` to switch the line off after calibration).
        """
        if self._resolve_power is None:
            return None
        resolved = self._resolve_power(channel)
        if resolved is None:
            return None
        device, prop, default_power = resolved
        value = default_power if power is None else power
        return [(device, prop, value)]

    def _set_calibration_power(self, channel, power):
        """Set the calibration line's power directly (no camera frame).

        Used to switch the calibration light off when the routine finishes,
        without running a throwaway blank capture.
        """
        props = self._calibration_properties(channel, power)
        if props:
            ((device, prop, value),) = props  # always exactly one: the LED power
            self.mmc.setProperty(device, prop, value)

    def affine_transform(self, img):
        """Applies transformation matrix on image in camera space. Returns mask in dmd space.
        Args:
            img: image in camera space
            affine: affine transformation matrix
        """

        if self.affine is None:
            raise ValueError(
                "DMD not calibrated. affine matrix is None. Run calibrate() first."
            )

        img_transformed = skimage.transform.warp(
            img,
            self.affine,
            output_shape=(self.height, self.width),
            order=None,
            mode="constant",
            cval=0.0,
            clip=True,
            preserve_range=True,
        )
        img_transformed = img_transformed.astype(np.uint8)
        if np.max(img_transformed) == 1:
            img_transformed = img_transformed * 255
        return img_transformed

    def display_livemode(self):
        """Display the current live-view pattern with a long SLM exposure.

        Used by all_on / all_off / checker_board and by the microscope's
        keep-alive loop, which calls this on every refresh so the pattern
        chosen for live view stays put (rather than reverting to all-on).
        """
        self.mmc.setSLMExposure(self.name, 200000.0)
        self.mmc.setSLMImage(self.name, self.livemode_image)
        self.mmc.displaySLMImage(self.name)

    def all_on(self):
        """Set the live-view DMD pattern to all-pixels-on (persists).

        Use to return the DMD to full-open after a focus-check pattern such
        as checker_board().
        """
        self.livemode_image = self.all_on_img()
        self.display_livemode()

    def all_off(self):
        """Set the live-view DMD pattern to all-pixels-off (persists)."""
        self.livemode_image = np.zeros((self.height, self.width), dtype=np.uint8)
        self.display_livemode()

    def all_on_img(self):
        """generate an image with all pixels on"""
        all_on_image = np.full((self.height, self.width), 255, dtype=np.uint8)
        return all_on_image

    def checker_board(self, pixels=20):
        """Set the live-view DMD pattern to a checkerboard (persists).

        Handy for checking DMD focus during live view: the keep-alive loop
        re-displays this pattern instead of reverting to all-on after a few
        seconds.
        """
        checker_board = (np.indices((self.height, self.width)) // pixels).sum(
            axis=0
        ) % 2
        self.livemode_image = checker_board.astype(np.uint8) * 255
        self.display_livemode()

    def select_well_distributed_points(self, valid_pixels, n_points):
        """
        Select well-distributed points from valid_pixels using a grid-based approach.

        Parameters:
        - valid_pixels (np.ndarray): Array of valid pixel coordinates with shape (N, 2).
        - n_points (int): Number of points to select.
        Returns:
        - selected_points (list of tuples): List of selected (x, y) points.
        """
        selected_points = []

        # Determine grid size based on the number of points
        grid_size = int(np.sqrt(n_points))
        if grid_size**2 < n_points:
            grid_size += 1

        # Compute the size of each grid cell
        cell_height = self.height // grid_size
        cell_width = self.width // grid_size

        # Shuffle valid_pixels to ensure random selection within each cell
        shuffled_pixels = valid_pixels.copy()
        np.random.shuffle(shuffled_pixels)

        for i in range(grid_size):
            for j in range(grid_size):
                if len(selected_points) >= n_points:
                    break

                # Define the boundaries of the current cell
                row_start = i * cell_height
                row_end = (i + 1) * cell_height if i < grid_size - 1 else self.height
                col_start = j * cell_width
                col_end = (j + 1) * cell_width if j < grid_size - 1 else self.width

                # Find valid pixels within the current cell
                cell_pixels = shuffled_pixels[
                    (shuffled_pixels[:, 0] >= row_start)
                    & (shuffled_pixels[:, 0] < row_end)
                    & (shuffled_pixels[:, 1] >= col_start)
                    & (shuffled_pixels[:, 1] < col_end)
                ]

                if len(cell_pixels) > 0:
                    # Select a random pixel from the cell
                    selected_point = tuple(
                        cell_pixels[random.randint(0, len(cell_pixels) - 1)]
                    )
                    selected_points.append(selected_point)

        # If not enough points are selected, randomly select remaining points from all valid_pixels
        if len(selected_points) < n_points:
            remaining = n_points - len(selected_points)
            additional_points = random.sample(list(map(tuple, valid_pixels)), remaining)
            selected_points.extend(additional_points)

        return selected_points

    def _run_events_unsequenced(self, events):
        """Run *events* as a single MDA with hardware sequencing disabled.

        A separate ``mmc.mda.run`` per event brackets each one with
        setup/teardown_sequence, which stop and restart ``KeepDMDAlive``; each
        restart re-displays the all-on live pattern and resets the SLM
        ExposureTime, so under OverlapMode a spot can end up on a short exposure
        that blanks before the camera opens. Running every event in one MDA
        pauses ``KeepDMDAlive`` just once.

        Sequencing must be off: with it on, pymmcore-plus tries to
        hardware-combine the consecutive SLM events into an ``slm_sequence`` and
        fails validation (``SLMImage`` is not ``bytes``). With it off, the
        events still run one at a time within the single MDA.
        """
        engine = getattr(self.mmc.mda, "engine", None)
        prev = getattr(engine, "use_hardware_sequencing", None)
        if engine is not None:
            engine.use_hardware_sequencing = False
        try:
            self.mmc.mda.run(events)
        finally:
            if engine is not None and prev is not None:
                engine.use_hardware_sequencing = prev

    def calibrate(
        self,
        calibration_channel,
        verbose=False,
        n_points=9,
        radius=4,
        exposure=25,
        marker_style="x",
        calibration_points_DMD=None,
        dmd_full_border_offset_x1=50,
        dmd_full_border_offset_x2=50,
        dmd_full_border_offset_y1=50,
        dmd_full_border_offset_y2=50,
    ):
        """Calibrate the dmd and camera coordinate systems.
        Projects 3 points in DMD space and detects them in camera space,
        then finds the affine transofmation matrix.
        Args:
            calibration_channel (Channel/PowerChannel): light path used to
                image the DMD spots (config/group set the channel; for a
                PowerChannel the power is resolved via the microscope's
                resolve_power). Pass per experiment — e.g. a UV or a cyan
                channel — so it isn't fixed on the microscope.
            verbose (bool, optional): Whether to display additional images during calibration. Defaults to False.
            blur (int, optional): Blur size for captured images. Defaults to 10.
            circle_size (int, optional): Size of the calibration circle projected. Defaults to 10.
            marker_style (str, optional): Marker style for calibration points. Defaults to 'x'.
            calibration_points_DMD (list, optional): List of X/Y DMD calibration points. Defaults to [(180,180),(700,130),(180,550)], which works well on our (800x600 DMD).
        """
        # good working points:  ([250, 380], [100,800], [900, 800], [250, 800], [490,380],[500,400],[1000,340])
        src = []
        dst = []
        event_p = []
        events = []
        calibration_images = []

        img_dmd_full = (np.ones((self.height, self.width)) * 255).astype(np.uint8)
        img_dmd_full_w_borders = img_dmd_full.copy()
        img_dmd_full_w_borders[0:dmd_full_border_offset_x1] = 0
        img_dmd_full_w_borders[:, 0:dmd_full_border_offset_y1] = 0
        img_dmd_full_w_borders[-dmd_full_border_offset_x2:] = 0
        img_dmd_full_w_borders[:, -dmd_full_border_offset_y2:] = 0

        valid_pixels = np.array(np.where(img_dmd_full_w_borders > 0)).T

        if calibration_points_DMD is None:
            calibration_points_DMD = []
            calibration_points_DMD = self.select_well_distributed_points(
                valid_pixels, n_points
            )
        for p in calibration_points_DMD:
            img_p = np.zeros((self.height, self.width)).astype(np.uint8)
            src.append((p[1], p[0]))
            rr, cc = skimage.draw.disk((p[0], p[1]), radius)
            img_p[rr, cc] = 255

            event_p = MDAEvent(
                slm_image=SLMImage(data=img_p, device=self.name),
                exposure=exposure,
                channel=self._calibration_channel_dict(calibration_channel),
                properties=self._calibration_properties(calibration_channel),
            )
            events.append(event_p)

        # Connect a calibration-only frame collector. Do NOT call the
        # no-arg frameReady.disconnect() -- that removes EVERY listener,
        # including napari-micromanager's preview updater and the faro
        # controller's pipeline callback, and they never come back.
        # Connect our own handler alongside the others and disconnect
        # only it when the collection loop is done.
        @self.mmc.mda.events.frameReady.connect
        def _collect_calibration_frame(img: np.ndarray, event: MDAEvent):
            calibration_images.append(img)

        self._run_events_unsequenced(events)
        self.mmc.mda.events.frameReady.disconnect(_collect_calibration_frame)
        calibration_images = np.array(calibration_images)

        for img in calibration_images:
            img = skimage.filters.gaussian(img, sigma=1)
            max_x = np.argmax(img.max(axis=0))
            max_y = np.argmax(img.max(axis=1))
            dst.append((max_x, max_y))

        if verbose:
            n = len(calibration_images)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            fig, axs = plt.subplots(
                rows, cols, figsize=(3 * cols, 3 * rows), dpi=120, squeeze=False
            )
            for ax, img, (mx, my), (sx, sy) in zip(
                axs.flat, calibration_images, dst, src
            ):
                ax.imshow(img, cmap="gray")
                ax.scatter(
                    mx,
                    my,
                    marker="o",
                    s=120,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                )
                ax.set_title(f"DMD ({sx},{sy}) -> cam ({mx},{my})", fontsize=8)
                ax.axis("off")
            for ax in axs.flat[n:]:
                ax.axis("off")
            fig.suptitle("Calibration frames with detected spots (red x)")
            fig.tight_layout()
            plt.show()

        src = np.array(src)
        dst = np.array(dst)

        affine_model, inliers = skimage.measure.ransac(
            (src, dst),
            skimage.transform.AffineTransform,
            min_samples=3,
            residual_threshold=5,
            max_trials=5000,
        )

        if np.sum(inliers) < 4:
            self._set_calibration_power(calibration_channel, 0)
            print(
                f"Not enough inliers found for calibration. Total inliers: {np.sum(inliers)}, required: 5. Try again. "
            )
            self.all_on()
            return
        self.affine = affine_model.params

        if verbose:
            # test the calibration on three new points
            event_p = []
            events = []
            test_image = []
            test_src = []
            test_dst = []
            camera_height = self.mmc.getImageHeight()
            camera_width = self.mmc.getImageWidth()
            # Scale test points to the live camera dimensions. Hardcoded
            # values fall outside the ROI/binning combo on most setups,
            # which leaves img_p empty -> img_warp empty -> no spot fires.
            p0 = [camera_height // 4, camera_width // 4]
            p1 = [camera_height // 2, camera_width // 2]
            p2 = [3 * camera_height // 4, 3 * camera_width // 4]

            for p in [p0, p1, p2]:
                img_p = np.zeros((camera_height, camera_width)).astype(np.uint8)
                rr, cc = skimage.draw.disk((p[0], p[1]), radius)
                test_src.append((p[1], p[0]))
                img_p[rr, cc] = 255
                img_warp = self.affine_transform(img_p)

                # No exposure on SLMImage — match the calibration events
                # above. Setting SLM exposure to ``exposure`` (e.g. 25 ms)
                # blanks the DMD long before the camera opens when the
                # scope is in OverlapMode=Off (which the focus-aid cells
                # leave it in), so the test capture comes back blank.
                event_p = MDAEvent(
                    slm_image=SLMImage(data=img_warp, device=self.name),
                    exposure=exposure,
                    channel=self._calibration_channel_dict(calibration_channel),
                    properties=self._calibration_properties(calibration_channel),
                )
                events.append(event_p)

            @self.mmc.mda.events.frameReady.connect
            def _collect_test_frame(img: np.ndarray, event: MDAEvent):
                test_image.append(img)

            self._run_events_unsequenced(events)
            self.mmc.mda.events.frameReady.disconnect(_collect_test_frame)
            calibration_images = np.array(calibration_images)
            for img in test_image:
                img = skimage.filters.gaussian(img, sigma=1)
                max_x = np.argmax(img.max(axis=0))
                max_y = np.argmax(img.max(axis=1))
                test_dst.append((max_x, max_y))

            test_src = np.array(test_src)
            test_dst = np.array(test_dst)

            fig, axs = plt.subplots(figsize=(20, 4), ncols=4, dpi=250)

            for i in range(3):
                axs[i].imshow(test_image[i], cmap="gray")
                axs[i].scatter(
                    test_dst[i][0],
                    test_dst[i][1],
                    marker="o",
                    s=120,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                    label="detected" if i == 0 else None,
                )
                axs[i].scatter(
                    test_src[i][0],
                    test_src[i][1],
                    marker="o",
                    s=120,
                    facecolors="none",
                    edgecolors="lime",
                    linewidths=1.2,
                    label="requested" if i == 0 else None,
                )
                if i == 0:
                    axs[i].legend(loc="upper right", fontsize=8)

            for i in range(3):
                axs[3].scatter(
                    test_dst[i][0],
                    test_dst[i][1],
                    marker="o",
                    s=120,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                )
                axs[3].scatter(
                    test_src[i][0],
                    test_src[i][1],
                    marker="o",
                    s=120,
                    facecolors="none",
                    edgecolors="lime",
                    linewidths=1.2,
                )
            axs[3].set_xlim(0, camera_width)
            axs[3].set_ylim(camera_height, 0)

            plt.show()
        # Switch the calibration line off, then restore the live-view pattern
        # (all-on) so the DMD doesn't sit blank after a successful calibration.
        self._set_calibration_power(calibration_channel, 0)
        self.all_on()
