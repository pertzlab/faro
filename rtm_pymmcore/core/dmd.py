import skimage
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time
import scipy

# from .acquisition import acq
from pymmcore_plus import CMMCorePlus
from useq import PropertyTuple
from useq._mda_event import SLMImage
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
        calibration_profile,
        affine_matrix=None,
        test_mode: bool = False,
    ):
        """Args:
        mmc: core object from CMMCorePlus()
        test_mode: try the function without a DMD set up in uManager. Defaults to False.
        """
        # Load all dmd properties from micro-manager
        self.mmc = mmc
        self.test_mode = test_mode
        self.affine = None
        self.calibration_profile = calibration_profile

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

    def all_on(self):
        """turn on projector all pixels for a long time"""
        self.mmc.setSLMPixelsTo(self.name, 255)
        self.mmc.displaySLMImage(self.name)

    def all_off(self):
        """turn off pixels"""
        self.mmc.setSLMPixelsTo(self.name, 0)
        self.mmc.displaySLMImage(self.name)

    def all_on_img(self):
        """generate an image with all pixels on"""
        all_on_image = np.full((self.height, self.width), 255, dtype=np.uint8)
        return all_on_image

    def checker_board(self, pixels=20):
        """display a checkerboard pattern for a long time"""
        # build checkerboard
        checker_board = (np.indices((self.height, self.width)) // pixels).sum(
            axis=0
        ) % 2
        checker_board = checker_board.astype(np.uint8) * 255
        self.mmc.setSLMImage(self.name, checker_board)
        self.mmc.displaySLMImage(self.name)

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

    def calibrate(
        self,
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
                channel={
                    "config": self.calibration_profile["channel_config"],
                    "group": self.calibration_profile["channel_group"],
                },
                properties=[
                    (
                        self.calibration_profile["device_name"],
                        self.calibration_profile["property_name"],
                        self.calibration_profile["power"],
                    )
                ],
            )
            events.append(event_p)

        self.mmc.mda.events.frameReady.disconnect()

        @self.mmc.mda.events.frameReady.connect
        def new_frame(img: np.ndarray, event: MDAEvent):
            calibration_images.append(img)
            if verbose:
                plt.imshow(img, cmap="gray")
                plt.show()

        for event in events:
            self.mmc.mda.run([event])
            time.sleep(0.1)
        calibration_images = np.array(calibration_images)

        for img in calibration_images:
            img = skimage.filters.gaussian(img, sigma=1)
            max_x = np.argmax(img.max(axis=0))
            max_y = np.argmax(img.max(axis=1))
            dst.append((max_x, max_y))

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
            self.mmc.mda.events.frameReady.disconnect()
            self.mmc.mda.run(
                [
                    MDAEvent(
                        slm_image=SLMImage(data=self.sample_mask_off, device=self.name),
                        exposure=1,
                        properties=[
                            (
                                self.calibration_profile["device_name"],
                                self.calibration_profile["property_name"],
                                0,
                            )
                        ],
                    )
                ]
            )

            print(
                f"Not enough inliers found for calibration. Total inliers: {np.sum(inliers)}, required: 5. Try again. "
            )
            self.all_on()
            return
        self.affine = affine_model.params

        if True:
            # test the calibration on three new points
            event_p = []
            events = []
            test_image = []
            test_src = []
            test_dst = []
            p0, p1, p2 = ([300, 500], [650, 200], [500, 800])
            camera_height = self.mmc.getImageHeight()
            camera_width = self.mmc.getImageWidth()

            for p in [p0, p1, p2]:
                img_p = np.zeros((camera_height, camera_width)).astype(np.uint8)
                rr, cc = skimage.draw.disk((p[0], p[1]), 20)
                test_src.append((p[1], p[0]))
                img_p[rr, cc] = 255
                img_warp = self.affine_transform(img_p)

                event_p = MDAEvent(
                    slm_image=SLMImage(
                        data=img_warp, device=self.name, exposure=exposure
                    ),
                    exposure=exposure,
                    channel={
                        "config": self.calibration_profile["channel_config"],
                        "group": self.calibration_profile["channel_group"],
                    },
                    properties=[
                        (
                            self.calibration_profile["device_name"],
                            self.calibration_profile["property_name"],
                            self.calibration_profile["power"],
                        )
                    ],
                )
                events.append(event_p)

            self.mmc.mda.events.frameReady.disconnect()

            @self.mmc.mda.events.frameReady.connect
            def new_frame(img: np.ndarray, event: MDAEvent):
                test_image.append(img)

            for event in events:
                self.mmc.mda.run([event])
                time.sleep(0.5)
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
                    test_dst[i][0], test_dst[i][1], marker="x", facecolors="red"
                )
                axs[i].scatter(
                    test_src[i][0], test_src[i][1], marker="x", facecolors="green"
                )

            for i in range(3):
                axs[3].scatter(
                    test_src[i][0], test_src[i][1], marker="x", facecolor="red"
                )
                axs[3].scatter(
                    test_dst[i][0], test_dst[i][1], marker="x", facecolor="green"
                )
            axs[3].set_xlim(0, camera_width)
            axs[3].set_ylim(camera_height, 0)

            plt.show()
        self.mmc.mda.events.frameReady.disconnect()
        self.mmc.mda.run(
            [
                MDAEvent(
                    slm_image=SLMImage(data=self.sample_mask_off, device=self.name),
                    exposure=1,
                    properties=[
                        (
                            self.calibration_profile["device_name"],
                            self.calibration_profile["property_name"],
                            0,
                        )
                    ],
                )
            ]
        )
