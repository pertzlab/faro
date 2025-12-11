from pymmcore_plus import CMMCorePlus
from rtm_pymmcore.img_processing_pip import store_img, ImageProcessingPipeline
from rtm_pymmcore.data_structures import Fov, ImgType
from rtm_pymmcore.dmd import DMD

import threading
from useq._mda_event import SLMImage
from useq import HardwareAutofocus
import useq
from useq import MDAEvent
from queue import Queue
import numpy as np
import threading
import pandas as pd
import time
import tifffile
import os
from concurrent.futures import ThreadPoolExecutor


class Analyzer:
    """When a new image is acquired, decide what to do here. Segment, get stim mask, just store.

    When the processing queue is full, saves images to disk and processes from disk to avoid
    memory overflow. Direct in-memory processing is used when queue has capacity.
    """

    def __init__(
        self,
        pipeline: ImageProcessingPipeline = None,
        max_workers: int = 4,
        max_queue_size: int = 10,
    ):
        """
        Args:
            pipeline: ImageProcessingPipeline instance
            max_workers: Number of worker threads (default: 4)
            max_queue_size: Maximum number of tasks in queue before switching to disk-based processing (default: 10)
        """
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_queue_size = max_queue_size
        self.active_tasks = 0
        self.task_lock = threading.Lock()

    def run(self, img: np.array, event: MDAEvent) -> dict:
        metadata = event.metadata
        img_type = metadata["img_type"]
        if img_type == ImgType.IMG_RAW:
            # raw image, send to pipeline and store
            store_img(img, metadata, self.pipeline.storage_path, "raw")
            # Submit pipeline processing to thread pool with smart memory management
            if self.pipeline is not None:
                self._submit_analysis(img, event, metadata, folder="raw")

        elif img_type == ImgType.IMG_STIM:
            # stim image, store
            store_img(img, metadata, self.pipeline.storage_path, "stim")

        elif img_type == ImgType.IMG_OPTOCHECK:
            # on one side store image as normal raw, but also send a copy to optocheck pipeline
            # raw image, send to pipeline and store
            len_raw_img = len(metadata["channels"])
            img_raw = img[0:len_raw_img]
            # Submit pipeline processing to thread pool with smart memory management
            if self.pipeline is not None:
                self._submit_analysis(img, event, metadata, folder="optocheck")
            store_img(img_raw, metadata, self.pipeline.storage_path, "raw")
            if not os.path.exists(
                os.path.join(self.pipeline.storage_path, "optocheck")
            ):
                os.makedirs(os.path.join(self.pipeline.storage_path, "optocheck"))
            store_img(img, metadata, self.pipeline.storage_path, "optocheck")

        return {"result": "STOP"}

    def _submit_analysis(
        self, img: np.array, event: MDAEvent, metadata: dict, folder: str = "raw"
    ):
        """Submit image analysis task, using file_path when queue is full to avoid memory overflow.

        Args:
            img: Image array (used if queue has capacity)
            event: MDAEvent with metadata
            metadata: Image metadata dict
            folder: Folder where image is stored ("raw" or "optocheck")
        """

        with self.task_lock:
            queue_is_full = self.active_tasks >= self.max_queue_size
            if not queue_is_full:
                self.active_tasks += 1

        if queue_is_full:
            # Queue is full: process from disk instead of keeping image in memory
            file_path = os.path.join(
                self.pipeline.storage_path, folder, metadata["fname"] + ".tiff"
            )
            future = self.executor.submit(
                self.pipeline.run, img=None, event=event, file_path=file_path
            )
        else:
            # Queue has capacity: process directly from memory
            future = self.executor.submit(
                self.pipeline.run, img=img, event=event, file_path=None
            )

        # Decrement counter when task completes
        future.add_done_callback(lambda f: self._task_done())

    def _task_done(self):
        """Called when a task completes to decrement the active task counter."""
        with self.task_lock:
            self.active_tasks -= 1

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool executor. Call this when done with acquisition."""
        self.executor.shutdown(wait=wait)


class Controller:
    STOP_EVENT = object()

    def __init__(
        self,
        analyzer: Analyzer,
        mmc,
        queue,
        use_autofocus_event=False,
        dmd=None,
        dmd_needs_to_be_waken=False,
    ):
        self._queue = queue  # queue of MDAEvents
        self._analyzer = analyzer  # analyzer object
        self._results: dict = {}  # results of analysis
        self._current_group = mmc.getChannelGroup()
        self._frame_buffer = (
            []
        )  # buffer to hold the frames until one sequence is complete
        self._dmd = dmd
        self._mmc = mmc
        self.use_autofocus_event = use_autofocus_event
        self.dmd_needs_to_be_waken = dmd_needs_to_be_waken
        self._mmc.mda.events.frameReady.disconnect()
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        # Analyze the image+
        self._frame_buffer.append(img)
        # check if it's the last acquisition for this MDAsequence
        if event.metadata["last_channel"]:
            frame_complete = np.stack(self._frame_buffer, axis=-1)
            # move new axis to the first position
            frame_complete = np.moveaxis(frame_complete, -1, 0)

            self._frame_buffer = []
            self._results = self._analyzer.run(frame_complete, event)

    def stop_run(self):
        self._queue.put(self.STOP_EVENT)
        self._mmc.mda.cancel()
        # Shutdown the thread pool executor to allow pending tasks to complete
        self._analyzer.shutdown(wait=True)

    def is_running(self):
        return self._queue.qsize() > 0

    def run(self, df_acquire: pd.DataFrame):
        queue_sequence = iter(self._queue.get, self.STOP_EVENT)
        self._mmc.run_mda(queue_sequence)
        try:
            for exp_time in df_acquire["time"].unique():
                # extract the lines with the current timestep from the DF
                current_time_df = df_acquire[df_acquire["time"] == exp_time]
                for index, row in current_time_df.iterrows():
                    # Pause if queue is getting too full to allow analyzer to catch up
                    while self._queue.qsize() >= 3:
                        time.sleep(0.1)  # Wait 1s before checking again
                    # Get FOV data directly from the DataFrame
                    timestep = row["timestep"]
                    fov_obj = row["fov_object"]
                    fov_index = row["fov"]
                    fov_x = row["fov_x"]
                    fov_y = row["fov_y"]
                    fov_z = row.get("fov_z", None)
                    fov_af_offset = row.get("fov_af_offset", None)

                    event_start_time = float(row["time"])

                    channels = row["channels"]

                    metadata_dict = dict(row)
                    metadata_dict["img_type"] = ImgType.IMG_RAW
                    metadata_dict["last_channel"] = channels[-1]

                    if "stim" not in df_acquire.columns:
                        stim = False
                        metadata_dict["stim"] = False
                    else:
                        stim = row["stim"]

                    if "optocheck" not in df_acquire.columns:
                        optocheck = False
                        metadata_dict["optocheck"] = False
                    else:
                        optocheck = row["optocheck"]

                    if self.use_autofocus_event:
                        acquisition_event = useq.MDAEvent(
                            index={"t": timestep, "c": 0, "p": fov_index},
                            x_pos=fov_x,
                            y_pos=fov_y,
                            z_pos=fov_z,
                            min_start_time=event_start_time,
                            action=HardwareAutofocus(
                                autofocus_motor_offset=fov_af_offset
                            ),
                        )
                        self._queue.put(acquisition_event)

                    slm_image = None
                    if self._dmd is not None and self.dmd_needs_to_be_waken:
                        slm_image = SLMImage(data=True, device=self._dmd.name)

                    for i, channel_i in enumerate(channels):
                        metadata_dict["last_channel"] = False
                        if i == 0:
                            x_pos = fov_x
                            y_pos = fov_y
                        else:
                            x_pos = None
                            y_pos = None
                        if not optocheck:
                            last_channel: bool = i == len(channels) - 1
                            metadata_dict["last_channel"] = last_channel
                        power_prop = (
                            channel_i.get("device_name", None),
                            channel_i.get("property_name", None),
                            channel_i.get("power", None),
                        )
                        if any(el is None for el in power_prop):
                            power_prop = None

                        acquisition_event = useq.MDAEvent(
                            index={
                                "t": timestep,
                                "c": i,
                                "p": fov_index,
                            },  # the index of the event in the sequence
                            channel={
                                "config": channel_i["name"],
                                "group": (
                                    channel_i["group"]
                                    if channel_i["group"] is not None
                                    else self._current_group
                                ),
                            },
                            metadata=metadata_dict,
                            x_pos=x_pos,
                            y_pos=y_pos,
                            z_pos=fov_z,
                            min_start_time=event_start_time,
                            exposure=channel_i.get("exposure", None),
                            properties=[power_prop] if power_prop is not None else None,
                            slm_image=slm_image,
                        )

                        self._queue.put(acquisition_event)

                    if optocheck:
                        metadata_dict["img_type"] = ImgType.IMG_OPTOCHECK

                        for i, optocheck_ch in enumerate(row["optocheck_channels"]):
                            last_channel: bool = i == len(row["optocheck_channels"]) - 1
                            metadata_dict["last_channel"] = last_channel

                            power_prop = (
                                optocheck_ch.get("device_name", None),
                                optocheck_ch.get("property_name", None),
                                optocheck_ch.get("power", None),
                            )
                            if any(el is None for el in power_prop):
                                power_prop = None

                            acquisition_event = useq.MDAEvent(
                                index={
                                    "t": timestep,
                                    "c": i,
                                    "p": fov_index,
                                },  # the index of the event in the sequence
                                channel={
                                    "config": optocheck_ch["name"],
                                    "group": (
                                        optocheck_ch["group"]
                                        if optocheck_ch["group"] is not None
                                        else self._current_group
                                    ),
                                },
                                metadata=metadata_dict,
                                x_pos=None,
                                y_pos=None,
                                z_pos=fov_z,
                                min_start_time=event_start_time,
                                exposure=optocheck_ch.get("exposure", None),
                                properties=(
                                    [power_prop] if power_prop is not None else None
                                ),
                                slm_image=slm_image,
                            )
                            self._queue.put(acquisition_event)

                    if stim:
                        if row["stim_power"] == 0 or row["stim_exposure"] == 0:
                            continue
                        metadata_dict["img_type"] = ImgType.IMG_STIM
                        metadata_dict["last_channel"] = True

                        power_prop = (
                            row["stim_channel_device_name"],
                            row["stim_channel_power_property_name"],
                            row["stim_power"],
                        )
                        if any(el is None for el in power_prop):
                            power_prop = None
                        stim_channel_name = row["stim_channel_name"]
                        stim_channel_group = row.get(
                            "stim_channel_group", self._current_group
                        )
                        stim_exposure = row.get("stim_exposure", None)
                        slm_image = None

                        if self._dmd is not None:
                            if self._analyzer.pipeline.stimulator.use_labels:
                                try:
                                    stim_mask = fov_obj.stim_mask_queue.get(
                                        block=True, timeout=35
                                    )
                                    stim_mask = self._dmd.affine_transform(stim_mask)
                                except Exception as e:
                                    print(f"Exception: {str(e)}")
                                    stim_mask = False
                            else:
                                stim_mask, _ = (
                                    self._analyzer.pipeline.stimulator.get_stim_mask(
                                        {}, metadata=metadata_dict
                                    )
                                )
                                stim_mask = self._dmd.affine_transform(stim_mask)
                            slm_image = SLMImage(
                                data=stim_mask,
                                device=self._dmd.name,
                            )

                        stimulation_event = useq.MDAEvent(
                            index={
                                "t": timestep,
                                "p": row["fov"],
                            },
                            channel={
                                "config": stim_channel_name,
                                "group": stim_channel_group,
                            },
                            metadata=metadata_dict,
                            x_pos=None,
                            y_pos=None,
                            z_pos=fov_z,
                            exposure=stim_exposure,
                            min_start_time=event_start_time,
                            properties=[power_prop] if power_prop is not None else None,
                            slm_image=slm_image,
                        )

                        self._queue.put(stimulation_event)

        finally:
            # Put the stop event in the queue
            self._queue.put(self.STOP_EVENT)
            while self._queue.qsize() > 0:
                time.sleep(1)


class ControllerSimulated(Controller):
    def __init__(
        self,
        analyzer,
        mmc,
        queue,
        use_autofocus_event=False,
        dmd=None,
        project_path=None,
    ):
        super().__init__(analyzer, mmc, queue, dmd)
        self._project_path = project_path

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        if event.metadata["last_channel"]:
            fname = event.metadata["fname"]
            if event.metadata["img_type"] == ImgType.IMG_RAW:
                frame_complete = tifffile.imread(
                    os.path.join(self._project_path, "raw", fname + ".tiff")
                )
                self._results = self._analyzer.run(frame_complete, event)
            elif event.metadata["img_type"] == ImgType.IMG_OPTOCHECK:
                frame_complete = tifffile.imread(
                    os.path.join(self._project_path, "optocheck", fname + ".tiff")
                )
                self._results = self._analyzer.run(frame_complete, event)
