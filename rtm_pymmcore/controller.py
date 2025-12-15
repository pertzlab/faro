from pymmcore_plus import CMMCorePlus
from rtm_pymmcore.img_processing_pip import store_img, ImageProcessingPipeline
from rtm_pymmcore.data_structures import Fov, ImgType
from rtm_pymmcore.dmd import DMD

import threading
from useq._mda_event import SLMImage
from useq import HardwareAutofocus
import useq
from useq import MDAEvent
from queue import Queue, Empty as QueueEmpty
import numpy as np
import pandas as pd
import time
import tifffile
import os
from concurrent.futures import ThreadPoolExecutor


class Analyzer:
    """Image analyzer with priority: Get → Store >> Pipeline.

    Priority order:
    1. get(img) - immediate return to MDA (< 1ms)
    2. store_img() - disk save (guaranteed, no skip)
    3. pipeline.run() - only if resources available (can skip if overloaded)

    This ensures:
    - Real-time MDA unaffected
    - Data always saved
    - Pipeline runs when possible without blocking anything
    """

    def __init__(
        self,
        pipeline: ImageProcessingPipeline = None,
        max_workers: int = 4,  # Number of parallel processing threads
        max_queue_size: int = 60,
        *,
        debug: bool = False,
        debug_every: int = 10,
    ):
        """
        Args:
            pipeline: ImageProcessingPipeline instance (optional for analysis)
            max_workers: Number of worker threads for pipeline (default: 4)
            max_queue_size: Maximum images in executor queue before deferring (default: 60)
        """
        self.pipeline = pipeline
        # Pipeline executor with fewer workers - low priority
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_queue_size = max_queue_size
        self.active_pipeline_tasks = 0
        self.task_lock = threading.Lock()
        # Debug settings
        self.debug = debug
        self.debug_every = max(1, int(debug_every))
        self._debug_counter = 0

        # High-priority: storage queue (separate from pipeline)
        self._storage_queue: Queue = Queue(maxsize=max_queue_size)
        # Deferred / control flags must be initialized BEFORE starting threads
        self._is_running = True
        self._stop_event = threading.Event()

        self._storage_thread = threading.Thread(
            target=self._storage_worker, daemon=True, name="StorageWorker"
        )
        self._storage_thread.start()

        # Deferred pipeline queue: metadata-only (images loaded from disk when processing)
        # Stores (event, metadata, folder) tuples instead of full images to save RAM
        self._deferred_queue: Queue = Queue()
        self._deferred_thread = threading.Thread(
            target=self._deferred_worker, daemon=True, name="DeferredWorker"
        )
        self._deferred_thread.start()

        # Statistics for monitoring
        self.stored_images = 0
        self.skipped_pipeline = 0
        self.deferred_processed = 0

    def _storage_worker(self):
        """Worker thread for storage - high priority, never skipped."""
        while self._is_running and not self._stop_event.is_set():
            try:
                img, event, metadata, folder = self._storage_queue.get(timeout=0.5)
            except QueueEmpty:
                continue

            try:
                # PRIORITY 1: Always store the image
                self._do_store(img, metadata, folder)
                self.stored_images += 1

                if self.debug:
                    pass
                    print(
                        f"[Analyzer] Stored image type={metadata.get('img_type')} t={metadata.get('timestep')} fov={metadata.get('fov')} pending_storage={self._storage_queue.qsize()}"
                    )

                # PRIORITY 2: Pipeline only if resources available
                self._try_submit_pipeline(img, event, metadata, folder)

            except OSError as e:
                print(f"Error storing image: {type(e).__name__}: {str(e)}")
            except Exception as e:
                print(
                    f"Unexpected error processing image: {type(e).__name__}: {str(e)}"
                )
            finally:
                self._storage_queue.task_done()

    def _do_store(self, img: np.array, metadata: dict, folder: str):
        """Actually store image to disk (guaranteed, never skipped)."""
        img_type = metadata["img_type"]

        if img_type == ImgType.IMG_RAW:
            store_img(img, metadata, self.pipeline.storage_path, "raw")

        elif img_type == ImgType.IMG_STIM:
            store_img(img, metadata, self.pipeline.storage_path, "stim")

        elif img_type == ImgType.IMG_OPTOCHECK:
            len_raw_img = len(metadata["channels"])
            img_raw = img[0:len_raw_img]

            store_img(img_raw, metadata, self.pipeline.storage_path, "raw")
            if not os.path.exists(
                os.path.join(self.pipeline.storage_path, "optocheck")
            ):
                os.makedirs(os.path.join(self.pipeline.storage_path, "optocheck"))
            store_img(img, metadata, self.pipeline.storage_path, "optocheck")

    def _try_submit_pipeline(
        self, img: np.array, event: MDAEvent, metadata: dict, folder: str
    ):
        """Try to submit to pipeline, but defer if overloaded (non-blocking).

        Optimization: Pass image directly in memory if capacity available (faster),
        defer to later if overloaded (guaranteed processing).
        """
        if self.pipeline is None:
            return

        if metadata["img_type"] == ImgType.IMG_STIM:
            # Don't pipeline stim images
            return

        with self.task_lock:
            # Check if we have capacity for pipeline
            if self.active_pipeline_tasks >= self.max_queue_size:
                # Pipeline is overloaded - defer this image for later processing
                self.skipped_pipeline += 1
                # Queue for deferred processing (metadata only, image will be loaded from disk)
                try:
                    self._deferred_queue.put_nowait((event, metadata, folder))
                    if self.debug:
                        print(
                            f"[Analyzer] Pipeline overloaded → defer (active={self.active_pipeline_tasks}, max={self.max_queue_size}, pending_deferred={self._deferred_queue.qsize()})"
                        )
                except Exception:
                    pass  # Deferred queue also full - metadata lost (acceptable, storage is priority)
                return

            # We have capacity, increment counter
            self.active_pipeline_tasks += 1

        # Submit to pipeline with low priority
        try:
            # Optimization: Use memory if capacity available (faster than disk read)
            future = self.executor.submit(
                self.pipeline.run, img=img, event=event, file_path=None
            )
            future.add_done_callback(lambda f: self._pipeline_task_done(future=f))
            if self.debug:
                print(
                    f"[Analyzer] Pipeline submitted (active={self.active_pipeline_tasks}, pending_deferred={self._deferred_queue.qsize()})"
                )
        except (RuntimeError, OSError) as e:
            print(f"Could not submit pipeline task: {str(e)}")
            with self.task_lock:
                self.active_pipeline_tasks -= 1

    def _deferred_worker(self):
        """Worker thread that processes deferred images when capacity becomes available.

        Loads images from disk instead of keeping them in RAM.
        """
        while self._is_running and not self._stop_event.is_set():
            try:
                # Try to get deferred metadata (non-blocking check)
                event, metadata, folder = self._deferred_queue.get(timeout=1.0)
            except QueueEmpty:
                continue

            try:
                # Check if we have capacity now
                with self.task_lock:
                    if self.active_pipeline_tasks >= self.max_queue_size:
                        # Still overloaded - put back in queue and wait
                        self._deferred_queue.put_nowait((event, metadata, folder))
                        if self.debug:
                            pass
                            print(
                                f"[Analyzer] Still overloaded → requeue deferred (active={self.active_pipeline_tasks}, max={self.max_queue_size})"
                            )
                        time.sleep(0.5)
                        continue

                    # Capacity available - increment counter
                    self.active_pipeline_tasks += 1

                # Construct file path to load image from disk
                fname = metadata["fname"]
                file_path = os.path.join(
                    self.pipeline.storage_path, "raw", fname + ".tiff"
                )

                # Submit deferred image to pipeline (will load from disk)
                try:
                    future = self.executor.submit(
                        self.pipeline.run, img=None, event=event, file_path=file_path
                    )
                    future.add_done_callback(
                        lambda f: self._pipeline_task_done(future=f)
                    )
                    self.deferred_processed += 1
                    if self.debug:
                        print(
                            f"[Analyzer] Deferred submitted (loading from {file_path}, deferred_processed={self.deferred_processed})"
                        )
                except (RuntimeError, OSError):
                    with self.task_lock:
                        self.active_pipeline_tasks -= 1
                    # Put back in queue for retry
                    self._deferred_queue.put_nowait((event, metadata, folder))

            except Exception as e:
                print(f"Error processing deferred image: {type(e).__name__}: {str(e)}")

    def run(self, img: np.array, event: MDAEvent) -> dict:
        """Called from MDA callback - must return INSTANTLY.

        Just queues the image for storage, actual work happens in storage thread.
        """
        metadata = event.metadata
        # Optionally print stats periodically for live debugging
        try:
            # Put in storage queue (high priority)
            # Non-blocking: if queue full, just skip (images before it will be stored)
            self._storage_queue.put_nowait((img, event, metadata, "raw"))
            if self.debug:
                self._debug_counter += 1
                if (self._debug_counter % self.debug_every) == 0:
                    stats = self.get_stats()
                    print(
                        f"[Analyzer] Stats {stats} (storage_q={self._storage_queue.qsize()}, deferred_q={self._deferred_queue.qsize()})"
                    )
        except RuntimeError:
            # Queue full - image skipped (but previous images are being stored)
            # This is acceptable as storage is non-blocking
            pass

        return {"result": "STOP"}

    def _pipeline_task_done(self, future=None):
        """Called when pipeline task completes.

        Args:
            future: The Future object from the executor (if provided as callback arg)
        """
        with self.task_lock:
            self.active_pipeline_tasks -= 1

        # Check if the task raised an exception
        if future is not None:
            try:
                future.result()  # This will re-raise any exception that occurred
            except Exception as e:
                import traceback

                print(f"[Analyzer] Pipeline task FAILED with exception:")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                print("Full traceback:")
                traceback.print_exc()

        if self.debug:
            print(
                f"[Analyzer] Pipeline task done (active={self.active_pipeline_tasks})"
            )

    def shutdown(self, wait: bool = True):
        """Shutdown storage thread, deferred thread, and pipeline executor."""
        self._stop_event.set()

        if wait:
            # Wait for storage queue to empty
            try:
                self._storage_queue.join()
            except Exception:
                pass

            # Wait for deferred queue to empty
            try:
                self._deferred_queue.join()
            except Exception:
                pass

            # Wait for storage thread to finish
            self._storage_thread.join(timeout=10)

            # Wait for deferred thread to finish
            self._deferred_thread.join(timeout=10)

        self._is_running = False
        self.executor.shutdown(wait=wait)

    def get_stats(self) -> dict:
        """Get analyzer statistics."""
        with self.task_lock:
            return {
                "stored_images": self.stored_images,
                "skipped_pipeline": self.skipped_pipeline,
                "deferred_processed": self.deferred_processed,
                "pending_storage": self._storage_queue.qsize(),
                "pending_deferred": self._deferred_queue.qsize(),
                "active_pipeline_tasks": self.active_pipeline_tasks,
            }


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
        # Analyze the image
        self._frame_buffer.append(img)
        try:
            md = event.metadata or {}
            print(
                f"[Controller] frameReady: last_channel={md.get('last_channel')} img_type={md.get('img_type')} stack_size={len(self._frame_buffer)} fname={md.get('fname')}"
            )
        except Exception:
            pass
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

                        # Use a per-event copy of metadata to avoid cross-event mutation/race
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
                            metadata=dict(metadata_dict),
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

                            # Use a per-event copy of metadata to avoid cross-event mutation/race
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
                                metadata=dict(metadata_dict),
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
                                except (TimeoutError, QueueEmpty) as e:
                                    print(
                                        f"Warning: Stimulation mask not ready (timeout): {str(e)}"
                                    )
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

                        # Use a per-event copy of metadata to avoid cross-event mutation/race
                        stimulation_event = useq.MDAEvent(
                            index={
                                "t": timestep,
                                "p": row["fov"],
                            },
                            channel={
                                "config": stim_channel_name,
                                "group": stim_channel_group,
                            },
                            metadata=dict(metadata_dict),
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
        dmd_needs_to_be_waken=False,
        project_path=None,
    ):
        super().__init__(
            analyzer, mmc, queue, use_autofocus_event, dmd, dmd_needs_to_be_waken
        )
        self._project_path = project_path

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        """Override to load images from disk for simulated controller.

        Maintains the same frame aggregation logic as the real Controller but loads
        images from the project path instead of from the microscope.
        """
        if event.metadata["last_channel"]:
            fname = event.metadata["fname"]
            img_type = event.metadata["img_type"]

            # Load image from disk based on type
            if img_type == ImgType.IMG_RAW:
                img_loaded = tifffile.imread(
                    os.path.join(self._project_path, "raw", fname + ".tiff")
                )
                self._results = self._analyzer.run(img_loaded, event)

            elif img_type == ImgType.IMG_OPTOCHECK:
                img_loaded = tifffile.imread(
                    os.path.join(self._project_path, "optocheck", fname + ".tiff")
                )
                self._results = self._analyzer.run(img_loaded, event)

            elif img_type == ImgType.IMG_STIM:
                pass  # Stim images are not processed in this simulation
            else:
                raise ValueError(f"Unknown image type: {img_type}")

            try:
                md = event.metadata or {}
                print(
                    f"[ControllerSimulated] frameReady: last_channel={md.get('last_channel')} img_type={md.get('img_type')} stack_size={len(self._frame_buffer)} fname={md.get('fname')}"
                )
            except Exception:
                pass
