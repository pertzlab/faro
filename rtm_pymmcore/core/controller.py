from pymmcore_plus import CMMCorePlus
from rtm_pymmcore.core.pipeline import store_img, ImageProcessingPipeline
from rtm_pymmcore.core.data_structures import FovState, ImgType
from rtm_pymmcore.core.dmd import DMD

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
        if self.pipeline is not None:
            self.pipeline._analyzer = self
        self.fov_states: dict[int, FovState] = {}
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

    def get_fov_state(self, fov_index: int) -> FovState:
        """Return the FovState for *fov_index*, creating it lazily if needed."""
        if fov_index not in self.fov_states:
            self.fov_states[fov_index] = FovState()
        return self.fov_states[fov_index]

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

                if metadata.get("stim", False):
                    if (
                        not self.pipeline.stimulator.use_labels
                        and self.pipeline.stimulator.use_imgs
                    ):
                        img_type = metadata["img_type"]
                        if img_type == ImgType.IMG_RAW:
                            # stim mask does not require to use segmented cell labels.
                            self._put_stim_mask_if_no_labels(
                                {}, metadata=metadata, img=img
                            )
                        elif img_type == ImgType.IMG_OPTOCHECK:
                            img_raw = self._optocheck_to_raw_img(img, metadata=metadata)
                            self._put_stim_mask_if_no_labels(
                                {}, metadata=metadata, img=img_raw
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

    def _optocheck_to_raw_img(self, img: np.array, metadata: dict):
        len_raw_img = len(metadata["channels"])
        return img[0:len_raw_img]

    def _do_store(self, img: np.array, metadata: dict, folder: str):
        """Actually store image to disk (guaranteed, never skipped)."""
        img_type = metadata["img_type"]

        if img_type == ImgType.IMG_RAW:
            store_img(img, metadata, self.pipeline.storage_path, "raw")

        elif img_type == ImgType.IMG_STIM:
            store_img(img, metadata, self.pipeline.storage_path, "stim")

        elif img_type == ImgType.IMG_OPTOCHECK:
            img_raw = self._optocheck_to_raw_img(img, metadata=metadata)

            store_img(img_raw, metadata, self.pipeline.storage_path, "raw")
            if not os.path.exists(
                os.path.join(self.pipeline.storage_path, "optocheck")
            ):
                os.makedirs(os.path.join(self.pipeline.storage_path, "optocheck"))
            store_img(img, metadata, self.pipeline.storage_path, "optocheck")

    def _put_stim_mask_if_no_labels(
        self, label_images: dict, metadata: dict, img: np.array
    ) -> np.ndarray:
        """Generate stimulation mask if stim mask does not use cell labels."""
        if self.pipeline is None or self.pipeline.stimulator is None:
            raise RuntimeError(
                "No pipeline or stimulator defined for generating stim mask."
            )
        fov_state = self.get_fov_state(metadata["fov"])
        stim_mask, _ = self.pipeline.stimulator.get_stim_mask(
            label_images, metadata=metadata, img=img
        )
        fov_state.stim_mask_queue.put(stim_mask)

        return

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
        self._queue = queue
        self._analyzer = analyzer
        self._results: dict = {}
        self._current_group = mmc.getChannelGroup()
        self._dmd = dmd
        self._mmc = mmc
        self.use_autofocus_event = use_autofocus_event
        self.dmd_needs_to_be_waken = dmd_needs_to_be_waken
        self._n_channels: int = 1
        self._frame_buffers: dict[tuple, list] = {}
        self._mmc.mda.events.frameReady.connect(self._on_frame_ready)

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        meta = event.metadata or {}
        img_type = meta.get("img_type", ImgType.IMG_RAW)

        if self._analyzer.debug:
            try:
                tp = (event.index.get("t", 0), event.index.get("p", 0))
                print(
                    f"[Controller] frameReady: img_type={img_type} tp={tp} fname={meta.get('fname')}"
                )
            except Exception:
                pass

        # Stim frames: process immediately (single image, not multi-channel)
        if img_type == ImgType.IMG_STIM:
            self._results = self._analyzer.run(img[np.newaxis, ...], event)
            return

        # Imaging: buffer by (t, p), submit when all channels received
        tp = (event.index.get("t", 0), event.index.get("p", 0))
        buf = self._frame_buffers.setdefault(tp, [])
        buf.append(img)

        if len(buf) >= self._n_channels:
            frame = np.stack(buf, axis=0)
            del self._frame_buffers[tp]
            self._results = self._analyzer.run(frame, event)

    def stop_run(self):
        self._queue.put(self.STOP_EVENT)
        self._mmc.mda.cancel()
        self._analyzer.shutdown(wait=True)
        self._mmc.mda.events.frameReady.disconnect(self._on_frame_ready)

    def is_running(self):
        return self._queue.qsize() > 0

    @staticmethod
    def _get_power_prop(device_name, property_name, power):
        if any(v is None for v in (device_name, property_name, power)):
            return None
        return (device_name, property_name, power)

    def _make_slm(self, exposure) -> SLMImage | None:
        if self._dmd is not None and self.dmd_needs_to_be_waken:
            return SLMImage(data=True, device=self._dmd.name, exposure=exposure)
        return None

    def _resolve_group(self, config_name: str) -> str:
        """Return the channel group for *config_name*, auto-detecting if needed."""
        if self._current_group:
            return self._current_group
        # getChannelGroup() was empty — find a group containing this preset
        for group in self._mmc.getAvailableConfigGroups():
            if config_name in self._mmc.getAvailableConfigs(group):
                self._current_group = group
                return group
        return ""

    def _put_event(self, event: MDAEvent) -> None:
        """Queue an MDA event."""
        self._queue.put(event)

    def _make_metadata(self, row) -> dict:
        """Serializable base metadata from a DataFrame row."""
        return {k: v for k, v in row.items() if k != "fov_object"}

    def _queue_channels(self, row, metadata: dict):
        channels = row["channels"]
        self._n_channels = len(channels)
        for i, ch in enumerate(channels):
            power_prop = self._get_power_prop(
                ch.get("device_name"), ch.get("property_name"), ch.get("power")
            )
            self._put_event(useq.MDAEvent(
                index={"t": row["timestep"], "c": i, "p": row["fov"]},
                channel={"config": ch["name"], "group": ch.get("group") or self._resolve_group(ch["name"])},
                metadata={**metadata, "img_type": ImgType.IMG_RAW},
                x_pos=row["fov_x"] if i == 0 else None,
                y_pos=row["fov_y"] if i == 0 else None,
                z_pos=row.get("fov_z"),
                min_start_time=float(row["time"]),
                exposure=ch.get("exposure"),
                properties=[power_prop] if power_prop else None,
                slm_image=self._make_slm(ch.get("exposure")),
            ))

    def _queue_optocheck(self, row, metadata: dict):
        optocheck_channels = row["optocheck_channels"]
        meta_base = {**metadata, "img_type": ImgType.IMG_OPTOCHECK}
        for i, ch in enumerate(optocheck_channels):
            power_prop = self._get_power_prop(
                ch.get("device_name"), ch.get("property_name"), ch.get("power")
            )
            self._put_event(useq.MDAEvent(
                index={"t": row["timestep"], "c": i, "p": row["fov"]},
                channel={"config": ch["name"], "group": ch.get("group") or self._resolve_group(ch["name"])},
                metadata=meta_base,
                x_pos=None, y_pos=None,
                z_pos=row.get("fov_z"),
                min_start_time=float(row["time"]),
                exposure=ch.get("exposure"),
                properties=[power_prop] if power_prop else None,
                slm_image=self._make_slm(ch.get("exposure")),
            ))

    def _queue_stim(self, row, metadata: dict, fov_obj: FovState):
        if not row.get("stim_power") or not row.get("stim_exposure"):
            return
        stim_exposure = row.get("stim_exposure")
        meta = {**metadata, "img_type": ImgType.IMG_STIM}
        slm_image = None
        if self._dmd is not None:
            stimulator = self._analyzer.pipeline.stimulator
            if not stimulator.use_labels and not stimulator.use_imgs:
                stim_mask, _ = stimulator.get_stim_mask({}, metadata=meta, img=None)
                stim_mask = self._dmd.affine_transform(stim_mask)
            else:
                try:
                    stim_mask = fov_obj.stim_mask_queue.get(block=True, timeout=80)
                    stim_mask = self._dmd.affine_transform(stim_mask)
                except (TimeoutError, QueueEmpty) as e:
                    print(f"Warning: Stimulation mask not ready (timeout): {str(e)}")
                    stim_mask = False
            slm_image = SLMImage(data=stim_mask, device=self._dmd.name, exposure=stim_exposure)
        power_prop = self._get_power_prop(
            row.get("stim_channel_device_name"),
            row.get("stim_channel_power_property_name"),
            row.get("stim_power"),
        )
        self._put_event(useq.MDAEvent(
            index={"t": row["timestep"], "p": row["fov"]},
            channel={"config": row["stim_channel_name"], "group": row.get("stim_channel_group") or self._resolve_group(row["stim_channel_name"])},
            metadata=meta,
            x_pos=None, y_pos=None,
            z_pos=row.get("fov_z"),
            exposure=stim_exposure,
            min_start_time=float(row["time"]),
            properties=[power_prop] if power_prop else None,
            slm_image=slm_image,
        ))

    def run(self, events=None, *, df_acquire=None, stim_mode="current"):
        """Run the acquisition.

        Args:
            events: Iterable of RTMEvent (primary path).
            df_acquire: Legacy DataFrame path (backwards compat).
            stim_mode: How stim masks are resolved when the stimulator needs
                labels or images (ignored when ``use_labels=False`` and
                ``use_imgs=False``).

                * ``"current"`` – acquire the imaging frame, wait for the
                  pipeline to segment it and produce the mask, then stimulate
                  within the same timepoint.  Higher latency but the mask
                  matches the exact cell positions.
                * ``"previous"`` – stimulate using the mask produced from the
                  *previous* timepoint (for the same FOV).  The pipeline runs
                  in the background while the next frame is acquired, so
                  there is no blocking wait.  The first stim-eligible frame
                  for each FOV is skipped (no previous mask exists).
        """
        if events is not None:
            self._run_from_events(events, stim_mode=stim_mode)
        elif df_acquire is not None:
            self._run_from_df(df_acquire)
        else:
            raise ValueError("Either events or df_acquire must be provided")

    def _run_from_events(self, events, *, stim_mode="current"):
        """Primary acquisition path using RTMEvent objects.

        ``stim_mode="current"``
            acquire → wait for mask → stim  (within the same timepoint)

        ``stim_mode="previous"``
            stim (mask from t-1) → acquire t  (pipeline processes t in background)
            When revisiting the FOV at t+1, the mask from t is collected
            (blocking if not yet ready).  The first stim frame per FOV is
            skipped (no previous mask exists).
        """
        from rtm_pymmcore.core.data_structures import ImgType as _IT

        queue_sequence = iter(self._queue.get, self.STOP_EVENT)
        mda_thread = self._mmc.run_mda(queue_sequence)

        # For "previous" mode: track which FOVs have had a stim frame
        # queued so we know when to expect a mask from the pipeline.
        _stim_pending: set[int] = set()

        try:
            for rtm_event in events:
                while self._queue.qsize() >= 3:
                    time.sleep(0.1)
                self._n_channels = len(rtm_event.channels) + len(rtm_event.optocheck_channels)

                has_stim = len(rtm_event.stim_channels) > 0
                fov_index = rtm_event.index.get("p", 0)

                # Convert to MDAEvents (stim_slm_image=None; we fill it below)
                mda_events = rtm_event.to_mda_events(
                    resolve_group=self._resolve_group,
                    stim_slm_image=None,
                )
                img_events = [e for e in mda_events if e.metadata.get("img_type") != _IT.IMG_STIM]
                stim_events = [e for e in mda_events if e.metadata.get("img_type") == _IT.IMG_STIM]

                if stim_mode == "previous":
                    # --- "previous" mode ---
                    # Stim with mask from t-1, then acquire t.
                    # The pipeline processes t in the background; its mask
                    # is collected the next time we visit this FOV.

                    # 1. Stim (mask from previous frame)
                    if has_stim and stim_events and self._dmd:
                        stimulator = self._analyzer.pipeline.stimulator
                        if not stimulator.use_labels and not stimulator.use_imgs:
                            # No pipeline dependency — compute immediately
                            slm = self._build_stim_slm(rtm_event)
                            for ev in stim_events:
                                ev = ev.model_copy(update={"slm_image": slm})
                                self._put_event(ev)
                        elif fov_index in _stim_pending:
                            # A previous frame was queued for this FOV —
                            # block until its mask is ready (must be from t-1).
                            fov_state = self._analyzer.get_fov_state(fov_index)
                            try:
                                stim_mask = fov_state.stim_mask_queue.get(
                                    block=True, timeout=80
                                )
                                stim_mask = self._dmd.affine_transform(stim_mask)
                                stim_ch = rtm_event.stim_channels[0]
                                slm = SLMImage(
                                    data=stim_mask,
                                    device=self._dmd.name,
                                    exposure=stim_ch.exposure,
                                )
                                for ev in stim_events:
                                    ev = ev.model_copy(update={"slm_image": slm})
                                    self._put_event(ev)
                            except Exception as e:
                                print(f"Warning: Stimulation mask not ready (timeout): {e}")
                        # else: first stim frame for this FOV — skip (no previous mask)

                    # 2. Queue imaging events (never blocks)
                    for ev in img_events:
                        self._put_event(ev)

                    if has_stim:
                        _stim_pending.add(fov_index)

                else:
                    # --- "current" mode (default) ---
                    # 1. Queue imaging events first
                    for ev in img_events:
                        self._put_event(ev)

                    # 2. Wait for mask, then queue stim events
                    if has_stim and stim_events:
                        stim_slm_image = None
                        if self._dmd:
                            stim_slm_image = self._build_stim_slm(rtm_event)
                        for ev in stim_events:
                            if stim_slm_image is not None:
                                ev = ev.model_copy(update={"slm_image": stim_slm_image})
                            self._put_event(ev)
        finally:
            self._queue.put(self.STOP_EVENT)
            if mda_thread is not None:
                mda_thread.join()
            self._mmc.mda.events.frameReady.disconnect(self._on_frame_ready)

    def _build_stim_slm(self, rtm_event) -> SLMImage | None:
        """Build SLMImage for stimulation from fov_state mask queue or stimulator."""
        stimulator = self._analyzer.pipeline.stimulator
        fov_index = rtm_event.index.get("p", 0)
        fov_state = self._analyzer.get_fov_state(fov_index)
        stim_ch = rtm_event.stim_channels[0]
        stim_exposure = stim_ch.exposure

        meta = {**rtm_event.metadata, "fov": fov_index,
                "timestep": rtm_event.index.get("t", 0)}

        if not stimulator.use_labels and not stimulator.use_imgs:
            stim_mask, _ = stimulator.get_stim_mask({}, metadata=meta, img=None)
            stim_mask = self._dmd.affine_transform(stim_mask)
        else:
            try:
                stim_mask = fov_state.stim_mask_queue.get(block=True, timeout=80)
                stim_mask = self._dmd.affine_transform(stim_mask)
            except Exception as e:
                print(f"Warning: Stimulation mask not ready (timeout): {str(e)}")
                stim_mask = False

        return SLMImage(data=stim_mask, device=self._dmd.name, exposure=stim_exposure)

    def _run_from_df(self, df_acquire: pd.DataFrame):
        """Legacy acquisition path using df_acquire DataFrame."""
        queue_sequence = iter(self._queue.get, self.STOP_EVENT)
        mda_thread = self._mmc.run_mda(queue_sequence)
        try:
            for _, row in df_acquire.iterrows():
                while self._queue.qsize() >= 3:
                    time.sleep(0.1)

                metadata = self._make_metadata(row)

                if self.use_autofocus_event:
                    self._put_event(useq.MDAEvent(
                        index={"t": row["timestep"], "c": 0, "p": row["fov"]},
                        x_pos=row["fov_x"], y_pos=row["fov_y"], z_pos=row.get("fov_z"),
                        min_start_time=float(row["time"]),
                        action=HardwareAutofocus(autofocus_motor_offset=row.get("fov_af_offset")),
                    ))

                self._queue_channels(row, metadata)

                if row.get("optocheck", False):
                    self._n_channels += len(row["optocheck_channels"])
                    self._queue_optocheck(row, metadata)

                if row.get("stim", False):
                    fov_state = self._analyzer.get_fov_state(row["fov"])
                    self._queue_stim(row, metadata, fov_state)
        finally:
            self._queue.put(self.STOP_EVENT)
            if mda_thread is not None:
                mda_thread.join()
            self._mmc.mda.events.frameReady.disconnect(self._on_frame_ready)


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

        Uses channel counting (same as real Controller) but loads images from
        the project path instead of from the microscope.
        """
        meta = event.metadata or {}
        img_type = meta.get("img_type", ImgType.IMG_RAW)

        if img_type == ImgType.IMG_STIM:
            return  # Stim images are not processed in this simulation

        # Buffer by (t, p), submit when all channels received
        tp = (event.index.get("t", 0), event.index.get("p", 0))
        buf = self._frame_buffers.setdefault(tp, [])
        buf.append(img)

        if len(buf) >= self._n_channels:
            del self._frame_buffers[tp]
            fname = meta["fname"]

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

            else:
                raise ValueError(f"Unknown image type: {img_type}")

            try:
                print(
                    f"[ControllerSimulated] frameReady: img_type={img_type} fname={meta.get('fname')}"
                )
            except Exception:
                pass
