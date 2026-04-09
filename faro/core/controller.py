from faro.core.pipeline import store_img, ImageProcessingPipeline
from faro.core.data_structures import FovState, ImgType, StimMode
from faro.core.writers import (
    Writer,
    TiffWriter,
    OmeZarrWriter,
    OmeZarrWriterPlate,
    _extract_positions_from_events,
    _extract_channel_names_from_events,
    _extract_n_timepoints_from_events,
    _extract_n_stim_channels_from_events,
)
from faro.stimulation.base import Stim, StimWithImage, StimWithPipeline

import threading
import traceback
from faro.core._useq_compat import SLMImage
from useq import MDAEvent
from queue import Queue, Empty as QueueEmpty
import numpy as np
import time
import tifffile
import os
from concurrent.futures import ThreadPoolExecutor


class Analyzer:
    """Image analyzer with priority: Get -> Store >> Pipeline.

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
        writer: Writer | None = None,
        debug: bool = False,
        debug_every: int = 10,
    ):
        """
        Args:
            pipeline: ImageProcessingPipeline instance (optional for analysis)
            max_workers: Number of worker threads for pipeline (default: 4)
            max_queue_size: Maximum images in executor queue before deferring (default: 60)
            writer: Storage backend. Defaults to TiffWriter if pipeline has storage_path.
        """
        self.pipeline = pipeline
        if writer is not None:
            self.writer = writer
        elif pipeline is not None:
            self.writer = TiffWriter(pipeline.storage_path)
        else:
            self.writer = None
        if self.pipeline is not None:
            self.pipeline._analyzer = self
            self.pipeline._writer = self.writer
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
        # Stop event must be initialized BEFORE starting threads
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

    @property
    def stimulator_needs_data(self) -> bool:
        """True if stim masks come from the mask queue (StimWithImage/StimWithPipeline).

        False if generated from metadata alone (base Stim) or no stimulator configured.
        """
        if self.pipeline is None or self.pipeline.stimulator is None:
            return False
        return isinstance(self.pipeline.stimulator, (StimWithImage, StimWithPipeline))

    def get_stim_mask(
        self, fov_index: int, metadata: dict, *, timeout: float = 80
    ) -> np.ndarray | None:
        """Return a stim mask array, or None if unavailable.

        Dispatches by stimulator type:
        - Queue-based (StimWithImage / StimWithPipeline): blocks on fov_state.stim_mask_queue.
        - Metadata-only (base Stim): calls stimulator.get_stim_mask() directly.
        - No stimulator: returns None.
        """
        if self.pipeline is None or self.pipeline.stimulator is None:
            return None
        stimulator = self.pipeline.stimulator
        if isinstance(stimulator, (StimWithImage, StimWithPipeline)):
            fov_state = self.get_fov_state(fov_index)
            try:
                return fov_state.stim_mask_queue.get(block=True, timeout=timeout)
            except Exception as e:
                print(f"Warning: Stimulation mask not ready (timeout): {e}")
                return None
        else:
            metadata["img_shape"] = metadata.get("img_shape", (1024, 1024))
            stim_mask, _ = stimulator.get_stim_mask(metadata=metadata)
            return stim_mask

    def _storage_worker(self):
        """Worker thread for storage - high priority, never skipped.

        Drains all remaining items before exiting after ``_stop_event`` is set,
        so no queued images are silently dropped.
        """
        while True:
            try:
                img, event, metadata, folder = self._storage_queue.get(timeout=0.5)
            except QueueEmpty:
                if self._stop_event.is_set():
                    break
                continue

            try:
                # PRIORITY 1: Always store the image
                self._do_store(img, metadata, folder)
                self.stored_images += 1

                if self.debug:
                    print(
                        f"[Analyzer] Stored image type={metadata.get('img_type')} t={metadata.get('timestep')} fov={metadata.get('fov')} pending_storage={self._storage_queue.qsize()}"
                    )

                if metadata.get("stim", False):
                    if isinstance(self.pipeline.stimulator, StimWithImage):
                        if metadata["img_type"] == ImgType.IMG_RAW:
                            self._put_stim_mask_if_no_labels(metadata=metadata, img=img)

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

    def _do_store(self, img: np.array, metadata: dict, folder: str) -> None:
        """Store image to disk (guaranteed, never skipped)."""
        if self.writer is None:
            return

        img_type = metadata["img_type"]

        if img_type == ImgType.IMG_RAW:
            self.writer.write(img, metadata, "raw")

        elif img_type == ImgType.IMG_STIM:
            self.writer.write(img, metadata, "stim")

        elif img_type == ImgType.IMG_REF:
            self.writer.write(img, metadata, "ref")

    def _put_stim_mask_if_no_labels(
        self,
        metadata: dict,
        img: np.ndarray = None,
    ) -> None:
        """Generate stimulation mask if stim mask does not use cell labels."""
        if self.pipeline is None or self.pipeline.stimulator is None:
            raise RuntimeError(
                "No pipeline or stimulator defined for generating stim mask."
            )
        stimulator = self.pipeline.stimulator
        fov_state = self.get_fov_state(metadata["fov"])
        if isinstance(stimulator, StimWithImage):
            stim_mask, _ = stimulator.get_stim_mask(metadata=metadata, img=img)
        else:
            # Base Stim — needs nothing
            metadata["img_shape"] = (img.shape[-2], img.shape[-1])
            stim_mask, _ = stimulator.get_stim_mask(metadata=metadata)
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
                            f"[Analyzer] Pipeline overloaded -> defer (active={self.active_pipeline_tasks}, max={self.max_queue_size}, pending_deferred={self._deferred_queue.qsize()})"
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
        Drains all remaining items before exiting after ``_stop_event`` is set.
        During shutdown the capacity check is skipped so the queue drains
        instead of spinning on requeue.
        """
        while True:
            try:
                event, metadata, folder = self._deferred_queue.get(timeout=1.0)
            except QueueEmpty:
                if self._stop_event.is_set():
                    break
                continue

            try:
                # During shutdown, skip capacity check to ensure the queue drains.
                shutting_down = self._stop_event.is_set()
                with self.task_lock:
                    if (
                        not shutting_down
                        and self.active_pipeline_tasks >= self.max_queue_size
                    ):
                        # Still overloaded - put back in queue and wait
                        self._deferred_queue.put_nowait((event, metadata, folder))
                        if self.debug:
                            print(
                                f"[Analyzer] Still overloaded -> requeue deferred (active={self.active_pipeline_tasks}, max={self.max_queue_size})"
                            )
                        time.sleep(0.5)
                        continue

                    # Capacity available (or shutting down) - increment counter
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
                    if not shutting_down:
                        # Put back in queue for retry (only when not shutting down)
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
        """Shutdown storage thread, deferred thread, and pipeline executor.

        Workers drain their queues before exiting, so setting ``_stop_event``
        first and then joining the threads guarantees no queued items are lost.
        """
        self._stop_event.set()

        if wait:
            # Workers drain remaining items before exiting, so joining
            # the threads is sufficient (no queue.join() needed).
            self._storage_thread.join(timeout=30)
            self._deferred_thread.join(timeout=30)

        self.executor.shutdown(wait=wait)

        if self.writer is not None:
            self.writer.close()

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
    """Experiment orchestrator.

    Converts RTMEvents to MDAEvents, queues them through the microscope's
    MDA runner, and dispatches acquired frames to the Analyzer.

    The Controller accesses hardware exclusively through the microscope's
    abstract interface (run_mda, connect/disconnect_frame, cancel_mda,
    resolve_group, resolve_power) and never imports pymmcore-plus.
    """

    STOP_EVENT = object()

    def __init__(self, mic, pipeline, *, writer: Writer | None = None):
        """
        Args:
            mic: AbstractMicroscope instance (hardware + config).
            pipeline: ImageProcessingPipeline instance.
            writer: Storage backend. If None, Analyzer uses TiffWriter (default).
                Pass an OmeZarrWriter for OME-Zarr output.
        """
        self._mic = mic
        self._pipeline = pipeline
        self._writer = writer
        self._queue: Queue = Queue()
        self._analyzer: Analyzer | None = None
        self._n_channels: int = 1
        self._frame_buffers: dict[tuple, list] = {}
        self._ref_imaging_cache: dict[tuple, np.ndarray] = {}

        # Continuation state
        self._t_offset: int = 0
        self._time_offset: float = 0.0
        self._experiment_start: float | None = None
        self._event_queue: Queue | None = None  # for extend_experiment
        self._pending_sentinels: int = 0  # number of None sentinels yet to consume
        self._fov_positions: dict[int, tuple[float, float, float]] = {}
        self._pre_loop_hook: callable | None = None  # testing hook
        self._all_events: list = []  # accumulated events for JSON persistence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_events(self, events) -> bool:
        """Validate events against both pipeline and hardware.

        Combines ``pipeline.validate_pipeline(events)`` (signatures +
        required metadata) with ``mic.validate_hardware(events)`` (channel
        configs, exposure/power limits).

        Returns True if **all** checks pass, False otherwise.
        """
        ok = True
        if self._pipeline is not None:
            ok = self._pipeline.validate_pipeline(events) and ok
        ok = self._mic.validate_hardware(events) and ok
        return ok

    def run_experiment(self, events, *, stim_mode="current", validate=True):
        """Run an acquisition from a list of RTMEvents.

        Args:
            events: Iterable of RTMEvent.  Will be materialised to a list
                when ``validate=True`` so it can be iterated twice.
            stim_mode: How stim masks are resolved when the stimulator needs
                labels or images.

                * ``"current"`` -- acquire the imaging frame, wait for the
                  pipeline to segment it and produce the mask, then stimulate
                  within the same timepoint.
                * ``"previous"`` -- stimulate using the mask produced from the
                  *previous* timepoint (for the same FOV).
            validate: Run :meth:`validate_events` before starting.  Set to
                ``False`` to skip (e.g. if you already validated manually).
        """
        events = list(events)
        if validate:
            if not self.validate_events(events):
                raise ValueError(
                    "Event validation failed (see warnings above). "
                    "Fix the issues or pass validate=False to skip."
                )

        if self._experiment_start is None:
            self._experiment_start = time.monotonic()

        # Pre-compute offset so extend_experiment can use it during the run
        if events:
            self._t_offset = max(e.index.get("t", 0) for e in events) + 1

        # Persist events to storage as JSON
        self._all_events = list(events)
        if self._writer is not None:
            self._writer.save_events(self._all_events)

        # Initialize writer stream with values derived from events + microscope
        if (
            isinstance(self._writer, OmeZarrWriter)
            and self._writer._stream is None
            and self._writer._raw_array is None
        ):
            self._writer.init_stream(
                position_names=_extract_positions_from_events(events),
                channel_names=_extract_channel_names_from_events(events),
                image_height=self._mic.mmc.getImageHeight(),
                image_width=self._mic.mmc.getImageWidth(),
                n_timepoints=_extract_n_timepoints_from_events(events),
                n_stim_channels=_extract_n_stim_channels_from_events(events),
            )

        self._analyzer = Analyzer(self._pipeline, writer=self._writer)
        self._validate_fov_positions(events)
        self._run_mda_with_events(events, stim_mode=stim_mode)

        # Update wall-clock offset for continuation
        self._time_offset = time.monotonic() - self._experiment_start

    def continue_experiment(self, events, *, stim_mode="current", validate=True):
        """Continue acquisition with new events, preserving all pipeline state.

        Reuses the existing ``Analyzer`` (and its ``FovState`` objects) so
        that tracking, timestep counters, and filenames continue seamlessly
        from the previous ``run_experiment()`` or ``continue_experiment()``
        call.

        Args:
            events: Iterable of RTMEvent.  Timesteps and metadata will be
                offset automatically.
            stim_mode: Same as :meth:`run_experiment`.
            validate: Same as :meth:`run_experiment`.

        Raises:
            RuntimeError: If no previous experiment exists to continue.
        """
        if self._analyzer is None:
            raise RuntimeError(
                "No experiment to continue. Call run_experiment() first."
            )

        events = list(events)
        if validate:
            if not self.validate_events(events):
                raise ValueError(
                    "Event validation failed (see warnings above). "
                    "Fix the issues or pass validate=False to skip."
                )

        offset_events = self._offset_events(events)

        # Pre-compute offset so extend_experiment can use it during the run
        if offset_events:
            self._t_offset = max(e.index.get("t", 0) for e in offset_events) + 1

        # Append to accumulated events and persist
        self._all_events.extend(offset_events)
        if self._writer is not None:
            self._writer.save_events(self._all_events)

        self._validate_fov_positions(offset_events)
        self._run_mda_with_events(offset_events, stim_mode=stim_mode)

        # Update wall-clock offset for continuation
        self._time_offset = time.monotonic() - self._experiment_start

    def extend_experiment(self, events):
        """Add more events to a running experiment (non-blocking).

        The events are offset and pushed into the internal event queue so
        the running event loop picks them up.

        Raises:
            RuntimeError: If no experiment is currently running.
        """
        if self._event_queue is None:
            raise RuntimeError("No running experiment to extend.")

        events = list(events)
        offset_events = self._offset_events(events)
        # Add events + sentinel; bump counter so the loop keeps going
        self._pending_sentinels += 1
        for ev in offset_events:
            self._event_queue.put(ev)
        self._event_queue.put(None)  # sentinel for this batch

        # Update offset for future extensions
        if offset_events:
            self._t_offset = max(e.index.get("t", 0) for e in offset_events) + 1

    def finish_experiment(self):
        """Shutdown the Analyzer and reset continuation state.

        Call after all ``run_experiment`` / ``continue_experiment`` calls
        are done.
        """
        if self._analyzer is not None:
            self._analyzer.shutdown(wait=True)
            self._analyzer = None
        self._t_offset = 0
        self._time_offset = 0.0
        self._experiment_start = None
        self._event_queue = None
        self._all_events.clear()
        self._fov_positions.clear()
        self._frame_buffers.clear()
        self._ref_imaging_cache.clear()

    def stop_run(self):
        self._queue.put(self.STOP_EVENT)
        self._mic.cancel_mda()
        if self._analyzer is not None:
            self._analyzer.shutdown(wait=True)
        self._mic.disconnect_frame(self._on_frame_ready)
        self._frame_buffers.clear()

    # ------------------------------------------------------------------
    # Internal helpers for continuation
    # ------------------------------------------------------------------

    def _offset_events(self, events):
        """Offset event timesteps and metadata for continuation."""
        offset_events = []
        for ev in events:
            new_t = ev.index.get("t", 0) + self._t_offset
            offset_events.append(
                ev.model_copy(
                    update={
                        "index": {**dict(ev.index), "t": new_t},
                        "metadata": {
                            **ev.metadata,
                            "time_offset": self._time_offset,
                        },
                    }
                )
            )
        return offset_events

    def _validate_fov_positions(self, events):
        """Warn if FOV positions changed between continuations."""
        import warnings

        for ev in events:
            fov = ev.index.get("p", 0)
            pos = (ev.x_pos, ev.y_pos, ev.z_pos)
            if fov in self._fov_positions:
                old = self._fov_positions[fov]
                if pos != old:
                    warnings.warn(
                        f"FOV {fov} position changed: {old} -> {pos}. "
                        f"Tracking continuity may be broken.",
                        UserWarning,
                        stacklevel=3,
                    )
            self._fov_positions[fov] = pos

    def _run_mda_with_events(self, events, *, stim_mode):
        """Run the MDA event loop — shared by run/continue_experiment."""
        self._mic.connect_frame(self._on_frame_ready)

        # Set up event queue for extend_experiment support.
        # _pending_sentinels tracks how many extra batches (from
        # extend_experiment) still need to be drained.
        self._event_queue = Queue()
        self._pending_sentinels = 0
        events = sorted(
            events, key=lambda e: (e.min_start_time or 0, e.index.get("p", 0))
        )
        for ev in events:
            self._event_queue.put(ev)
        self._event_queue.put(None)  # sentinel for this initial batch

        if self._pre_loop_hook is not None:
            self._pre_loop_hook()

        queue_sequence = iter(self._queue.get, self.STOP_EVENT)
        mda_thread = self._mic.run_mda(queue_sequence)

        # For "previous" mode: track which FOVs have had a stim frame
        _stim_pending: set[int] = set()

        try:
            while True:
                rtm_event = self._event_queue.get()
                if rtm_event is None:
                    # Sentinel consumed — stop only if no extension pending
                    if self._pending_sentinels > 0:
                        self._pending_sentinels -= 1
                        continue
                    break

                while self._queue.qsize() >= 3:
                    time.sleep(0.1)
                self._n_channels = len(rtm_event.channels)

                fov_index = rtm_event.index.get("p", 0)
                # "previous" mode at t=0: the analyzer has no prior
                # frame yet, so skip stim for this FOV until it's
                # acquired imaging once.
                suppress_stim = (
                    stim_mode == StimMode.PREVIOUS
                    and rtm_event.has_stim
                    and self._analyzer.stimulator_needs_data
                    and fov_index not in _stim_pending
                )

                for ev in rtm_event.plan_events(
                    stim_mode=stim_mode,
                    build_slm=self._build_stim_slm if self._mic.dmd else None,
                    resolve_group=self._mic.resolve_group,
                    resolve_power=self._mic.resolve_power,
                    suppress_stim=suppress_stim,
                ):
                    self._put_event(ev)

                if stim_mode == StimMode.PREVIOUS and rtm_event.has_stim:
                    _stim_pending.add(fov_index)
        finally:
            self._event_queue = None
            self._queue.put(self.STOP_EVENT)
            if mda_thread is not None:
                mda_thread.join()
            self._mic.disconnect_frame(self._on_frame_ready)

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        meta = event.metadata or {}
        img_type = meta.get("img_type", ImgType.IMG_RAW)

        if self._analyzer and self._analyzer.debug:
            try:
                tp = (event.index.get("t", 0), event.index.get("p", 0))
                print(
                    f"[Controller] frameReady: img_type={img_type} tp={tp} fname={meta.get('fname')}"
                )
            except Exception:
                pass

        # Stim frames: process immediately (single image)
        if img_type == ImgType.IMG_STIM:
            self._analyzer.run(img[np.newaxis, ...], event)
            return

        # Ref frames: stack with cached imaging channels from the same (t, p)
        if img_type == ImgType.IMG_REF:
            tp = (event.index.get("t", 0), event.index.get("p", 0))
            cached_imaging = self._ref_imaging_cache.pop(tp, None)
            if cached_imaging is not None:
                ref_frame = np.concatenate(
                    [cached_imaging, img[np.newaxis, ...]],
                    axis=0,
                )
            else:
                ref_frame = img[np.newaxis, ...]
            self._analyzer.run(ref_frame, event)
            return

        # Imaging: buffer by (t, p), submit when all channels received
        tp = (event.index.get("t", 0), event.index.get("p", 0))
        buf = self._frame_buffers.setdefault(tp, [])
        buf.append(img)

        if len(buf) >= self._n_channels:
            frame = np.stack(buf, axis=0)
            del self._frame_buffers[tp]
            # Cache imaging frame for ref channels that arrive later
            self._ref_imaging_cache[tp] = frame
            self._analyzer.run(frame, event)

    # ------------------------------------------------------------------
    # Stim helpers
    # ------------------------------------------------------------------

    def _build_stim_slm(self, rtm_event) -> SLMImage | None:
        """Build SLMImage for stimulation via Analyzer's stim-mask API."""
        fov_index = rtm_event.index.get("p", 0)
        stim_ch = rtm_event.stim_channels[0]

        meta = {
            **rtm_event.metadata,
            "fov": fov_index,
            "timestep": rtm_event.index.get("t", 0),
        }

        stim_mask = self._analyzer.get_stim_mask(fov_index, meta)
        if stim_mask is None:
            print("Warning: Stimulation mask unavailable, sending False to SLM.")
            stim_mask = False
        elif isinstance(stim_mask, np.ndarray):
            stim_mask = self._mic.dmd.affine_transform(stim_mask)

        return SLMImage(
            data=stim_mask, device=self._mic.dmd.name, exposure=stim_ch.exposure
        )

    @staticmethod
    def _make_slm(dmd, exposure, dmd_needs_to_be_waken) -> SLMImage | None:
        if dmd is not None and dmd_needs_to_be_waken:
            return SLMImage(data=True, device=dmd.name, exposure=exposure)
        return None

    def _put_event(self, event: MDAEvent) -> None:
        """Queue an MDA event."""
        self._queue.put(event)


class ControllerSimulated(Controller):
    """Controller that loads images from disk instead of from the camera.

    Supports both TIFF (``raw/``, ``ref/`` folders) and OME-Zarr
    (``acquisition.ome.zarr``) source layouts.  If an ``acquisition.ome.zarr``
    directory is found inside *old_data_project_path*, raw frames are read
    from the zarr store; reference images still fall back to TIFFs in
    ``ref/``.
    """

    def __init__(
        self, mic, pipeline, old_data_project_path: str, *, writer: Writer | None = None
    ):
        super().__init__(mic, pipeline, writer=writer)
        self._project_path = old_data_project_path

        # Detect OME-Zarr source
        zarr_path = os.path.join(old_data_project_path, "acquisition.ome.zarr")
        if os.path.isdir(zarr_path):
            import zarr

            self._zarr_store = zarr.open_group(zarr_path, mode="r")
            self._zarr_raw = self._zarr_store["0"]
            ome = self._zarr_store.attrs.get("ome", {})
            axes = ome.get("multiscales", [{}])[0].get("axes", [])
            self._zarr_axes = [a["name"] for a in axes]
        else:
            self._zarr_store = None

    def _read_zarr_raw(self, timestep: int, fov: int) -> np.ndarray:
        """Read a raw frame from the zarr store, returning (c, y, x)."""
        axes = self._zarr_axes
        has_p = "p" in axes
        has_c = "c" in axes
        arr = self._zarr_raw

        if has_p and has_c:
            img = np.asarray(arr[timestep, fov])
        elif has_p:
            img = np.asarray(arr[timestep, fov])[np.newaxis]
        elif has_c:
            img = np.asarray(arr[timestep])
        else:
            img = np.asarray(arr[timestep])[np.newaxis]
        return img

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        """Override to load images from disk for simulated controller."""
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
            t_idx = event.index.get("t", 0)
            p_idx = event.index.get("p", 0)

            if img_type == ImgType.IMG_RAW and self._zarr_store is not None:
                img_loaded = self._read_zarr_raw(t_idx, p_idx)
            else:
                # TIFF fallback (raw/) or always for ref images
                folder = {
                    ImgType.IMG_RAW: "raw",
                    ImgType.IMG_REF: "ref",
                }.get(img_type)
                if folder is None:
                    raise ValueError(f"Unknown image type: {img_type}")
                img_loaded = tifffile.imread(
                    os.path.join(self._project_path, folder, fname + ".tiff")
                )
            self._analyzer.run(img_loaded, event)

            try:
                print(
                    f"[ControllerSimulated] frameReady: img_type={img_type} fname={meta.get('fname')}"
                )
            except Exception:
                pass
