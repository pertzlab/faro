from faro.core.pipeline import store_img, ImageProcessingPipeline
from faro.core.data_structures import (
    FovState,
    FrameWaitCancelled,
    ImgType,
    StimMode,
    WaitEvent,
)
from faro.core.qt import pump_qt_events
from faro.core.run_status import RunHandle, RunStatus
from faro.core.writers import (
    Writer,
    TiffWriter,
    OmeZarrWriter,
    OmeZarrWriterPlate,
    OmeZarrRawReader,
    _extract_positions_from_events,
    _extract_channel_names_from_events,
    _extract_n_timepoints_from_events,
    _extract_n_stim_channels_from_events,
)
from faro.stimulation.base import Stim, StimWithImage, StimWithPipeline

import contextlib
import logging
import threading
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Literal
from faro.core._useq_compat import SLMImage
from psygnal import Signal
from useq import MDAEvent
from queue import Queue, Empty as QueueEmpty
import numpy as np
import time
import tifffile
import os
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)

# Background errors of one kind (source and exception type) that are kept
# with their traceback. Later ones are only counted.
KEEP_ERRORS_PER_KIND = 50


BackgroundErrorSource = Literal["storage", "deferred", "pipeline", "stim_mask"]


@dataclass(frozen=True)
class BackgroundError:
    """A background-thread exception recorded for later inspection."""

    source: BackgroundErrorSource
    exc_type: str
    message: str
    traceback: str


@dataclass(frozen=True)
class QueueStats:
    """Snapshot of the Analyzer's queue depths, for backpressure display.

    The three depths each flag a distinct way the analyzer can fall
    behind real time:

    * ``storage_*``  -- images buffered in RAM awaiting a disk write.
      Bounded; if it saturates, the camera buffer is at risk.
    * ``pipeline_*`` -- tracking/segmentation tasks submitted to the
      executor and not yet finished. At ``pipeline_max`` new frames
      start being deferred instead of run inline.
    * ``deferred_depth`` -- frames the pipeline could not keep up with,
      queued (metadata only) to be reloaded from disk and processed
      later. Unbounded; a steadily growing value means the pipeline is
      permanently behind.
    """

    storage_depth: int  # images buffered, awaiting disk write
    storage_max: int  # storage queue capacity
    pipeline_inflight: int  # pipeline tasks submitted, not yet finished
    pipeline_max: int  # depth at which new frames get deferred
    deferred_depth: int  # frames deferred for later reprocessing
    stored_images: int  # cumulative images written
    skipped_pipeline: int  # cumulative frames deferred
    deferred_processed: int  # cumulative deferred frames later processed


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
        stim_mask_timeout: float = 30,
    ):
        """
        Args:
            pipeline: ImageProcessingPipeline instance (optional for analysis)
            max_workers: Number of worker threads for pipeline (default: 4)
            max_queue_size: Maximum images in executor queue before deferring (default: 60)
            writer: Storage backend. Defaults to TiffWriter if pipeline has storage_path.
            stim_mask_timeout: Slip allowance, in seconds, past a stim event's
                scheduled due time to keep waiting for its mask before the
                controller records a background error and fires the all-off
                fallback. The source frame is always in the pipeline before
                the wait starts, so this bounds segmentation latency only.
                Increase for slow first-frame segmenters (cellpose SAM, remote).
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

        # Deferred pipeline queue: metadata-only. Frames that arrive while the
        # pipeline is saturated drop their in-RAM pixels and queue just the
        # event/metadata; the pixels are reloaded from storage when capacity
        # frees up (see _deferred_worker). This caps peak RAM at the in-flight
        # set rather than growing with the backlog. The reload goes through the
        # writer backend (writer.read_raw) so it works for TIFF *and* OME-Zarr.
        self._deferred_queue: Queue = Queue()
        self._deferred_thread = threading.Thread(
            target=self._deferred_worker, daemon=True, name="DeferredWorker"
        )
        self._deferred_thread.start()

        # Statistics for monitoring
        self.stored_images = 0
        self.skipped_pipeline = 0
        self.deferred_processed = 0

        # Experiments intentionally log-and-continue on background-thread
        # errors (one bad filter wheel shouldn't crash a 24h run). Tests
        # assert this list is empty at end of run to catch silent failures.
        # The list keeps the first errors of each kind; _error_counts counts
        # all of them (see _record_background_error).
        self.background_errors: list[BackgroundError] = []
        self._error_counts: Counter = Counter()
        self._error_lock = threading.Lock()
        self._stim_mask_timeout = stim_mask_timeout

        # Stim-mode coordination: set by Controller.run_experiment /
        # continue_experiment. Pipeline and storage workers read this to decide
        # whether non-stim frames need to compute+put a mask (previous mode
        # only) so the next stim event's t-1 peek has something to read.
        self.stim_mode: str | None = None

    def get_fov_state(self, fov_index: int) -> FovState:
        """Return the FovState for *fov_index*, creating it lazily if needed."""
        if fov_index not in self.fov_states:
            self.fov_states[fov_index] = FovState()
        return self.fov_states[fov_index]

    def cancel_pending_waits(self) -> None:
        """Wake the event stream if it is parked in :meth:`get_stim_mask`.

        ``get_stim_mask`` blocks on ``stim_mask_queue.wait_for_frame``
        for up to ``stim_mask_timeout`` seconds. Cancelling the
        dispensers makes that wait raise ``FrameWaitCancelled``
        immediately, so a cancelled run tears down without waiting out
        the timeout. Called (via ``Controller._cancel_stim_waits``)
        from ``RunHandle.cancel`` on the caller's thread.

        Only the stim-mask dispensers are cancelled: the tracks-queue
        waiters run on pipeline workers that ``shutdown`` already
        drains, and they have a file-based fallback for timeouts.
        """
        # Snapshot: the MDA thread may insert a new FovState via
        # get_fov_state() concurrently with this iteration.
        for fov_state in list(self.fov_states.values()):
            fov_state.stim_mask_queue.cancel()

    def queue_stats(self) -> "QueueStats":
        """Return a thread-safe snapshot of the current queue depths.

        Cheap enough to poll from a UI timer. ``Queue.qsize`` is
        approximate under concurrency but fine for a read-out.
        """
        with self.task_lock:
            inflight = self.active_pipeline_tasks
        return QueueStats(
            storage_depth=self._storage_queue.qsize(),
            storage_max=self._storage_queue.maxsize,
            pipeline_inflight=inflight,
            pipeline_max=self.max_queue_size,
            deferred_depth=self._deferred_queue.qsize(),
            stored_images=self.stored_images,
            skipped_pipeline=self.skipped_pipeline,
            deferred_processed=self.deferred_processed,
        )

    def _record_background_error(
        self, source: BackgroundErrorSource, exc: BaseException
    ) -> None:
        """Log a background-thread exception and record it for later inspection.

        Must be called from inside an active ``except`` block — the
        ``traceback.format_exc()`` call below reads the current
        exception context and returns ``'NoneType: None\\n'`` if
        there is none.

        Only the first ``KEEP_ERRORS_PER_KIND`` errors of each kind (source
        and exception type) are kept and printed with their traceback. A bug
        that fails on every frame would otherwise add one traceback per
        frame. Later errors of the same kind are only counted, with a short
        log line every 100. ``error_summary()`` gives the counts.
        """
        exc_type = type(exc).__name__
        msg = str(exc)
        kind = (source, exc_type)
        with self._error_lock:
            self._error_counts[kind] += 1
            n = self._error_counts[kind]
            if n <= KEEP_ERRORS_PER_KIND:
                tb = traceback.format_exc()
                self.background_errors.append(
                    BackgroundError(source, exc_type, msg, tb)
                )

        if n <= KEEP_ERRORS_PER_KIND:
            print(f"[Analyzer] {source} error: {exc_type}: {msg}")
            print(tb)
            logger.error("%s error: %s: %s\n%s", source, exc_type, msg, tb)
            if n == KEEP_ERRORS_PER_KIND:
                logger.error(
                    "%s error %s happened %d times. Further ones are only "
                    "counted (see error_summary()).",
                    source,
                    exc_type,
                    n,
                )
        elif n % 100 == 0:
            logger.error("%s error %s, %d times so far: %s", source, exc_type, n, msg)

    def error_summary(self) -> Counter:
        """Number of background errors per (source, exception type)."""
        with self._error_lock:
            return Counter(self._error_counts)

    @property
    def stimulator_needs_data(self) -> bool:
        """True if stim masks come from the mask queue (StimWithImage/StimWithPipeline).

        False if generated from metadata alone (base Stim) or no stimulator configured.
        """
        if self.pipeline is None or self.pipeline.stimulator is None:
            return False
        return isinstance(self.pipeline.stimulator, (StimWithImage, StimWithPipeline))

    def get_stim_mask(
        self, fov_index: int, metadata: dict, *, timeout: float | None = None
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
            if timeout is None:
                timeout = self._stim_mask_timeout
            frame_idx = metadata.get("timestep", 0)
            try:
                mask = fov_state.stim_mask_queue.wait_for_frame(
                    frame_idx, timeout=timeout
                )
            except FrameWaitCancelled:
                # Run is being cancelled — the dispenser was woken by
                # Analyzer.cancel_pending_waits(). Unwind quietly so the
                # event stream reaches its cancel check; not a failure,
                # so no background error is recorded.
                return None
            except QueueEmpty as e:
                # _build_stim_slm still log-and-continues with False, but
                # hardware tests check background_errors so the dropped stim
                # frame is no longer silent. Raise-then-catch so
                # _record_background_error's format_exc() sees the
                # TimeoutError with its QueueEmpty cause chain.
                try:
                    raise TimeoutError(
                        f"Stim mask not ready for FOV {fov_index} frame "
                        f"{frame_idx} after {timeout}s — pipeline didn't "
                        "produce one in time"
                    ) from e
                except TimeoutError as terr:
                    self._record_background_error("stim_mask", terr)
                return None
            if mask is None:
                # Pipeline explicitly skipped this frame (tracking/stim crashed).
                # A background_error was already recorded by the pipeline path.
                print(f"Warning: Stimulation mask skipped for frame {frame_idx}")
            return mask
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

                if (
                    self.pipeline is not None  # a run can have only a writer
                    and isinstance(self.pipeline.stimulator, StimWithImage)
                    and metadata["img_type"] == ImgType.IMG_RAW
                ):
                    # Always put in "previous" mode so the next stim event's
                    # t-1 peek finds a mask (even for non-stim predecessors).
                    # In "current" mode, only compute on stim frames — no
                    # consumer will ever peek non-stim frames.
                    if metadata.get("stim", False) or self.stim_mode == "previous":
                        self._put_stim_mask_if_no_labels(metadata=metadata, img=img)

                # PRIORITY 2: Pipeline only if resources available
                self._try_submit_pipeline(img, event, metadata, folder)

            except Exception as e:
                self._record_background_error("storage", e)
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
            # The bundled stack at a ref frame is [imaging | ref] (in
            # that channel order — see RTMSequence.to_mda_events). Only
            # the ref slice belongs in ref/ TIFF; the imaging slice goes
            # to "raw" alongside other timepoints, otherwise the last
            # frame in the main zarr is dark.
            n_channels = len(metadata.get("channels", ()))
            n_ref = len(metadata.get("ref_channels", ()))
            if n_ref > 0 and img.ndim == 3 and img.shape[0] > n_channels:
                self.writer.write(img[:n_channels], metadata, "raw")
                self.writer.write(img[n_channels:], metadata, "ref")
            else:
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
        frame_idx = metadata.get("timestep", 0)
        try:
            if isinstance(stimulator, StimWithImage):
                stim_mask, _ = stimulator.get_stim_mask(metadata=metadata, img=img)
            else:
                # Base Stim — needs nothing
                metadata["img_shape"] = (img.shape[-2], img.shape[-1])
                stim_mask, _ = stimulator.get_stim_mask(metadata=metadata)
        except Exception:
            fov_state.stim_mask_queue.skip_frame(frame_idx)
            raise
        fov_state.stim_mask_queue.put_for_frame(frame_idx, stim_mask)

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

        # Pure-stim pipelines have nothing for pipeline.run() to do —
        # tracking/FE/mask generation all require labels. The stim path
        # is driven directly by Analyzer.get_stim_mask().
        if self.pipeline.segmentators is None:
            return

        with self.task_lock:
            # Check if we have capacity for pipeline
            if self.active_pipeline_tasks >= self.max_queue_size:
                # Pipeline is overloaded - defer this image for later processing
                self.skipped_pipeline += 1
                # Queue metadata only — the image is reloaded from storage in
                # _deferred_worker (via writer.read_raw) so we don't hold the
                # full backlog of frames in RAM.
                try:
                    self._deferred_queue.put_nowait((event, metadata, folder))
                    if self.debug:
                        print(
                            f"[Analyzer] Pipeline overloaded -> defer (active={self.active_pipeline_tasks}, max={self.max_queue_size}, pending_deferred={self._deferred_queue.qsize()})"
                        )
                except Exception:
                    # Deferred queue full — frame truly lost. Mark it skipped
                    # on the tracks dispenser so downstream frames aren't stuck
                    # waiting for its put forever.
                    fov_idx = metadata.get("fov", 0)
                    frame_idx = event.index.get("t", 0)
                    self.get_fov_state(fov_idx).tracks_queue.skip_frame(frame_idx)
                return

            # We have capacity, increment counter
            self.active_pipeline_tasks += 1

        # Submit to pipeline with low priority
        try:
            # Optimization: Use memory if capacity available (faster than disk read)
            future = self.executor.submit(
                self.pipeline.run, img=img, event=event, file_path=None
            )
            future.add_done_callback(self._make_pipeline_done_callback(event, metadata))
            if self.debug:
                print(
                    f"[Analyzer] Pipeline submitted (active={self.active_pipeline_tasks}, pending_deferred={self._deferred_queue.qsize()})"
                )
        except (RuntimeError, OSError) as e:
            print(f"Could not submit pipeline task: {str(e)}")
            with self.task_lock:
                self.active_pipeline_tasks -= 1

    def _reload_frame(self, metadata: dict) -> np.ndarray:
        """Read a deferred frame back from storage, as the pipeline expects it.

        A reference frame is stored in two parts (see ``_do_store``): the
        imaging channels as "raw" and the reference channels as "ref". The
        pipeline expects both in one stack, imaging channels first.
        """
        img = self.writer.read_raw(metadata)
        if metadata.get("img_type") != ImgType.IMG_REF or not metadata.get(
            "ref_channels"
        ):
            return img
        ref = self.writer.read_ref(metadata)
        if img.ndim == 2:
            img = img[np.newaxis]
        if ref.ndim == 2:
            ref = ref[np.newaxis]
        return np.concatenate([img, ref], axis=0)

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
                        # Keep this sleep inside the lock. Holding the lock
                        # here blocks _try_submit_pipeline, so a freed
                        # pipeline slot goes to a deferred (older) frame and
                        # not to a new frame. Tracking needs frame N-1 before
                        # frame N. If new frames take the slots, they block
                        # in get_predecessor until _queue_timeout, and
                        # TestBurstNoSignalLoss fails. See TODO.md for a
                        # cleaner fix.
                        time.sleep(0.5)
                        continue

                    # Capacity available (or shutting down) - increment counter
                    self.active_pipeline_tasks += 1

                # Reload the frame's pixels from storage through the writer
                # backend. This is the crux of the deferral design: the in-RAM
                # array was dropped at defer time to bound memory, so it must be
                # read back from wherever it was written — a per-frame TIFF for
                # TiffWriter, or the OME-Zarr store for OmeZarrWriter. Asking the
                # writer (rather than hardcoding raw/<fname>.tiff) keeps this
                # correct across backends.
                try:
                    img = self._reload_frame(metadata)
                except Exception as e:
                    with self.task_lock:
                        self.active_pipeline_tasks -= 1
                    # The frame can't be reloaded (missing/not flushed/backend
                    # without read support). Record the cause first, then mark
                    # the frame skipped so downstream get_predecessor waiters
                    # resolve — ordering the skip last means anything observing
                    # "frame resolved" also sees the recorded error.
                    self._record_background_error("deferred", e)
                    fov_idx = metadata.get("fov", 0)
                    frame_idx = event.index.get("t", 0)
                    self.get_fov_state(fov_idx).tracks_queue.skip_frame(frame_idx)
                    continue

                # Submit deferred image to pipeline (loaded above).
                try:
                    future = self.executor.submit(
                        self.pipeline.run, img=img, event=event, file_path=None
                    )
                    future.add_done_callback(
                        self._make_pipeline_done_callback(event, metadata)
                    )
                    self.deferred_processed += 1
                    if self.debug:
                        print(
                            f"[Analyzer] Deferred submitted (deferred_processed={self.deferred_processed})"
                        )
                except (RuntimeError, OSError):
                    with self.task_lock:
                        self.active_pipeline_tasks -= 1
                    if not shutting_down:
                        # Put back in queue for retry (only when not shutting down)
                        self._deferred_queue.put_nowait((event, metadata, folder))

            except Exception as e:
                self._record_background_error("deferred", e)
            finally:
                # wait_idle counts the frame until here (a frame put back
                # above counts again as a new task).
                self._deferred_queue.task_done()

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

    def _make_pipeline_done_callback(self, event: MDAEvent, metadata: dict):
        """Build a Future done-callback bound to this frame's event/metadata.

        Capturing the frame's identity lets the callback release a downstream
        waiter if ``pipeline.run`` raised *before* reaching its own
        dispenser-skipping ``finally`` (e.g. an error during image load or
        frame setup). Without that, the tracks dispenser never resolves this
        frame and every later frame's ``get_predecessor`` blocks until it
        times out — the "FrameDispenser: timeout waiting for frame N" symptom.
        """
        return lambda f: self._pipeline_task_done(
            future=f, event=event, metadata=metadata
        )

    def _pipeline_task_done(self, future=None, event=None, metadata=None):
        """Called when pipeline task completes.

        Args:
            future: The Future object from the executor (if provided as callback arg)
            event: The frame's MDAEvent, used on error to release downstream waiters.
            metadata: The frame's metadata, used on error to resolve its FOV.
        """
        with self.task_lock:
            self.active_pipeline_tasks -= 1

        # Check if the task raised an exception
        if future is not None:
            try:
                future.result()  # This will re-raise any exception that occurred
            except Exception as e:
                self._record_background_error("pipeline", e)
                # The pipeline's own `finally` skips the tracks dispenser when a
                # frame fails mid-run, but only once it has reached that block.
                # An exception raised earlier (e.g. before frame setup) leaves
                # the frame unresolved, hanging every downstream predecessor
                # wait. Mark it skipped here as a backstop — skip never
                # overrides a real put (wait_for_frame/get_predecessor check
                # entries first), so this is safe even if the pipeline already
                # produced tracks before failing later.
                if event is not None and metadata is not None:
                    try:
                        fov_idx = metadata.get("fov", 0)
                        frame_idx = event.index.get("t", 0)
                        self.get_fov_state(fov_idx).tracks_queue.skip_frame(frame_idx)
                    except Exception:
                        pass

        if self.debug:
            print(
                f"[Analyzer] Pipeline task done (active={self.active_pipeline_tasks})"
            )

    def shutdown(self, wait: bool = True, *, drain_timeout: float = 300.0):
        """Shutdown storage thread, deferred thread, and pipeline executor.

        With ``wait=True``: drain the storage / deferred / pipeline queues
        first (workers still active), then signal stop, then join. The
        drain has a finite ``drain_timeout`` (default 300 s); if that
        elapses without the queues going idle, raises ``TimeoutError``
        BEFORE any teardown, so the call is safe to retry — typically
        with a larger ``drain_timeout`` after investigating
        ``get_stats()``.

        Without the up-front drain the storage thread could still be
        pulling items and submitting pipeline tasks when the 30 s
        join-timeout below fired, those late tasks would never reach the
        executor, and ``finish_experiment`` would return before per-FOV
        track parquets had been written — leaving
        ``generate_exp_data_from_tracks`` to crash on an empty
        ``tracks/`` with ``pd.concat([])``.
        """
        if wait:
            if not self.wait_idle(timeout=drain_timeout):
                stats = self.get_stats()
                raise TimeoutError(
                    f"Analyzer.shutdown: queues did not drain within "
                    f"{drain_timeout}s. State: {stats}. No teardown done — "
                    "call shutdown(drain_timeout=N) again with a larger "
                    "N if the experiment legitimately needs more time."
                )

        self._stop_event.set()

        if wait:
            # Watchdog only — wait_idle above already proved the queues
            # are empty, so the workers should exit on the next 0.5 s
            # poll once they see _stop_event.
            self._storage_thread.join(timeout=30)
            self._deferred_thread.join(timeout=30)

        self.executor.shutdown(wait=wait)

        if self.writer is not None:
            self.writer.close()

    def wait_idle(self, timeout: float | None = 30.0, poll: float = 0.05) -> bool:
        """Block until storage, pipeline, and deferred queues all drain.

        Returns True if idle was reached before the timeout, False
        otherwise. Pass ``timeout=None`` to wait indefinitely (used by
        ``shutdown(wait=True)`` so finish_experiment can't return while
        per-FOV track parquets are still being written).
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            # A frame counts until its worker calls task_done(), after it is
            # handed to the pipeline. qsize() alone would miss a frame that a
            # worker has taken from its queue but not handed over yet.
            storage_empty = self._storage_queue.unfinished_tasks == 0
            deferred_empty = self._deferred_queue.unfinished_tasks == 0
            with self.task_lock:
                pipeline_idle = self.active_pipeline_tasks == 0
            if storage_empty and deferred_empty and pipeline_idle:
                return True
            time.sleep(poll)
        return False

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

    Converts RTMEvents to MDAEvents, streams them to the microscope's
    MDA runner one pull at a time, and dispatches acquired frames to the
    Analyzer.

    The Controller accesses hardware exclusively through the microscope's
    abstract interface (run_mda, connect/disconnect_frame, cancel_mda,
    resolve_group, resolve_power) and never imports pymmcore-plus.
    """

    # Emitted on each new ``run_experiment`` / ``continue_experiment`` call,
    # carrying the freshly-created RunHandle. Widgets subscribe to this so
    # they can re-bind to whichever run is current.
    runStarted = Signal(object)

    # Emitted by ``load_experiment`` with the validated, sorted events list.
    # Status widgets subscribe to preview the run plan (event strip, FOV
    # map, planned duration) before the acquisition is started.
    experimentLoaded = Signal(object)

    def __init__(self, mic, pipeline, *, writer: Writer | None = None):
        """
        Args:
            mic: AbstractMicroscope instance (hardware + config).
            pipeline: ImageProcessingPipeline instance.
            writer: Storage backend. If None, Analyzer uses TiffWriter (default).
                Pass an OmeZarrWriter for OME-Zarr output.

        Note:
            ``run_experiment`` and ``continue_experiment`` are *non-blocking*
            in this version: they spawn a worker thread and return a
            :class:`RunHandle` immediately. Call ``handle.wait()`` to block
            until the run finishes, ``handle.cancel()`` to abort, or
            subscribe to ``handle.statusChanged`` for live updates. The
            ``runStarted`` signal on the controller fires for every new run
            so widgets can re-bind.
        """
        self._mic = mic
        self._pipeline = pipeline
        self._writer = writer
        self._analyzer: Analyzer | None = None
        self._n_channels: int = 1
        self._frame_buffers: dict[tuple, list] = {}

        # Stim events that fired the all-off fallback mask this run. Stamped
        # into RunStatus as it happens; the warning log carries the (t, p)
        # locations.
        self._n_stim_fallbacks: int = 0
        self._active_handle: RunHandle | None = None

        # Continuation state
        self._t_offset: int = 0
        self._time_offset: float = 0.0
        self._experiment_start: float | None = None
        self._event_queue: Queue | None = None  # for extend_experiment
        self._pending_sentinels: int = 0  # number of None sentinels yet to consume
        self._pending_sentinels_lock = threading.Lock()
        self._fov_positions: dict[int, tuple[float, float, float]] = {}
        self._pre_loop_hook: callable | None = None  # testing hook
        self._all_events: list = []  # accumulated events for JSON persistence

        # Background-thread errors harvested from the Analyzer on shutdown.
        # Survives finish_experiment() so tests/notebooks can inspect it.
        self.background_errors: list[BackgroundError] = []
        self._error_counts: Counter = Counter()  # all errors, see error_summary()

        # Fatal condition raised from the signal-callback thread, surfaced
        # through the handle's RunStatus.fatal_error.
        self._fatal_error: BaseException | None = None

        # Current run handle (None when no run is in progress / between runs).
        # The worker thread owns it; status update sites use it via this attr.
        self._current_handle: RunHandle | None = None

        # Staged (loaded-but-not-started) experiment: set by
        # load_experiment, consumed by start_experiment. (events, stim_mode).
        self._pending_run: tuple[list, str] | None = None

        # (p, t) index of the RTMEvent whose frames are currently arriving.
        # _bump_status_for_frame uses it to detect RTMEvent boundaries so
        # n_events_acquired / lag update once per RTMEvent, not per frame.
        self._rtm_key: tuple | None = None

        # monotonic-clock origin for lag, anchored to the *first* frame of
        # the run (see _bump_status_for_frame). None until that frame.
        self._lag_origin: float | None = None

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

    @property
    def current_handle(self) -> RunHandle | None:
        """Handle of the current or most recent run, None before the first.

        Lets a status widget attached mid-run bind to the run in progress.
        """
        return self._current_handle

    # pipeline and writer are fixed when the Controller is created. Without
    # these properties, ``ctrl.pipeline = p`` would silently create a new
    # attribute that nothing reads, and every run would store nothing.

    @property
    def pipeline(self) -> ImageProcessingPipeline | None:
        """The pipeline every acquisition of this Controller runs (read-only)."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value) -> None:
        raise AttributeError(
            "Controller.pipeline cannot be changed. Create a new "
            "Controller(mic, pipeline, writer=...) instead."
        )

    @property
    def writer(self) -> Writer | None:
        """The storage backend frames are written through (read-only)."""
        return self._writer

    @writer.setter
    def writer(self, value) -> None:
        raise AttributeError(
            "Controller.writer cannot be changed. Create a new "
            "Controller(mic, pipeline, writer=...) instead."
        )

    @property
    def pending_events(self) -> list | None:
        """Events staged via :meth:`load_experiment` but not yet started."""
        return self._pending_run[0] if self._pending_run is not None else None

    def load_experiment(self, events, *, stim_mode="current", validate=True) -> list:
        """Validate and stage an experiment without starting it.

        Validates and sorts the events exactly like :meth:`run_experiment`,
        stores the plan, and emits the ``experimentLoaded`` signal so
        status widgets can preview it (event strip, FOV positions,
        planned duration). Start it with :meth:`start_experiment` or the
        widget's Start button.

        Loading replaces a previously staged experiment, and a direct
        ``run_experiment`` / ``continue_experiment`` call discards it.
        Continuations cannot be staged: their events must be offset
        against the finished prior run, so call ``continue_experiment``
        directly.

        Returns the sorted events list.

        Raises:
            RuntimeError: If a run is still in progress, or if the run would
                store nothing (no writer and no pipeline).
            ValueError: If ``validate=True`` and events fail validation.
        """
        self._require_no_active_run()
        self._require_storage()
        events = list(events)
        if validate:
            if not self.validate_events(events):
                raise ValueError(
                    "Event validation failed (see warnings above). "
                    "Fix the issues or pass validate=False to skip."
                )
        events = sorted(
            events, key=lambda e: (e.min_start_time or 0, e.index.get("p", 0))
        )
        self._pending_run = (events, stim_mode)
        self.experimentLoaded.emit(events)
        return events

    def start_experiment(self) -> RunHandle:
        """Start the experiment staged by :meth:`load_experiment`.

        Validation already ran at load time, so it is skipped here.

        Raises:
            RuntimeError: If no experiment is loaded, or if the staged
                experiment was already started (e.g. via the status
                widget's Start button) and is still running.
        """
        if self._pending_run is None:
            if self._current_handle is not None and self._current_handle.is_running():
                raise RuntimeError(
                    "Experiment already started and still running. Use "
                    "controller.current_handle to reach it."
                )
            raise RuntimeError(
                "No experiment loaded. Call load_experiment(events) first."
            )
        events, stim_mode = self._pending_run
        self._pending_run = None
        return self.run_experiment(events, stim_mode=stim_mode, validate=False)

    def run_experiment(
        self, events, *, stim_mode="current", validate=True
    ) -> RunHandle:
        """Start an acquisition asynchronously. Returns immediately.

        The MDA runs supervised by a worker thread; ``RunHandle`` exposes
        ``wait()`` / ``cancel()`` / ``status()`` and a ``statusChanged``
        signal for live observation. There is **no** synchronous fallback —
        callers that previously did ``ctrl.run_experiment(events)`` must
        now do ``ctrl.run_experiment(events).wait()`` if they want the old
        blocking semantics. ``handle.wait()`` re-raises any worker-side
        ``fatal_error`` so the prior raise-on-failure behaviour is
        preserved when callers explicitly opt in.

        Args:
            events: Iterable of RTMEvent. Materialised to a list.
            stim_mode: How stim masks are resolved (``"current"`` /
                ``"previous"``). See previous docstring for the gritty
                semantics.
            validate: Run :meth:`validate_events` before starting. Validation
                still happens *synchronously* before the worker spawns so
                bad event lists surface as exceptions on the calling thread.

        Raises:
            RuntimeError: If a previous run is still in progress (call
                ``handle.wait()`` or ``handle.cancel()`` first), or if the run
                would store nothing (no writer and no pipeline).
            ValueError: If ``validate=True`` and events fail validation.
        """
        self._require_no_active_run()
        self._require_storage()
        self._pending_run = None  # a direct run supersedes any staged load
        events = list(events)
        if validate:
            if not self.validate_events(events):
                raise ValueError(
                    "Event validation failed (see warnings above). "
                    "Fix the issues or pass validate=False to skip."
                )

        # Sort once here so the order the worker processes them matches
        # the order the widget displays. _run_mda_with_events also sorts
        # (idempotent on an already-sorted list); doing it here lets us
        # stash the canonical sequence on the handle for status widgets.
        events = sorted(
            events, key=lambda e: (e.min_start_time or 0, e.index.get("p", 0))
        )

        handle = RunHandle(
            n_events_total=len(events),
            events=events,
            on_cancel=self._cancel_stim_waits,
        )
        self._current_handle = handle

        handle._thread = threading.Thread(
            target=self._run_worker,
            args=(events, stim_mode, handle),
            kwargs={"is_continue": False},
            name="FaroRunWorker",
            daemon=True,
        )
        handle._thread.start()
        self.runStarted.emit(handle)
        return handle

    def continue_experiment(
        self, events, *, stim_mode="current", validate=True
    ) -> RunHandle:
        """Continue acquisition with new events, preserving Analyzer state.

        Same async semantics as :meth:`run_experiment`. Reuses the existing
        ``Analyzer`` and per-FOV state so tracking and timestep counters
        continue seamlessly across runs.

        Raises:
            RuntimeError: If no previous experiment exists to continue, or
                if a run is still in progress.
            ValueError: If ``validate=True`` and events fail validation.
        """
        self._require_no_active_run()
        self._pending_run = None  # a direct run supersedes any staged load
        if self._analyzer is None:
            raise RuntimeError(
                "No experiment to continue. Call run_experiment() first."
            )
        if self._analyzer.stim_mode != stim_mode:
            raise RuntimeError(
                f"Cannot continue experiment with stim_mode={stim_mode!r}; the "
                f"running experiment is in {self._analyzer.stim_mode!r} mode. "
                "Call finish_experiment() first to start a new experiment with a "
                "different mode."
            )

        events = list(events)
        if validate:
            if not self.validate_events(events):
                raise ValueError(
                    "Event validation failed (see warnings above). "
                    "Fix the issues or pass validate=False to skip."
                )

        offset_events = self._offset_events(events)
        offset_events = sorted(
            offset_events,
            key=lambda e: (e.min_start_time or 0, e.index.get("p", 0)),
        )
        self._grow_writer_timeline(offset_events)

        handle = RunHandle(
            n_events_total=len(offset_events),
            events=offset_events,
            on_cancel=self._cancel_stim_waits,
        )
        self._current_handle = handle

        handle._thread = threading.Thread(
            target=self._run_worker,
            args=(offset_events, stim_mode, handle),
            kwargs={"is_continue": True},
            name="FaroRunWorker",
            daemon=True,
        )
        handle._thread.start()
        self.runStarted.emit(handle)
        return handle

    def _require_no_active_run(self) -> None:
        """Raise if a run is still in progress."""
        if self._current_handle is not None and self._current_handle.is_running():
            raise RuntimeError(
                "An experiment is already running. Call handle.wait() or "
                "handle.cancel() first."
            )

    def _require_storage(self) -> None:
        """Raise if a run would store nothing.

        Without a writer, the Analyzer writes TIFFs to the pipeline's
        storage_path. Without both, every frame is acquired and dropped.
        """
        if self._writer is None and self._pipeline is None:
            raise RuntimeError(
                "This Controller has no writer and no pipeline, so the run would "
                "store nothing. Create it with a pipeline, a writer, or both: "
                "Controller(mic, pipeline, writer=...)."
            )

    def _cancel_stim_waits(self) -> None:
        """RunHandle ``on_cancel`` hook: wake the event stream's mask wait.

        The event stream checks ``handle.cancel_event`` between events,
        but cannot poll while it is blocked on a stim mask inside
        ``Analyzer.get_stim_mask``. Cancelling the dispensers releases
        that wait immediately, so a cancel takes effect at once instead
        of after the stim-mask timeout.
        """
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.cancel_pending_waits()

    def _get_image_size(self) -> tuple[int, int]:
        """Return (height, width) of the microscope's camera frames.

        Prefers ``self._mic.image_height`` / ``image_width`` (the
        ``AbstractMicroscope``-level convention; ``Moench.init_scope`` and
        peers populate them on the microscope instance). Falls back to a
        pymmcore-plus core call when the microscope exposes one. Raises if
        neither is available.
        """
        h = getattr(self._mic, "image_height", None)
        w = getattr(self._mic, "image_width", None)
        if h is not None and w is not None:
            return h, w
        mmc = getattr(self._mic, "mmc", None)
        if mmc is not None:
            return mmc.getImageHeight(), mmc.getImageWidth()
        raise RuntimeError(
            "Microscope does not expose image dimensions. Set "
            "self.image_height / self.image_width on the microscope, or "
            "provide a CMMCorePlus instance via self.mmc."
        )

    def _run_worker(
        self, events, stim_mode: str, handle: RunHandle, /, *, is_continue: bool
    ) -> None:
        """Background-thread entry point for an experiment run.

        Owns: writer init (incl. potentially-slow zarr ``rmtree`` on overwrite),
        ``Analyzer`` construction (or reuse for continue), MDA supervision,
        and the final wall-clock offset update. All status updates flow
        through ``handle.update`` so listeners see the progression.
        """
        # started_at is set on the first acquired frame (see _on_frame_ready)
        # so a pre-acquisition WaitEvent doesn't tick elapsed/lag.
        handle.update(state="running")
        # Fresh RTMEvent-boundary + lag-origin trackers for this run.
        self._rtm_key = None
        self._lag_origin = None
        try:
            # ---- pre-loop setup -----------------------------------------
            if self._experiment_start is None:
                self._experiment_start = time.monotonic()

            if events:
                self._t_offset = max(e.index.get("t", 0) for e in events) + 1

            if is_continue:
                self._all_events.extend(events)
            else:
                self._all_events = list(events)

            if self._writer is not None:
                self._writer.save_events(self._all_events)

            if (
                not is_continue
                and isinstance(self._writer, OmeZarrWriter)
                and self._writer._stream is None
                and self._writer._raw_array is None
            ):
                # NB: on a network drive an overwrite-existing init can do a
                # multi-minute rmtree. This runs on a worker thread so it
                # no longer freezes napari; status stays "running" until
                # the actual MDA starts.
                img_h, img_w = self._get_image_size()
                self._writer.init_stream(
                    position_names=_extract_positions_from_events(events),
                    channel_names=_extract_channel_names_from_events(events),
                    image_height=img_h,
                    image_width=img_w,
                    n_timepoints=_extract_n_timepoints_from_events(events),
                    n_stim_channels=_extract_n_stim_channels_from_events(events),
                )

            if not is_continue:
                self._analyzer = Analyzer(self._pipeline, writer=self._writer)
                self._analyzer.stim_mode = stim_mode

            self._validate_fov_positions(events)

            # ---- the MDA ------------------------------------------------
            self._run_mda_with_events(events, stim_mode=stim_mode, handle=handle)

        except BaseException as exc:
            traceback.print_exc()
            handle.update(state="error", fatal_error=exc, finished_at=time.monotonic())
        else:
            # Update wall-clock offset for continuation
            if self._experiment_start is not None:
                self._time_offset = time.monotonic() - self._experiment_start
            handle.update(state="done", finished_at=time.monotonic())

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
        self._grow_writer_timeline(offset_events)
        # Add events + sentinel; bump counter so the stream keeps going. Lock
        # because the event stream reads _pending_sentinels from the MDA
        # thread while extend_experiment runs from the caller's thread.
        with self._pending_sentinels_lock:
            self._pending_sentinels += 1
        for ev in offset_events:
            self._event_queue.put(ev)
        self._event_queue.put(None)  # sentinel for this batch

        # Update offset for future extensions
        if offset_events:
            self._t_offset = max(e.index.get("t", 0) for e in offset_events) + 1

    def finish_experiment(self, *, drain_timeout: float = 300.0):
        """Shutdown the Analyzer and reset continuation state.

        Call after all ``run_experiment`` / ``continue_experiment`` calls
        are done. ``drain_timeout`` (default 300 s) bounds how long
        ``Analyzer.shutdown`` will wait for the storage / deferred /
        pipeline queues to drain before raising ``TimeoutError``. On
        timeout no teardown happens, so the call is safe to retry with
        a larger value.

        If a run is still in progress this blocks until it finishes —
        cancel it first via ``handle.cancel()`` if you want to abort.

        Teardown (waiting on the run + draining the Analyzer queues) runs
        on a worker thread; this method pumps the Qt event loop while it
        waits, so napari stays responsive instead of freezing for the
        whole drain.
        """
        done = threading.Event()
        box: list[BaseException] = []

        def _teardown() -> None:
            try:
                handle = self._current_handle
                if handle is not None and handle.is_running():
                    try:
                        handle.wait()
                    except BaseException as run_exc:
                        # Remember a run-side failure but still tear down.
                        box.append(run_exc)
                self._current_handle = None

                if self._analyzer is not None:
                    # shutdown is the gate — only snapshot background_errors
                    # and drop the Analyzer once it succeeds. On TimeoutError
                    # the Analyzer is still alive and the caller can retry.
                    self._analyzer.shutdown(wait=True, drain_timeout=drain_timeout)
                    self.background_errors.extend(self._analyzer.background_errors)
                    self._error_counts.update(self._analyzer.error_summary())
                    self._analyzer = None
                self._t_offset = 0
                self._time_offset = 0.0
                self._experiment_start = None
                self._event_queue = None
                self._all_events.clear()
                self._fov_positions.clear()
                self._frame_buffers.clear()
            except BaseException as exc:
                box.append(exc)
            finally:
                done.set()

        threading.Thread(target=_teardown, name="FaroFinishWorker", daemon=True).start()
        while not done.wait(timeout=0.05):
            pump_qt_events()
        if box:
            raise box[0]

    def error_summary(self) -> Counter:
        """Number of background errors per (source, exception type).

        ``background_errors`` keeps only the first errors of each kind. This
        gives the full counts of all finished experiments.
        """
        return Counter(self._error_counts)

    def queue_stats(self) -> "QueueStats | None":
        """Snapshot of the Analyzer's queue depths, or None when idle.

        Returns None between experiments (no Analyzer). Status widgets
        poll this to show storage / pipeline / deferred backpressure.
        """
        analyzer = self._analyzer
        return analyzer.queue_stats() if analyzer is not None else None

    def stop_run(self):
        """Hard-stop the run path (legacy). Prefer ``handle.cancel()``."""
        if self._current_handle is not None:
            self._current_handle.cancel()
        self._mic.cancel_mda()
        if self._analyzer is not None:
            self._analyzer.shutdown(wait=True)
        self._mic.disconnect_frame(self._on_frame_ready)
        self._frame_buffers.clear()

    # ------------------------------------------------------------------
    # Internal helpers for continuation
    # ------------------------------------------------------------------

    def _grow_writer_timeline(self, offset_events) -> None:
        """Pre-size the writer's time axis for appended events.

        One resize up front instead of one per frame past the first run's
        declared length (see ``OmeZarrWriter.set_n_timepoints``). Writers
        without that method, and empty batches, are ignored.
        """
        grow = getattr(getattr(self, "_writer", None), "set_n_timepoints", None)
        if grow is None or not offset_events:
            return
        grow(max(e.index.get("t", 0) for e in offset_events) + 1)

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

    def _run_mda_with_events(self, events, *, stim_mode, handle: RunHandle):
        """Start the MDA on its own thread and supervise it until it finishes.

        Called from :meth:`_run_worker`. The events themselves are produced
        by :meth:`_event_stream`, a generator the MDA runner pulls from on
        the MDA thread; this worker thread only sets up the run, watches
        for cancellation, and tears down.
        """
        # Live mode (continuous sequence acquisition) and the MDA both drive
        # the camera. A live stream still running when the MDA's first
        # snapImage fires consumes the snap buffer (napari-micromanager's
        # _core_link._image_snapped) before the engine calls getImage, and the
        # engine raises "Camera image buffer read failed".
        #
        # This is not gated on isSequenceRunning(): napari-micromanager can
        # hold a live timer while the stream is already stopped, and that timer
        # restarts live acquisition on the configSet/exposureChanged signals
        # the engine emits every frame. stopSequenceAcquisition() emits
        # sequenceAcquisitionStopped, whose listener clears the timer, so it
        # has to run even when no stream is currently active.
        mmc = getattr(self._mic, "mmc", None)
        if mmc is not None:
            with contextlib.suppress(Exception):
                mmc.stopSequenceAcquisition()

        self._n_stim_fallbacks = 0
        self._active_handle = handle

        # Mirror this run's log records into <storage_path>/log.txt so
        # warnings survive the console. Attached to both the "faro" logger
        # (controller + microscope warnings) and the "pymmcore-plus" logger
        # (the MDA runner's per-event trace). Records still reach the
        # console through the package stderr handler (see faro/__init__.py).
        # delay=True defers file creation to the first record; a failed
        # attach must never block a run. The handler is removed when the
        # MDA finishes, so nothing writes to the experiment folder after
        # the run; only the "faro" level is touched, and its previous
        # value is restored on detach.
        log_handler = None
        prev_faro_level = logging.NOTSET
        storage_path = getattr(self._pipeline, "storage_path", None)
        if storage_path is not None:
            try:
                log_handler = logging.FileHandler(
                    os.path.join(storage_path, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                    delay=True,
                )
                log_handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
                )
                # The pymmcore-plus logger dispatches DEBUG records; keep
                # log.txt at INFO (runner per-event trace and up).
                log_handler.setLevel(logging.INFO)
                faro_logger = logging.getLogger("faro")
                prev_faro_level = faro_logger.level
                faro_logger.setLevel(logging.INFO)
                faro_logger.addHandler(log_handler)
                logging.getLogger("pymmcore-plus").addHandler(log_handler)
            except Exception:
                log_handler = None
                logger.exception("Could not open log.txt in the experiment folder")

        # Set up event queue for extend_experiment support.
        # _pending_sentinels tracks how many extra batches (from
        # extend_experiment) still need to be drained.
        self._event_queue = Queue()
        with self._pending_sentinels_lock:
            self._pending_sentinels = 0
        events = sorted(
            events, key=lambda e: (e.min_start_time or 0, e.index.get("p", 0))
        )
        for ev in events:
            self._event_queue.put(ev)
        self._event_queue.put(None)  # sentinel for this initial batch

        if self._pre_loop_hook is not None:
            self._pre_loop_hook()

        logger.info("MDA run started: %d events, stim_mode=%s", len(events), stim_mode)

        def _on_runner_canceled(*_args) -> None:
            # Turn an external runner cancel (e.g. the stop button of a
            # napari-micromanager MDA widget) into a faro cancel. Without
            # this the event stream sleeps through WaitEvents and
            # stim-mask waits, and the run ends reading "done".
            # handle.cancel() wakes those waits. Faro's own cancel path
            # sets cancel_event before calling cancel_mda, so this does
            # nothing for internal cancels.
            if not handle.cancel_event.is_set():
                logger.info("MDA runner cancelled externally; cancelling run")
                handle.externally_cancelled = True
                handle.cancel()

        # Connect inside the try, so that the finally also disconnects when
        # run_mda raises (e.g. another MDA is still running). psygnal allows
        # duplicate connections: a callback left connected would get every
        # frame twice in the next run.
        mda_thread = None
        try:
            self._mic.connect_frame(self._on_frame_ready)
            self._mic.connect_sequence_canceled(_on_runner_canceled)
            mda_thread = self._mic.run_mda(
                self._event_stream(stim_mode=stim_mode, handle=handle)
            )

            # Supervision only: the generator runs on the MDA thread. A
            # cancel is forwarded to the runner once so it interrupts a
            # min_start_time sleep the generator cannot reach; the
            # generator's own mask waits are woken by the RunHandle's
            # on_cancel hook (_cancel_stim_waits).
            cancelled = False
            while mda_thread.is_alive():
                if handle.cancel_event.is_set() and not cancelled:
                    cancelled = True
                    with contextlib.suppress(Exception):
                        self._mic.cancel_mda()
                mda_thread.join(timeout=0.2)
        finally:
            self._event_queue = None
            if mda_thread is not None:
                mda_thread.join()
            with contextlib.suppress(Exception):
                self._mic.disconnect_sequence_canceled(_on_runner_canceled)
            # Suppressed: connect_frame raises before it connects on the Qt
            # signal backend, and a Qt disconnect raises for a callback that
            # is not connected. That error would replace the clear one.
            with contextlib.suppress(Exception):
                self._mic.disconnect_frame(self._on_frame_ready)
            # A timepoint that misses a frame (cancel, failed snap) stays in
            # the buffer. It must not merge with the frames of a later run at
            # the same (t, p).
            self._frame_buffers.clear()
            s = handle.status()
            logger.info(
                "MDA run finished: %d/%d events acquired, %d stim fallbacks",
                s.n_events_acquired,
                s.n_events_total,
                s.n_stim_fallbacks,
            )
            if log_handler is not None:
                faro_logger = logging.getLogger("faro")
                faro_logger.removeHandler(log_handler)
                faro_logger.setLevel(prev_faro_level)
                logging.getLogger("pymmcore-plus").removeHandler(log_handler)
                log_handler.close()
            self._active_handle = None

        # A _fatal_error set from a frame callback or the event stream
        # surfaces through the handle's RunStatus.fatal_error, which
        # _run_worker reads after we return. Re-raise so the worker's
        # try/except can record it.
        if self._fatal_error is not None:
            fatal = self._fatal_error
            self._fatal_error = None
            raise fatal

    def _event_stream(self, *, stim_mode: str, handle: RunHandle) -> Iterator[MDAEvent]:
        """Yield MDAEvents to the MDA runner, one pull at a time.

        The runner asks for the next event only after the previous one
        fully completed, including the synchronous ``frameReady`` handling
        that submits the frame to the pipeline on this same thread. A stim
        event's source frame is therefore already in the pipeline when its
        mask is requested. This ordering holds by construction; no clock
        is involved.

        The construction relies on ``MDARunner._run`` bypassing the
        engine's ``event_iterator`` when handed an Iterator directly (as
        this generator is), so hardware sequencing never batches these
        events even if ``use_hardware_sequencing`` is set. Sequencing
        would break the invariant twice: ``iter_sequenced_events`` pulls
        ahead to find batch boundaries, and a SequencedEvent's frames
        arrive only after the whole batch was pulled. If sequencing is
        ever wanted, stim builds need an explicit frame-acquired gate
        instead of pull order.

        Pause takes effect between RTMEvents only: the stream holds
        without yielding while paused, and min_start_times keep their
        values on resume, so late events catch up. The body records any
        exception as the run's fatal error and ends the stream cleanly,
        because an exception raised here would otherwise kill the MDA
        thread with a bare traceback.
        """
        try:
            while True:
                if handle.cancel_event.is_set():
                    return

                if handle.pause_event.is_set():
                    handle.update(state="paused")
                    while handle.pause_event.is_set():
                        if handle.cancel_event.is_set():
                            return
                        time.sleep(0.05)
                    if handle.cancel_event.is_set():
                        return
                    handle.update(state="running")
                    continue

                # Short timeout so cancellation is responsive even when
                # the queue is empty (waiting for extend_experiment).
                try:
                    rtm_event = self._event_queue.get(timeout=0.1)
                except QueueEmpty:
                    continue

                if rtm_event is None:
                    # Sentinel consumed. Stop only if no extension pending.
                    with self._pending_sentinels_lock:
                        if self._pending_sentinels > 0:
                            self._pending_sentinels -= 1
                            continue
                    return

                if isinstance(rtm_event, WaitEvent):
                    if not self._run_wait_event(rtm_event, handle):
                        return
                    continue

                # Status update: the stream committed to this RTMEvent.
                prev = handle.status()
                fov = rtm_event.index.get("p")
                handle.update(
                    current_event_index=dict(rtm_event.index),
                    current_fov=fov,
                    n_events_consumed=prev.n_events_consumed + 1,
                )

                self._n_channels = len(rtm_event.channels)

                # In "previous" mode at t=0 there is no predecessor
                # mask, so suppress the stim event entirely. Firing a
                # blank mask would still activate the DMD (mirror
                # bleed-through ~1% of nominal intensity), and omitting
                # ``slm_image`` would leave the DMD in its previously-
                # latched state. Per-FOV first-visit suppression is
                # *not* needed: the pipeline always-computes in previous
                # mode, so the predecessor's mask exists for every t > 0.
                suppress_stim = (
                    stim_mode == "previous" and rtm_event.index.get("t", 0) == 0
                )

                planned = rtm_event.plan_events(
                    stim_mode=stim_mode,
                    build_slm=None,
                    resolve_group=self._mic.resolve_group,
                    resolve_power=self._mic.resolve_power,
                    suppress_stim=suppress_stim,
                )
                slm = None
                for ev in planned:
                    if ev.metadata.get("img_type") == ImgType.IMG_STIM:
                        if slm is None and self._mic.dmd:
                            slm = self._build_stim_slm(rtm_event, stim_mode=stim_mode)
                            if handle.cancel_event.is_set():
                                return
                        if slm is not None:
                            ev = ev.model_copy(update={"slm_image": slm})
                    yield ev
        except Exception as exc:
            # Catch Exception, not BaseException: GeneratorExit must
            # propagate so an abandoned generator closes normally.
            if self._fatal_error is None:
                self._fatal_error = exc
            logger.exception("Event stream failed; ending the MDA")
            return

    def _run_wait_event(self, rtm_event: WaitEvent, handle: RunHandle) -> bool:
        """Hold the event stream for the wait's duration; emits no MDAEvents.

        Returns False when the run was cancelled during the wait. The
        stream is strictly serial, so all prior events have already
        completed when this runs.
        """
        prev = handle.status()
        handle.update(
            current_event_index=dict(rtm_event.index),
            n_events_consumed=prev.n_events_consumed + 1,
        )
        base = self._experiment_start or time.monotonic()
        # Anchor to whichever is later: scheduled start or now. A run
        # behind schedule would otherwise flash through the wait.
        scheduled_start = base + (rtm_event.min_start_time or 0)
        deadline = max(scheduled_start, time.monotonic()) + rtm_event.duration_s
        handle.update(state="waiting", wait_remaining_s=rtm_event.duration_s)
        last_displayed = int(rtm_event.duration_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Throttle emissions to once per displayed second: the widget
            # renders int(remaining), so sub-second updates only churn
            # the Qt event queue.
            cur = int(remaining)
            if cur != last_displayed:
                handle.update(wait_remaining_s=remaining)
                last_displayed = cur
            if handle.cancel_event.wait(min(0.1, remaining)):
                break
        if handle.cancel_event.is_set():
            return False
        # Bump n_events_acquired so the cursor stays monotonic when state
        # leaves "waiting".
        new_state = "pausing" if handle.pause_event.is_set() else "running"
        handle.update(
            state=new_state,
            wait_remaining_s=None,
            n_events_acquired=handle.status().n_events_acquired + 1,
        )
        return True

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------

    def _bump_status_for_frame(self, event: MDAEvent) -> None:
        """Update RunHandle counters for the current frame; no-op if no handle.

        An RTMEvent expands into several MDAEvents (one per imaging/ref
        channel, plus stim). This handler fires per MDAEvent, so it does
        two different things:

        * ``n_frames_received`` -- bumped on every imaging/ref frame.
        * ``n_events_acquired`` + ``lag_ms`` -- updated only on the
          *first* frame of each RTMEvent, detected by the (p, t) index
          changing. So the widget's progress and the lag readout move
          once per RTMEvent, not once per channel-frame.

        Stim emissions are skipped entirely: a stim frame is the
        SLM-illuminated snap firing alongside its imaging frame, not a
        data frame.
        """
        handle = self._current_handle
        if handle is None:
            return
        img_type = (event.metadata or {}).get("img_type", ImgType.IMG_RAW)
        if img_type == ImgType.IMG_STIM:
            return
        prev = handle.status()
        wallclock = time.time()

        # RTMEvent boundary: (p, t) identifies one logical timepoint+FOV.
        key = (event.index.get("p"), event.index.get("t"))
        is_new_rtm_event = key != self._rtm_key

        updates: dict = {
            "n_frames_received": prev.n_frames_received + 1,
            "last_frame_wallclock": wallclock,
        }
        if is_new_rtm_event:
            self._rtm_key = key
            updates["n_events_acquired"] = prev.n_events_acquired + 1
            if prev.started_at is None:
                # Anchor elapsed clock to first acquisition (matches _lag_origin).
                exposure_s_for_start = (
                    getattr(event, "exposure", None) or 0.0
                ) / 1000.0
                updates["started_at"] = time.monotonic() - exposure_s_for_start
            # Lag = how far this RTMEvent's acquisition start drifted from
            # its scheduled min_start_time. frameReady fires when the frame
            # *finished*, so back out the exposure to estimate the start.
            #
            # The reference clock is anchored to the *first* frame of the
            # run, NOT to started_at: min_start_time is relative to when
            # the engine began acquiring, while started_at is stamped
            # before writer init + Analyzer construction + engine/hardware
            # startup (~1 s). Charging that constant startup to every lag
            # reading made an on-schedule run look ~1 s behind. Anchoring
            # to the first frame cancels it -- frame 0 reads ~0, later
            # frames show genuine drift. Engine-agnostic: it needs only
            # the frame callback firing plus useq's min_start_time /
            # exposure, no engine-specific timing metadata.
            min_start = getattr(event, "min_start_time", None)
            if min_start is not None:
                exposure_s = (getattr(event, "exposure", None) or 0.0) / 1000.0
                acq_start = time.monotonic() - exposure_s
                if self._lag_origin is None:
                    # First frame defines t0: its acquisition start
                    # corresponds to this event's scheduled min_start_time.
                    self._lag_origin = acq_start - min_start
                updates["lag_ms"] = (acq_start - self._lag_origin - min_start) * 1000.0
        handle.update(**updates)

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        # Drop subsequent frames after a fatal error — the MDA is winding down.
        if self._fatal_error is not None:
            return

        self._bump_status_for_frame(event)

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

        # Imaging + ref: buffer by (t, p), submit when all channels received.
        # Ref channels (present only on user-defined timepoints) are buffered
        # alongside imaging channels so the pipeline runs exactly once per
        # timepoint.  The expected count is derived from event metadata so it
        # is correct even when ref_channels vary across timepoints.
        tp = (event.index.get("t", 0), event.index.get("p", 0))
        buf = self._frame_buffers.setdefault(tp, [])
        if buf and buf[0].shape[-2:] != img.shape[-2:]:
            # Channels at the same (t, p) came back at different sizes —
            # almost always a sticky camera binning or ROI property.
            self._abort_mda_from_callback(
                f"Frame shape mismatch: channel {tuple(img.shape[-2:])} vs "
                f"previous {tuple(buf[0].shape[-2:])} at index={dict(event.index)}. "
                "Sticky camera Binning/ROI?"
            )
            return
        buf.append(img)

        n_expected = len(event.metadata.get("channels", ()))
        n_expected += len(event.metadata.get("ref_channels", ()))

        if len(buf) >= n_expected:
            frame = np.stack(buf, axis=0)
            del self._frame_buffers[tp]
            self._analyzer.run(frame, event)

    def _abort_mda_from_callback(self, message: str) -> None:
        """Stash a fatal error and cancel the MDA from a psygnal callback.

        Raising directly from frameReady would be swallowed by psygnal's
        callback error handler, so we store the exception and cancel —
        ``_run_mda_with_events`` re-raises after the MDA thread joins.
        """
        self._fatal_error = RuntimeError(message)
        print(f"[Controller] FATAL: {message}")
        try:
            self._mic.cancel_mda()
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Stim helpers
    # ------------------------------------------------------------------

    def _build_stim_slm(
        self, rtm_event, *, stim_mode: str = "current"
    ) -> SLMImage | None:
        """Build the SLMImage for a stim event via the Analyzer's stim-mask API.

        Runs on the MDA thread between the source frame's completion and
        the stim event's dispatch. The source frame is ``(t-1, p)`` in
        ``"previous"`` mode and ``(t, p)`` in ``"current"`` mode, and is
        already in the pipeline when this runs, so the wait below covers
        segmentation latency only. The mask request waits until
        ``stim_mask_timeout`` seconds past the event's scheduled due time,
        then fires the all-off fallback, warns, and bumps
        ``RunStatus.n_stim_fallbacks``. A mask that arrives past the due
        time still fires, with a warning. The timeout only tunes
        patience; it cannot cause a wrong mask.

        Returns None only when the run was cancelled during the wait.
        """
        fov_index = rtm_event.index.get("p", 0)
        stim_ch = rtm_event.stim_channels[0]

        t = rtm_event.index.get("t", 0)
        t_src = t
        if stim_mode == "previous":
            t_src -= 1
            # Previous-mode t=0 has no predecessor mask. The controller
            # passes ``suppress_stim=True`` to ``plan_events`` for that
            # case, so no stim event should reach this method with
            # ``t_src < 0``.
            assert t_src >= 0, "previous-mode t=0 stim event reached _build_stim_slm"
        meta = {
            **rtm_event.metadata,
            "fov": fov_index,
            "timestep": t_src,
        }

        analyzer = self._analyzer
        t0 = time.perf_counter()
        late_s = 0.0
        if analyzer is None:
            stim_mask = None
        elif not analyzer.stimulator_needs_data:
            # Metadata-only stimulator (or none): resolves synchronously.
            stim_mask = analyzer.get_stim_mask(fov_index, meta)
            logger.debug(
                "Stim mask for t=%s p=%s built in %.3f s",
                t,
                fov_index,
                time.perf_counter() - t0,
            )
        else:
            # Data-dependent: wait on the pipeline's mask queue until
            # ``stim_mask_timeout`` seconds past the event's due time.
            now = time.monotonic()
            origin = self._lag_origin
            if origin is None:
                origin = self._experiment_start
            if origin is None:
                origin = now
            if rtm_event.min_start_time is None:
                due = now
            else:
                due = origin + rtm_event.min_start_time
            slip = analyzer._stim_mask_timeout
            stim_mask = analyzer.get_stim_mask(
                fov_index, meta, timeout=max(0.0, due + slip - now)
            )
            late_s = time.monotonic() - due

        handle = self._active_handle
        if stim_mask is None and handle is not None and handle.cancel_event.is_set():
            return None
        if stim_mask is None:
            waited = time.perf_counter() - t0
            self._n_stim_fallbacks += 1
            msg = (
                f"Stim mask for t={t} p={fov_index} (source frame t={t_src}) "
                f"unavailable after waiting {waited:.1f} s; firing ALL-OFF "
                "fallback. See background_errors for the pipeline-side reason."
            )
            logger.warning(msg)
            print(f"[WARN] {msg}")
            if handle is not None:
                handle.update(n_stim_fallbacks=self._n_stim_fallbacks)
            stim_mask = False
        else:
            if late_s > 0.05:
                msg = (
                    f"Stim mask for t={t} p={fov_index} delivered "
                    f"{late_s:.2f} s past the event's scheduled time; "
                    "stim fires late."
                )
                logger.warning(msg)
                print(f"[WARN] {msg}")
            if isinstance(stim_mask, np.ndarray):
                stim_mask = self._mic.dmd.affine_transform(stim_mask)

        return SLMImage(
            data=stim_mask, device=self._mic.dmd.name, exposure=stim_ch.exposure
        )


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

        # Detect OME-Zarr source. Raw read-back goes through the same
        # OmeZarrRawReader the live writer uses, so the store-layout knowledge
        # (direct / single-position / plate) lives in exactly one place.
        zarr_path = os.path.join(old_data_project_path, "acquisition.ome.zarr")
        if os.path.isdir(zarr_path):
            self._zarr_reader: OmeZarrRawReader | None = OmeZarrRawReader(zarr_path)
        else:
            self._zarr_reader = None

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        """Override to load images from disk for simulated controller."""
        self._bump_status_for_frame(event)
        meta = event.metadata or {}
        img_type = meta.get("img_type", ImgType.IMG_RAW)

        if img_type == ImgType.IMG_STIM:
            return  # Stim images are not processed in this simulation

        # Buffer by (t, p), submit when all channels received
        tp = (event.index.get("t", 0), event.index.get("p", 0))
        buf = self._frame_buffers.setdefault(tp, [])
        buf.append(img)

        n_expected = len(meta.get("channels", ()))
        n_expected += len(meta.get("ref_channels", ()))

        if len(buf) >= n_expected:
            del self._frame_buffers[tp]
            fname = meta["fname"]
            t_idx = event.index.get("t", 0)
            p_idx = event.index.get("p", 0)

            if img_type == ImgType.IMG_RAW and self._zarr_reader is not None:
                # Strip stim-readout channels (if any) so re-analysis sees the
                # same imaging-only frame the live pipeline received.
                img_loaded = self._zarr_reader.read(
                    t_idx, p_idx, n_imaging_channels=len(meta.get("channels", ()))
                )
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
