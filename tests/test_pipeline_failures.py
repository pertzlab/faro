"""Failure-path and stress-test pipeline integration tests.

Covers:

* ``TestStressSlowSegmentation`` — 50-frame acquisition with slow
  segmentation exercises the deferred-queue backpressure path.
* ``TestCrashingSegmentator`` / ``Tracker`` / ``Stimulator`` /
  ``FeatureExtractor`` — each pipeline stage's fatal error path must
  still save raw images and leave downstream workers unblocked.
* ``TestBurstNoSignalLoss`` — 100-frame burst against a near-instant
  tiny pipeline; no frames should be dropped under back-pressure.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import pytest
import tifffile

from faro.core.controller import Controller
from faro.core.data_structures import SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.core.writers import OmeZarrWriter
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import OtsuSegmentator
from faro.stimulation.center_circle import CenterCircle
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import (
    CircleScene,
    CrashingStimulator,
    make_events,
    run_and_wait,
    run_and_wait_long,
    tracker,  # noqa: F401 — parametrized fixture, auto-discovered by pytest
)


class SlowSegmentator(OtsuSegmentator):
    """OtsuSegmentator that sleeps *delay* seconds per frame to simulate load."""

    def __init__(self, delay: float = 1.0):
        super().__init__()
        self._delay = delay

    def segment(self, image: np.ndarray) -> np.ndarray:
        time.sleep(self._delay)
        return super().segment(image)


# ===================================================================
# Test Class 9: Stress test — slow segmentation, rapid acquisition
# ===================================================================


N_STRESS_FRAMES = 50
STRESS_SEG_DELAY = 0.1  # seconds per segmentation


class TestStressSlowSegmentation:
    """Acquire 20 frames with no delay while segmentation takes 1s each.

    Verifies that the deferred-queue mechanism in Analyzer eventually
    processes every frame: all raw images stored, all segmentation masks
    produced, all stim masks produced, and tracking covers every frame.
    """

    STIM_FRAMES = tuple(range(5, 15))  # stim on frames 5-14

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = ImageProcessingPipeline(
            storage_path=self.path,
            segmentators=[
                SegmentationMethod(
                    "labels", SlowSegmentator(STRESS_SEG_DELAY), 0, False
                )
            ],
            tracker=tracker,
            feature_extractor=SimpleFE("labels"),
            stimulator=CenterCircle(),
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_STRESS_FRAMES, stim_frames=self.STIM_FRAMES)
        run_and_wait_long(
            self.ctrl,
            self.events,
            stim_mode="current",
            timeout=N_STRESS_FRAMES * STRESS_SEG_DELAY + 60,
        )

    def test_all_raw_images_stored(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_STRESS_FRAMES
        ), f"Expected {N_STRESS_FRAMES} raw TIFFs, got {len(files)}"

    def test_all_segmentation_masks_produced(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_STRESS_FRAMES
        ), f"Expected {N_STRESS_FRAMES} label masks, got {len(files)}"

    def test_all_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_STRESS_FRAMES
        ), f"Expected {N_STRESS_FRAMES} stim masks, got {len(files)}"

    def test_stim_masks_nonzero_on_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert mask.max() > 0, f"Frame {i}: stim frame should have nonzero mask"

    def test_tracking_covers_all_frames(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert (
                len(rows) == N_STRESS_FRAMES
            ), f"Particle {pid}: expected {N_STRESS_FRAMES} rows, got {len(rows)}"

    def test_segmentation_quality_maintained(self):
        """Even under load, every frame should segment into exactly 2 labels."""
        labels_dir = os.path.join(self.path, "labels")
        files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".tiff")])
        for f in files:
            labels = tifffile.imread(os.path.join(labels_dir, f))
            unique = set(np.unique(labels)) - {0}
            assert len(unique) == 2, f"{f}: expected 2 labels, got {len(unique)}"


class _RecordingPipeline:
    """Minimal pipeline stub that records how ``run`` was invoked."""

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.segmentators = ["seg"]  # non-None so the pipeline runs
        self.stimulator = None
        self._analyzer = None
        self._writer = None
        self.calls = []
        self._lock = threading.Lock()
        self.seen = threading.Event()

    def run(self, img=None, event=None, file_path=None):
        with self._lock:
            self.calls.append({"img": img, "file_path": file_path})
        self.seen.set()
        return {}


class TestDeferredReloadsViaWriter:
    """Regression: deferred frames reload through the writer backend, never via
    a hardcoded ``raw/<fname>.tiff``.

    The old deferred path built ``storage_path/raw/<fname>.tiff`` unconditionally
    and let ``pipeline.run`` imread it. With the OME-Zarr writer that TIFF never
    exists (raw frames live in the zarr store), so every deferred frame died
    with ``FileNotFoundError`` — and because such a frame never resolves its
    tracks dispenser, downstream frames then hung with "FrameDispenser: timeout
    waiting for frame N". The fix routes the reload through ``writer.read_raw``,
    which each backend implements against its own store.
    """

    def test_deferred_frame_reloads_via_writer_read_raw(self, tmp_dir):
        from types import SimpleNamespace

        from faro.core.controller import Analyzer

        sentinel = np.full((1, 8, 8), 7, dtype=np.uint16)

        class _FakeZarrWriter:
            """Stand-in for a non-TIFF backend: no ``raw/*.tiff`` on disk."""

            storage_path = tmp_dir

            def __init__(self):
                self.read_calls = []

            def write(self, img, metadata, folder):
                pass

            def read_raw(self, metadata):
                self.read_calls.append(metadata["fname"])
                return sentinel

            def save_events(self, events):
                pass

            def close(self):
                pass

        writer = _FakeZarrWriter()
        pipeline = _RecordingPipeline(tmp_dir)
        analyzer = Analyzer(pipeline, max_workers=1, max_queue_size=4, writer=writer)
        try:
            event = SimpleNamespace(index={"t": 3})
            metadata = {"fname": "003_00042", "fov": 3, "timestep": 3}

            # Enqueue exactly as _try_submit_pipeline does when overloaded.
            analyzer._deferred_queue.put((event, metadata, "raw"))

            assert pipeline.seen.wait(timeout=5), "deferred frame was never run"
            assert analyzer.wait_idle(timeout=5)

            # Reload went through the backend, not a reconstructed TIFF path.
            assert writer.read_calls == ["003_00042"]
            assert len(pipeline.calls) == 1
            call = pipeline.calls[0]
            assert call["file_path"] is None
            assert np.array_equal(call["img"], sentinel)
            assert analyzer.background_errors == []
        finally:
            analyzer.shutdown(wait=False)

    def test_unreloadable_deferred_frame_is_skipped_not_hung(self, tmp_dir):
        """If the reload fails, the frame is marked skipped so downstream
        ``get_predecessor`` waiters resolve instead of timing out."""
        from types import SimpleNamespace

        from faro.core.controller import Analyzer

        class _FailingWriter:
            storage_path = tmp_dir

            def write(self, img, metadata, folder):
                pass

            def read_raw(self, metadata):
                raise FileNotFoundError("frame not in store")

            def save_events(self, events):
                pass

            def close(self):
                pass

        pipeline = _RecordingPipeline(tmp_dir)
        analyzer = Analyzer(
            pipeline, max_workers=1, max_queue_size=4, writer=_FailingWriter()
        )
        try:
            event = SimpleNamespace(index={"t": 5})
            metadata = {"fname": "000_00005", "fov": 0, "timestep": 5}
            analyzer._deferred_queue.put((event, metadata, "raw"))

            # Frame 5 must resolve as skipped (not left unresolved) so that
            # downstream waiters don't block forever. wait_for_frame returns
            # None for a skipped frame, or raises queue.Empty on timeout.
            fov_state = analyzer.get_fov_state(0)
            assert fov_state.tracks_queue.wait_for_frame(5, timeout=5) is None
            assert any(e.source == "deferred" for e in analyzer.background_errors)
            # The unreloadable frame was never handed to the pipeline.
            assert pipeline.calls == []
        finally:
            analyzer.shutdown(wait=False)


class TestDeferredReloadFromRealOmeZarr:
    """The deferred reload pulls the *right* pixels from a real OME-Zarr store.

    Exercises the genuine defer→reload mechanism — the real deferred queue and
    the real deferred_worker calling the real ``OmeZarrWriter.read_raw`` — but
    drives the queue directly so the result is deterministic (no dependence on
    pipeline timing / tracking ordering, which is a separate concern). This is
    the exact path that crashed in production with
    ``FileNotFoundError: …/raw/<fname>.tiff`` on the OME-Zarr backend.

    Parametrized over single-position (stream layout → disk read-back) and
    multi-position (direct layout → live-array read-back).
    """

    N_T = 4

    @staticmethod
    def _frame(t: int, p: int) -> np.ndarray:
        """A (1, y, x) frame with a value unique to (t, p) for identification."""
        f = np.zeros((1, 16, 16), dtype=np.uint16)
        f[0, t, p] = 1000 + 100 * t + p
        return f

    @pytest.mark.parametrize("n_pos", [1, 2])
    def test_deferred_frame_reloads_its_own_pixels(self, n_pos, tmp_dir):
        from types import SimpleNamespace

        from faro.core.controller import Analyzer

        writer = OmeZarrWriter(tmp_dir, store_stim_images=False, n_timepoints=self.N_T)
        writer.init_stream(
            position_names=[f"Pos{p}" for p in range(n_pos)],
            channel_names=["phase-contrast"],
            image_height=16,
            image_width=16,
            n_timepoints=self.N_T,
            n_stim_channels=0,
        )

        recorded: list[tuple] = []
        lock = threading.Lock()

        class _RecordingPipeline:
            def __init__(self, storage_path):
                self.storage_path = storage_path
                self.segmentators = ["seg"]  # non-None so the pipeline runs
                self.stimulator = None
                self._analyzer = None
                self._writer = None

            def run(self, img=None, event=None, file_path=None):
                with lock:
                    recorded.append((event.index.get("t"), event.index.get("p"), img))
                return {}

        pipeline = _RecordingPipeline(tmp_dir)
        analyzer = Analyzer(pipeline, max_workers=1, max_queue_size=4, writer=writer)
        try:
            # Write each raw frame to the store exactly as the storage worker
            # would, then enqueue it on the deferred queue (pixels dropped — only
            # metadata is kept, to be reloaded from the store).
            metas = {}
            for t in range(self.N_T):
                for p in range(n_pos):
                    meta = {
                        "timestep": t,
                        "fov": p,
                        "fname": f"{p:03d}_{t:05d}",
                    }
                    metas[(t, p)] = meta
                    writer.write(self._frame(t, p), meta, "raw")
            for t in range(self.N_T):
                for p in range(n_pos):
                    event = SimpleNamespace(index={"t": t, "p": p})
                    analyzer._deferred_queue.put((event, metas[(t, p)], "raw"))

            assert analyzer.wait_idle(timeout=15), "deferred queue never drained"
        finally:
            analyzer.shutdown(wait=False)

        assert analyzer.background_errors == []
        assert len(recorded) == self.N_T * n_pos
        # Each deferred frame was reloaded from the store with ITS OWN pixels —
        # the reload is wired to the right backend and the right (t, p) index.
        for t, p, img in recorded:
            assert img is not None, f"frame t={t} p={p} reloaded as None"
            assert img.shape == (1, 16, 16)
            assert img[0, t, p] == 1000 + 100 * t + p, f"wrong pixels for t={t} p={p}"


# ===================================================================
# Failing component helpers
# ===================================================================


class CrashingSegmentator(OtsuSegmentator):
    """Segmentator that raises on every call."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError("Segmentation crashed!")


class CrashingTracker(TrackerTrackpy):
    """Tracker that raises on every call."""

    def track_cells(self, df_old, df_new, fov_state):
        raise RuntimeError("Tracking crashed!")


class CrashingFE(SimpleFE):
    """Feature extractor that raises on every call."""

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        raise RuntimeError("Feature extraction crashed!")


# ===================================================================
# Crash-test helpers
# ===================================================================
N_CRASH_FRAMES = 3
CRASH_QUEUE_TIMEOUT = 1  # seconds (instead of default 20)


def _make_crashing_pipeline(
    path, *, segmentator=None, tracker=None, fe=None, stim=None
):
    """Build a pipeline with short queue timeout for crash tests."""
    pipeline = ImageProcessingPipeline(
        storage_path=path,
        segmentators=[
            SegmentationMethod("labels", segmentator or OtsuSegmentator(), 0, False)
        ],
        tracker=tracker or TrackerTrackpy(search_range=50, memory=3),
        feature_extractor=fe or SimpleFE("labels"),
        stimulator=stim,
    )
    pipeline._queue_timeout = CRASH_QUEUE_TIMEOUT
    return pipeline


# ===================================================================
# Test Class 10: Failing segmentator — raw images must still be saved
# ===================================================================


class TestCrashingSegmentator:
    """Pipeline crashes during segmentation. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            segmentator=CrashingSegmentator(),
            tracker=tracker,
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_CRASH_FRAMES)
        run_and_wait(self.ctrl, self.events)

    def test_all_raw_images_saved(self):
        """Raw images are saved BEFORE pipeline.run(), so they must all exist."""
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == N_CRASH_FRAMES

    def test_no_segmentation_masks(self):
        """Pipeline crashed during segmentation, so no label masks should exist."""
        labels_dir = os.path.join(self.path, "labels")
        if not os.path.isdir(labels_dir):
            return  # directory never created ⇒ no masks saved
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == 0


# ===================================================================
# Test Class 11: Failing tracker — raw images saved
# ===================================================================


class TestCrashingTracker:
    """Pipeline crashes during tracking. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            tracker=CrashingTracker(search_range=50, memory=3),
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_CRASH_FRAMES)
        run_and_wait(self.ctrl, self.events)

    def test_all_raw_images_saved(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == N_CRASH_FRAMES

    def test_no_segmentation_masks_saved(self):
        """Crash happens after segmentation but before TIFF save (which is after put()).
        Since tracking crashes before put(), the TIFF saves never run."""
        labels_dir = os.path.join(self.path, "labels")
        if not os.path.isdir(labels_dir):
            return  # directory never created ⇒ no masks saved
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == 0


# ===================================================================
# Test Class 12: Failing stimulator — raw images saved
# ===================================================================


class TestCrashingStimulator:
    """Pipeline crashes during stim mask generation. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        # Stim on last frame only so no subsequent frame waits on the broken put()
        self.n_frames = N_CRASH_FRAMES
        self.stim_frames = (self.n_frames - 1,)
        self.pipeline = _make_crashing_pipeline(
            self.path,
            stim=CrashingStimulator(),
            tracker=tracker,
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(self.n_frames, stim_frames=self.stim_frames)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_all_raw_images_saved(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == self.n_frames


# ===================================================================
# Test Class 13: Failing feature extractor — raw images saved
# ===================================================================


class TestCrashingFeatureExtractor:
    """Pipeline crashes during feature extraction. Raw images should still be saved."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            fe=CrashingFE("labels"),
            tracker=tracker,
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_CRASH_FRAMES)
        run_and_wait(self.ctrl, self.events)

    def test_all_raw_images_saved(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert len(files) == N_CRASH_FRAMES

    def test_no_segmentation_masks_saved(self):
        """FE crash happens before put() and TIFF saves, so no masks are written."""
        labels_dir = os.path.join(self.path, "labels")
        if not os.path.isdir(labels_dir):
            return  # directory never created ⇒ no masks saved
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == 0


# ===================================================================
# Continuation / extension helpers
# ===================================================================


# ===================================================================
# Test Class 14: Continue experiment — sequential continuation
# ===================================================================

N_BURST_FRAMES = 100
FIRST_FRAME_DELAY = 1.0
_BURST_IMG_SIZE = 16


def _make_tiny_image() -> np.ndarray:
    """10x10 uint16 image with two bright pixels (labels 1 and 2)."""
    img = np.zeros((_BURST_IMG_SIZE, _BURST_IMG_SIZE), dtype=np.uint16)
    img[3, 3] = 50000
    img[12, 12] = 50000
    return img


class _TinyScene:
    """16×16 frames with two bright pixels, used by the burst stress test."""

    image_height = 16
    image_width = 16
    channels = ("phase-contrast",)

    def render(self, event) -> np.ndarray:
        return _make_tiny_image()


class _FastSegmentator(OtsuSegmentator):
    """Near-instant segmentator: threshold + label.

    First call is delayed to create pipeline back-pressure.
    """

    def __init__(self, first_delay: float = FIRST_FRAME_DELAY):
        super().__init__()
        self._first_delay = first_delay
        self._lock = threading.Lock()
        self._first_done = False

    def segment(self, image: np.ndarray) -> np.ndarray:
        from skimage.measure import label

        with self._lock:
            is_first = not self._first_done
            self._first_done = True
        if is_first:
            time.sleep(self._first_delay)
        return label(image > image.mean())


class _FastTracker:
    """Tracker that assigns particle IDs by label without trackpy linking."""

    required_metadata: set[str] = set()

    def track_cells(self, df_old, df_new, fov_state):
        if "particle" not in df_new.columns:
            df_new = df_new.copy()
            df_new["particle"] = df_new["label"]
        return pd.concat([df_old, df_new], ignore_index=True)


class _FastFE:
    """Feature extractor returning positions and area via regionprops."""

    def __init__(self, used_mask):
        self.used_mask = used_mask

    def extract_positions(self, labels):
        from skimage.measure import regionprops_table

        table = regionprops_table(
            labels[self.used_mask], properties=["label", "centroid"]
        )
        df = pd.DataFrame.from_dict(table)
        df = df.rename({"centroid-0": "x", "centroid-1": "y"}, axis="columns")
        return df

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        from skimage.measure import regionprops_table

        table = regionprops_table(labels[self.used_mask], properties=["label", "area"])
        return pd.DataFrame.from_dict(table), None


@pytest.fixture(scope="module")
def burst_result():
    """Run the burst experiment once and share results across all tests."""
    path = tempfile.mkdtemp()
    pipeline = ImageProcessingPipeline(
        storage_path=path,
        segmentators=[
            SegmentationMethod("labels", _FastSegmentator(), 0, False),
        ],
        tracker=_FastTracker(),
        feature_extractor=_FastFE("labels"),
        stimulator=CenterCircle(),
    )
    pipeline._queue_timeout = 120
    mic = FakeMicroscope(_TinyScene())
    ctrl = Controller(mic, pipeline)
    stim_frames = tuple(range(10, 20))
    events = make_events(N_BURST_FRAMES, stim_frames=stim_frames)
    run_and_wait_long(
        ctrl,
        events,
        stim_mode="current",
        timeout=FIRST_FRAME_DELAY + N_BURST_FRAMES * 0.1 + 60,
    )
    yield path
    shutil.rmtree(path)


class TestBurstNoSignalLoss:
    """Dispatch 100 frames rapidly while the first frame blocks the pipeline.

    Uses tiny images (16x16) and fast fakes for segmentation/tracking/FE so
    the test exercises the controller's dispatch plumbing (queues, deferred
    worker, storage) rather than real image-analysis performance.
    """

    def test_all_raw_images_stored(self, burst_result):
        """Storage path: no images silently dropped."""
        raw_dir = os.path.join(burst_result, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_BURST_FRAMES
        ), f"Expected {N_BURST_FRAMES} raw TIFFs, got {len(files)}"

    def test_all_segmentation_masks_produced(self, burst_result):
        """Pipeline path: every frame segmented despite initial block."""
        labels_dir = os.path.join(burst_result, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_BURST_FRAMES
        ), f"Expected {N_BURST_FRAMES} label masks, got {len(files)}"

    def test_all_stim_masks_produced(self, burst_result):
        """Stim path: every frame produced a stim mask."""
        stim_dir = os.path.join(burst_result, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_BURST_FRAMES
        ), f"Expected {N_BURST_FRAMES} stim masks, got {len(files)}"

    def test_tracking_covers_all_frames(self, burst_result):
        """Tracking: both particles tracked across every frame."""
        tracks_dir = os.path.join(burst_result, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert (
                len(rows) == N_BURST_FRAMES
            ), f"Particle {pid}: expected {N_BURST_FRAMES} rows, got {len(rows)}"


# ===================================================================
# Test Class: Analyzer.get_stim_mask timeout recording
# ===================================================================
