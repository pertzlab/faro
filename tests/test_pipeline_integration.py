"""End-to-end pipeline integration tests.

Uses a fake CircleMicroscope (no pymmcore dependency) that generates
synthetic images with two bright circles at known positions.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import tifffile
from useq import MDAEvent

from rtm_pymmcore.core.controller import Analyzer, Controller
from rtm_pymmcore.core.data_structures import (
    Channel,
    FovState,
    ImgType,
    RTMEvent,
    SegmentationMethod,
)
from rtm_pymmcore.core.pipeline import ImageProcessingPipeline
from rtm_pymmcore.feature_extraction.simple import SimpleFE
from rtm_pymmcore.microscope.base import AbstractMicroscope
from rtm_pymmcore.segmentation.base import OtsuSegmentator
from rtm_pymmcore.stimulation.base import Stim, StimWithImage, StimWholeFOV
from rtm_pymmcore.stimulation.center_circle import CenterCircle
from rtm_pymmcore.tracking.trackpy import TrackerTrackpy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 256
CIRCLE1_CENTER = (64, 64)  # (row, col)
CIRCLE1_RADIUS = 20
CIRCLE2_CENTER = (192, 192)
CIRCLE2_RADIUS = 15

EXPECTED_AREA_1 = math.pi * CIRCLE1_RADIUS**2  # ~1257
EXPECTED_AREA_2 = math.pi * CIRCLE2_RADIUS**2  # ~707

N_TIMEPOINTS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_circle_image() -> np.ndarray:
    """Generate a 256x256 uint16 image with two bright circles."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)
    y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]

    mask1 = (y - CIRCLE1_CENTER[0]) ** 2 + (
        x - CIRCLE1_CENTER[1]
    ) ** 2 <= CIRCLE1_RADIUS**2
    mask2 = (y - CIRCLE2_CENTER[0]) ** 2 + (
        x - CIRCLE2_CENTER[1]
    ) ** 2 <= CIRCLE2_RADIUS**2

    img[mask1] = 50000
    img[mask2] = 50000
    return img


def make_events(n_timepoints: int, *, stim_frames=()) -> list[RTMEvent]:
    """Create a list of RTMEvents for testing."""
    stim_set = set(stim_frames)
    stim_ch = (Channel(config="stim-405", exposure=100),)
    events = []
    for t in range(n_timepoints):
        has_stim = t in stim_set
        events.append(
            RTMEvent(
                index={"t": t, "p": 0},
                channels=(Channel(config="phase-contrast", exposure=50),),
                stim_channels=stim_ch if has_stim else (),
                metadata={},
            )
        )
    return events


# ---------------------------------------------------------------------------
# CircleMicroscope — fake microscope for testing
# ---------------------------------------------------------------------------


class CircleMicroscope(AbstractMicroscope):
    """Fake microscope that generates synthetic circle images.

    No pymmcore dependency. Fires callbacks from a daemon thread.
    """

    def __init__(self):
        super().__init__()
        self._callback = None
        self._cancel = threading.Event()

    def connect_frame(self, callback):
        self._callback = callback

    def disconnect_frame(self, callback):
        if self._callback is callback:
            self._callback = None

    def run_mda(self, event_iter: Iterator[MDAEvent]) -> threading.Thread:
        self._cancel.clear()

        def _run():
            for event in event_iter:
                if self._cancel.is_set():
                    break
                img = make_circle_image()
                if self._callback is not None:
                    self._callback(img, event)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def cancel_mda(self):
        self._cancel.set()


# ---------------------------------------------------------------------------
# run_and_wait helper
# ---------------------------------------------------------------------------


def run_and_wait(ctrl: Controller, events: list[RTMEvent], stim_mode: str = "current"):
    """Run an experiment and block until all pipeline work finishes."""
    ctrl.run_experiment(events, stim_mode=stim_mode, validate=False)

    analyzer = ctrl._analyzer
    # Poll until storage and pipeline are drained
    deadline = time.monotonic() + 30  # 30s timeout
    while time.monotonic() < deadline:
        storage_empty = analyzer._storage_queue.qsize() == 0
        with analyzer.task_lock:
            pipeline_idle = analyzer.active_pipeline_tasks == 0
        deferred_empty = analyzer._deferred_queue.qsize() == 0
        if storage_empty and pipeline_idle and deferred_empty:
            break
        time.sleep(0.1)

    analyzer.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _make_pipeline(path, *, with_stim=False):
    """Build a pipeline with real components for integration testing."""
    return ImageProcessingPipeline(
        storage_path=path,
        segmentators=[SegmentationMethod("labels", OtsuSegmentator(), 0, False)],
        tracker=TrackerTrackpy(search_range=50, memory=3),
        feature_extractor=SimpleFE("labels"),
        stimulator=CenterCircle() if with_stim else None,
    )


# ===================================================================
# Test Class 1: Pipeline component sanity checks
# ===================================================================


class TestPipelineComponents:
    """Unit-level sanity checks for pipeline components on circle images."""

    def test_otsu_segments_two_circles(self):
        img = make_circle_image()
        seg = OtsuSegmentator()
        labels = seg.segment(img)
        unique_labels = set(np.unique(labels)) - {0}
        assert len(unique_labels) == 2, f"Expected 2 labels, got {len(unique_labels)}"

    def test_otsu_labels_have_correct_area(self):
        img = make_circle_image()
        seg = OtsuSegmentator()
        labels = seg.segment(img)
        from skimage.measure import regionprops

        props = regionprops(labels)
        areas = sorted([p.area for p in props])
        # Smaller circle: r=15, expected ~707
        assert (
            abs(areas[0] - EXPECTED_AREA_2) < 10
        ), f"Small circle area {areas[0]} not ~{EXPECTED_AREA_2}"
        # Larger circle: r=20, expected ~1257
        assert (
            abs(areas[1] - EXPECTED_AREA_1) < 10
        ), f"Large circle area {areas[1]} not ~{EXPECTED_AREA_1}"

    def test_simple_fe_extracts_area(self):
        img = make_circle_image()
        seg = OtsuSegmentator()
        labels = seg.segment(img)
        fe = SimpleFE("labels")
        df, _ = fe.extract_features({"labels": labels}, img[np.newaxis, ...])
        assert "label" in df.columns
        assert "area" in df.columns
        assert len(df) == 2

    def test_simple_fe_extracts_positions(self):
        img = make_circle_image()
        seg = OtsuSegmentator()
        labels = seg.segment(img)
        fe = SimpleFE("labels")
        df = fe.extract_positions({"labels": labels})
        assert set(df.columns) >= {"label", "x", "y"}
        assert len(df) == 2
        # Check centroids are near expected positions
        centroids = df.sort_values("label")[["x", "y"]].values
        assert abs(centroids[0][0] - CIRCLE1_CENTER[0]) < 1
        assert abs(centroids[0][1] - CIRCLE1_CENTER[1]) < 1
        assert abs(centroids[1][0] - CIRCLE2_CENTER[0]) < 1
        assert abs(centroids[1][1] - CIRCLE2_CENTER[1]) < 1

    def test_trackpy_links_across_frames(self):
        img = make_circle_image()
        seg = OtsuSegmentator()
        labels = seg.segment(img)
        fe = SimpleFE("labels")
        tracker = TrackerTrackpy(search_range=50, memory=3)
        fov_state = FovState()

        df_tracked = pd.DataFrame()
        for _ in range(3):
            df_new = fe.extract_positions({"labels": labels})
            df_tracked = tracker.track_cells(df_tracked, df_new, fov_state)
            fov_state.fov_timestep_counter += 1

        particles = df_tracked["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        # Each particle should appear in 3 timepoints
        for pid in particles:
            count = len(df_tracked[df_tracked["particle"] == pid])
            assert count == 3, f"Particle {pid} has {count} entries, expected 3"


# ===================================================================
# Test Class 2: End-to-end, no stimulation
# ===================================================================


class TestEndToEndNoStim:
    """5 timepoints, no stimulation — full Controller → Microscope → Pipeline loop."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, with_stim=False)
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS)
        run_and_wait(self.ctrl, self.events)

    def test_raw_images_saved(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_TIMEPOINTS
        ), f"Expected {N_TIMEPOINTS} raw TIFFs, got {len(files)}"

    def test_segmentation_masks_two_labels(self):
        labels_dir = os.path.join(self.path, "labels")
        files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".tiff")])
        assert (
            len(files) == N_TIMEPOINTS
        ), f"Expected {N_TIMEPOINTS} label TIFFs, got {len(files)}"
        for f in files:
            labels = tifffile.imread(os.path.join(labels_dir, f))
            unique = set(np.unique(labels)) - {0}
            assert len(unique) == 2, f"{f}: expected 2 labels, got {len(unique)}"

    def test_tracking_parquet_exists(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        for col in ["particle", "label", "x", "y"]:
            assert col in df.columns, f"Missing column '{col}' in tracking parquet"

    def test_particles_tracked_across_timepoints(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert (
                len(rows) == N_TIMEPOINTS
            ), f"Particle {pid} tracked {len(rows)} times, expected {N_TIMEPOINTS}"

    def test_area_values_correct(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        assert "area" in df.columns, "Missing 'area' column"
        # Get mean area per particle
        mean_areas = df.groupby("particle")["area"].mean().sort_values()
        areas = mean_areas.values
        assert (
            abs(areas[0] - EXPECTED_AREA_2) < 10
        ), f"Small circle area {areas[0]} not ~{EXPECTED_AREA_2}"
        assert (
            abs(areas[1] - EXPECTED_AREA_1) < 10
        ), f"Large circle area {areas[1]} not ~{EXPECTED_AREA_1}"

    def test_centroid_positions_correct(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        # Average position per particle
        mean_pos = df.groupby("particle")[["x", "y"]].mean()
        mean_pos = mean_pos.sort_values("x")
        positions = mean_pos.values
        assert abs(positions[0][0] - CIRCLE1_CENTER[0]) < 1
        assert abs(positions[0][1] - CIRCLE1_CENTER[1]) < 1
        assert abs(positions[1][0] - CIRCLE2_CENTER[0]) < 1
        assert abs(positions[1][1] - CIRCLE2_CENTER[1]) < 1


# ===================================================================
# Test Class 3: End-to-end, stim mode="current"
# ===================================================================


class TestEndToEndStimCurrent:
    """5 timepoints, stim on frames 2-4, mode='current'."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, with_stim=True)
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_stim_masks_saved(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_TIMEPOINTS
        ), f"Expected {N_TIMEPOINTS} stim mask TIFFs, got {len(files)}"

    def test_stim_masks_nonzero_for_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert mask.max() > 0, f"Frame {i} should have nonzero stim mask"
            else:
                assert mask.max() == 0, f"Frame {i} should have zero stim mask"

    def test_segmentation_still_correct(self):
        labels_dir = os.path.join(self.path, "labels")
        files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".tiff")])
        assert len(files) == N_TIMEPOINTS
        for f in files:
            labels = tifffile.imread(os.path.join(labels_dir, f))
            unique = set(np.unique(labels)) - {0}
            assert len(unique) == 2, f"{f}: expected 2 labels, got {len(unique)}"


# ===================================================================
# Test Class 4: End-to-end, stim mode="previous"
# ===================================================================


class TestEndToEndStimPrevious:
    """5 timepoints, stim on frames 2-4, mode='previous'."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, with_stim=True)
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="previous")

    def test_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        assert os.path.isdir(stim_dir)
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_TIMEPOINTS
        ), f"Expected {N_TIMEPOINTS} stim mask files, got {len(files)}"

    def test_segmentation_works(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_consistent(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert len(rows) == N_TIMEPOINTS


# ===================================================================
# Test stimulators for shortcut-path testing
# ===================================================================


class MetadataOnlyStim(Stim):
    """Base Stim — returns all-ones mask using only metadata["img_shape"]."""

    def get_stim_mask(self, metadata: dict):
        h, w = metadata["img_shape"]
        return np.ones((h, w), dtype=np.uint8), None


class ImageBasedStim(StimWithImage):
    """StimWithImage — thresholds the raw image to build a stim mask."""

    def get_stim_mask(self, metadata: dict, img: np.ndarray):
        # Use first channel, threshold at half-max
        frame = img[0] if img.ndim == 3 else img
        thresh = frame.max() / 2
        mask = (frame > thresh).astype(np.uint8)
        return mask, None


def _make_pipeline_with_stim(path, stimulator):
    """Build a pipeline with a specific stimulator for shortcut testing."""
    return ImageProcessingPipeline(
        storage_path=path,
        segmentators=[SegmentationMethod("labels", OtsuSegmentator(), 0, False)],
        tracker=TrackerTrackpy(search_range=50, memory=3),
        feature_extractor=SimpleFE("labels"),
        stimulator=stimulator,
    )


# ===================================================================
# Test Class 5: Stim (metadata-only) shortcut — mode="current"
# ===================================================================


class TestStimMetadataOnlyCurrent:
    """Base Stim bypasses pipeline — mask computed synchronously in controller."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline_with_stim(self.path, MetadataOnlyStim())
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_stim_masks_saved(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_stim_masks_nonzero_for_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert (
                    mask.max() > 0
                ), f"Frame {i}: metadata-only stim should produce nonzero mask"
            else:
                assert (
                    mask.max() == 0
                ), f"Frame {i}: non-stim frame should have zero mask"

    def test_segmentation_still_runs(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_still_works(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2


# ===================================================================
# Test Class 6: Stim (metadata-only) shortcut — mode="previous"
# ===================================================================


class TestStimMetadataOnlyPrevious:
    """Base Stim in 'previous' mode — still computes synchronously."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline_with_stim(self.path, MetadataOnlyStim())
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="previous")

    def test_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_consistent(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2
        for pid in particles:
            assert len(df[df["particle"] == pid]) == N_TIMEPOINTS


# ===================================================================
# Test Class 7: StimWithImage shortcut — mode="current"
# ===================================================================


class TestStimWithImageCurrent:
    """StimWithImage — mask computed in storage worker before pipeline."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline_with_stim(self.path, ImageBasedStim())
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_stim_masks_saved(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_stim_masks_nonzero_for_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert (
                    mask.max() > 0
                ), f"Frame {i}: image-based stim should produce nonzero mask"
            else:
                assert (
                    mask.max() == 0
                ), f"Frame {i}: non-stim frame should have zero mask"

    def test_segmentation_still_runs(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_still_works(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2


# ===================================================================
# Test Class 8: StimWithImage shortcut — mode="previous"
# ===================================================================


class TestStimWithImagePrevious:
    """StimWithImage in 'previous' mode — mask from storage worker, used next frame."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline_with_stim(self.path, ImageBasedStim())
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="previous")

    def test_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_consistent(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2
        for pid in particles:
            assert len(df[df["particle"] == pid]) == N_TIMEPOINTS


# ===================================================================
# Stress test helpers
# ===================================================================


class SlowSegmentator(OtsuSegmentator):
    """OtsuSegmentator that sleeps *delay* seconds per frame to simulate load."""

    def __init__(self, delay: float = 1.0):
        super().__init__()
        self._delay = delay

    def segment(self, image: np.ndarray) -> np.ndarray:
        time.sleep(self._delay)
        return super().segment(image)


def run_and_wait_long(
    ctrl: Controller,
    events: list[RTMEvent],
    stim_mode: str = "current",
    timeout: float = 120,
):
    """Like run_and_wait but with a longer timeout for slow pipelines."""
    ctrl.run_experiment(events, stim_mode=stim_mode, validate=False)

    analyzer = ctrl._analyzer
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        storage_empty = analyzer._storage_queue.qsize() == 0
        with analyzer.task_lock:
            pipeline_idle = analyzer.active_pipeline_tasks == 0
        deferred_empty = analyzer._deferred_queue.qsize() == 0
        if storage_empty and pipeline_idle and deferred_empty:
            break
        time.sleep(0.5)

    analyzer.shutdown(wait=True)


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
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = ImageProcessingPipeline(
            storage_path=self.path,
            segmentators=[
                SegmentationMethod(
                    "labels", SlowSegmentator(STRESS_SEG_DELAY), 0, False
                )
            ],
            tracker=TrackerTrackpy(search_range=50, memory=3),
            feature_extractor=SimpleFE("labels"),
            stimulator=CenterCircle(),
        )
        self.mic = CircleMicroscope()
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


class CrashingStimulator(CenterCircle):
    """StimWithPipeline that raises on every stim call."""

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        raise RuntimeError("Stimulation crashed!")


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
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            segmentator=CrashingSegmentator(),
        )
        self.mic = CircleMicroscope()
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
        self.mic = CircleMicroscope()
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
    def setup(self, tmp_dir):
        self.path = tmp_dir
        # Stim on last frame only so no subsequent frame waits on the broken put()
        self.n_frames = N_CRASH_FRAMES
        self.stim_frames = (self.n_frames - 1,)
        self.pipeline = _make_crashing_pipeline(
            self.path,
            stim=CrashingStimulator(),
        )
        self.mic = CircleMicroscope()
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
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_crashing_pipeline(
            self.path,
            fe=CrashingFE("labels"),
        )
        self.mic = CircleMicroscope()
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


def wait_for_pipeline(analyzer: Analyzer, timeout: float = 30):
    """Block until all storage and pipeline work finishes (no shutdown)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        storage_empty = analyzer._storage_queue.qsize() == 0
        with analyzer.task_lock:
            pipeline_idle = analyzer.active_pipeline_tasks == 0
        deferred_empty = analyzer._deferred_queue.qsize() == 0
        if storage_empty and pipeline_idle and deferred_empty:
            break
        time.sleep(0.1)


# ===================================================================
# Test Class 14: Continue experiment — sequential continuation
# ===================================================================


N_PHASE1_FRAMES = 3
N_PHASE2_FRAMES = 3
N_TOTAL_FRAMES = N_PHASE1_FRAMES + N_PHASE2_FRAMES


class TestContinueExperiment:
    """Run 3 frames, continue with 3 more, verify seamless continuation."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, with_stim=False)
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)

        # Phase 1
        phase1_events = make_events(N_PHASE1_FRAMES)
        self.ctrl.run_experiment(phase1_events, validate=False)
        wait_for_pipeline(self.ctrl._analyzer)

        # Phase 2 — continue (reuses Analyzer)
        phase2_events = make_events(N_PHASE2_FRAMES)
        self.ctrl.continue_experiment(phase2_events, validate=False)
        wait_for_pipeline(self.ctrl._analyzer)

        self.ctrl.finish_experiment()

    def test_correct_number_of_raw_images(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_TOTAL_FRAMES
        ), f"Expected {N_TOTAL_FRAMES} raw TIFFs, got {len(files)}"

    def test_continuous_filenames(self):
        """Filenames should be 000_00000 through 000_00005."""
        raw_dir = os.path.join(self.path, "raw")
        files = sorted(os.listdir(raw_dir))
        expected = [f"000_{t:05d}.tiff" for t in range(N_TOTAL_FRAMES)]
        assert files == expected, f"Expected {expected}, got {files}"

    def test_segmentation_masks_for_all_frames(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == N_TOTAL_FRAMES

    def test_tracking_across_continuation_boundary(self):
        """Particles should be tracked across the continuation boundary."""
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert (
                len(rows) == N_TOTAL_FRAMES
            ), f"Particle {pid} tracked {len(rows)} times, expected {N_TOTAL_FRAMES}"

    def test_t_offset_after_finish(self):
        """After finish_experiment(), offsets should be reset."""
        assert self.ctrl._t_offset == 0
        assert self.ctrl._experiment_start is None


# ===================================================================
# Test Class 15: Extend experiment — dynamic extension mid-run
# ===================================================================


N_INITIAL_FRAMES = 3
N_EXTEND_FRAMES = 3
N_EXTENDED_TOTAL = N_INITIAL_FRAMES + N_EXTEND_FRAMES


class TestExtendExperiment:
    """Start 3 frames, extend with 3 more mid-run, verify all processed."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, with_stim=False)
        self.mic = CircleMicroscope()
        self.ctrl = Controller(self.mic, self.pipeline)

        initial_events = make_events(N_INITIAL_FRAMES)
        extend_events = make_events(N_EXTEND_FRAMES)

        # Use _pre_loop_hook to call extend_experiment after the event queue
        # is set up but before the loop starts draining it.
        def inject_extension():
            self.ctrl.extend_experiment(extend_events)

        self.ctrl._pre_loop_hook = inject_extension
        self.ctrl.run_experiment(initial_events, validate=False)

        wait_for_pipeline(self.ctrl._analyzer)
        self.ctrl.finish_experiment()

    def test_correct_number_of_raw_images(self):
        raw_dir = os.path.join(self.path, "raw")
        files = [f for f in os.listdir(raw_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_EXTENDED_TOTAL
        ), f"Expected {N_EXTENDED_TOTAL} raw TIFFs, got {len(files)}"

    def test_continuous_filenames(self):
        raw_dir = os.path.join(self.path, "raw")
        files = sorted(os.listdir(raw_dir))
        expected = [f"000_{t:05d}.tiff" for t in range(N_EXTENDED_TOTAL)]
        assert files == expected, f"Expected {expected}, got {files}"

    def test_tracking_covers_all_frames(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert (
                len(rows) == N_EXTENDED_TOTAL
            ), f"Particle {pid} tracked {len(rows)} times, expected {N_EXTENDED_TOTAL}"


# ===================================================================
# Test Class 16: continue_experiment without prior run raises error
# ===================================================================


class TestContinueWithoutRunRaises:
    """Calling continue_experiment without run_experiment should raise."""

    def test_raises_runtime_error(self, tmp_dir):
        pipeline = _make_pipeline(tmp_dir, with_stim=False)
        mic = CircleMicroscope()
        ctrl = Controller(mic, pipeline)
        events = make_events(3)
        with pytest.raises(RuntimeError, match="No experiment to continue"):
            ctrl.continue_experiment(events, validate=False)


# ===================================================================
# Test Class 17: Burst dispatch — no signal loss under back-pressure
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


class _TinyMicroscope(AbstractMicroscope):
    """Microscope emitting tiny images as fast as possible."""

    def __init__(self):
        super().__init__()
        self._callback = None
        self._cancel = threading.Event()

    def connect_frame(self, callback):
        self._callback = callback

    def disconnect_frame(self, callback):
        if self._callback is callback:
            self._callback = None

    def run_mda(self, event_iter):
        self._cancel.clear()

        def _run():
            for event in event_iter:
                if self._cancel.is_set():
                    break
                if self._callback is not None:
                    self._callback(_make_tiny_image(), event)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def cancel_mda(self):
        self._cancel.set()


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
    mic = _TinyMicroscope()
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
