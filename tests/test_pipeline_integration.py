"""Core end-to-end pipeline integration tests.

Drives the Controller → Pipeline stack through :class:`FakeMicroscope`
with :class:`CircleScene`. No pymmcore-plus install required; all
acquisition flow goes through the fake mmc's MDA signal chain.

Covers the happy-path end-to-end run and the continue/extend
experiment lifecycle. Stim-mode tests live in
:mod:`tests.test_pipeline_stim`; failure/crash/stress/burst tests
live in :mod:`tests.test_pipeline_failures`.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from faro.core.controller import Controller

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import (
    CIRCLE1_CENTER,
    CIRCLE1_RADIUS,
    CIRCLE2_CENTER,
    CIRCLE2_RADIUS,
    CircleScene,
    assert_no_background_errors,
    make_events,
    run_and_wait,
    tracker,  # noqa: F401 — parametrized fixture, auto-discovered by pytest
)
from tests.fixtures import make_pipeline as _make_pipeline


EXPECTED_AREA_1 = math.pi * CIRCLE1_RADIUS**2  # ~1257
EXPECTED_AREA_2 = math.pi * CIRCLE2_RADIUS**2  # ~707

N_TIMEPOINTS = 5


# ===================================================================
# Test Class 2: End-to-end, no stimulation
# ===================================================================


class TestEndToEndNoStim:
    """5 timepoints, no stimulation — full Controller → Microscope → Pipeline loop."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, tracker=tracker, with_stim=False)
        self.mic = FakeMicroscope(CircleScene())
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


class TestContinueExperimentOmeZarr:
    """Continued and extended single-position OME-Zarr runs grow the time axis.

    Regression: the single-position raw array used to be an ome-writers
    stream created with a fixed time count, so every frame past the first
    run's length failed with "would exceed total of N frames" and was
    silently dropped (only logged as a background error).
    """

    @staticmethod
    def _check_store(ctrl, tmp_dir, n_total):
        import zarr

        assert ctrl.background_errors == [], [e.message for e in ctrl.background_errors]
        raw = zarr.open(os.path.join(tmp_dir, "acquisition.ome.zarr"), mode="r")["0"]
        assert raw.shape[0] == n_total, raw.shape
        assert raw[n_total - 1].max() > 0, "last appended frame is empty"

    def test_continued_frames_land_in_the_store(self, tmp_dir, tracker):
        from faro.core.writers import OmeZarrWriter

        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=False)
        ctrl = Controller(
            FakeMicroscope(CircleScene()), pipeline, writer=OmeZarrWriter(tmp_dir)
        )
        ctrl.run_experiment(make_events(N_PHASE1_FRAMES), validate=False).wait()
        ctrl._analyzer.wait_idle()
        ctrl.continue_experiment(make_events(N_PHASE2_FRAMES), validate=False).wait()
        ctrl._analyzer.wait_idle()
        ctrl.finish_experiment()
        self._check_store(ctrl, tmp_dir, N_TOTAL_FRAMES)

    def test_extended_frames_land_in_the_store(self, tmp_dir, tracker):
        from faro.core.writers import OmeZarrWriter

        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=False)
        ctrl = Controller(
            FakeMicroscope(CircleScene()), pipeline, writer=OmeZarrWriter(tmp_dir)
        )
        # Same hook as TestExtendExperiment: extend once the queue exists.
        ctrl._pre_loop_hook = lambda: ctrl.extend_experiment(
            make_events(N_PHASE2_FRAMES)
        )
        ctrl.run_experiment(make_events(N_PHASE1_FRAMES), validate=False).wait()
        ctrl._analyzer.wait_idle()
        ctrl.finish_experiment()
        self._check_store(ctrl, tmp_dir, N_TOTAL_FRAMES)


class TestContinueExperimentModeMismatchRaises:
    """``continue_experiment`` must refuse to change ``stim_mode`` mid-run."""

    def test_raises_on_mode_change(self, tmp_dir, tracker):
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=False)
        ctrl = Controller(FakeMicroscope(CircleScene()), pipeline)
        events = make_events(2)
        try:
            ctrl.run_experiment(events, stim_mode="current", validate=False).wait()
            ctrl._analyzer.wait_idle()
            with pytest.raises(RuntimeError, match="stim_mode"):
                ctrl.continue_experiment(
                    make_events(2), stim_mode="previous", validate=False
                )
        finally:
            ctrl.finish_experiment()


N_PHASE1_FRAMES = 3
N_PHASE2_FRAMES = 3
N_TOTAL_FRAMES = N_PHASE1_FRAMES + N_PHASE2_FRAMES


class TestContinueExperiment:
    """Run 3 frames, continue with 3 more, verify seamless continuation."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, tracker=tracker, with_stim=False)
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)

        # Phase 1
        phase1_events = make_events(N_PHASE1_FRAMES)
        self.ctrl.run_experiment(phase1_events, validate=False).wait()
        self.ctrl._analyzer.wait_idle()

        # Phase 2 — continue (reuses Analyzer)
        phase2_events = make_events(N_PHASE2_FRAMES)
        self.ctrl.continue_experiment(phase2_events, validate=False).wait()
        self.ctrl._analyzer.wait_idle()

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
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, tracker=tracker, with_stim=False)
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)

        initial_events = make_events(N_INITIAL_FRAMES)
        extend_events = make_events(N_EXTEND_FRAMES)

        # Use _pre_loop_hook to call extend_experiment after the event queue
        # is set up but before the loop starts draining it.
        def inject_extension():
            self.ctrl.extend_experiment(extend_events)

        self.ctrl._pre_loop_hook = inject_extension
        self.ctrl.run_experiment(initial_events, validate=False).wait()

        self.ctrl._analyzer.wait_idle()
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

    def test_raises_runtime_error(self, tmp_dir, tracker):
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=False)
        mic = FakeMicroscope(CircleScene())
        ctrl = Controller(mic, pipeline)
        events = make_events(3)
        with pytest.raises(RuntimeError, match="No experiment to continue"):
            ctrl.continue_experiment(events, validate=False)


# ===================================================================
# Test Class 17: Burst dispatch — no signal loss under back-pressure
# ===================================================================


class TestEmptyFrameEdgeCases:
    """Pipeline must not crash when segmentation finds zero cells.

    Covers the guards in extract_and_merge_features, labels_to_particles,
    and the ref-channel FE path.
    """

    def test_all_frames_blank_no_errors(self, tmp_dir, tracker):
        """Every frame is blank — pipeline runs, no background errors."""
        mic = FakeMicroscope(CircleScene(blank_frames={0, 1, 2, 3, 4}))
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=False)
        ctrl = Controller(mic, pipeline)
        events = make_events(5)
        run_and_wait(ctrl, events)
        assert_no_background_errors(ctrl)

    def test_all_frames_blank_with_stim_no_errors(self, tmp_dir, tracker):
        """Blank frames + stim active — stim mask dispatch must not crash."""
        mic = FakeMicroscope(CircleScene(blank_frames={0, 1, 2, 3, 4}))
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=True)
        ctrl = Controller(mic, pipeline)
        events = make_events(5, stim_frames=(1, 2, 3, 4))
        run_and_wait(ctrl, events, stim_mode="current")
        assert_no_background_errors(ctrl)

    def test_cells_appear_after_blank_start(self, tmp_dir, tracker):
        """Frames 0-1 blank, frames 2-4 have cells — tracking picks up."""
        mic = FakeMicroscope(CircleScene(blank_frames={0, 1}))
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=False)
        ctrl = Controller(mic, pipeline)
        events = make_events(5)
        run_and_wait(ctrl, events)
        assert_no_background_errors(ctrl)
        tracks_dir = os.path.join(tmp_dir, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert parquet_files, "No parquet files written"
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        # Cells appear at frames 2-4 → 3 observations per particle
        for pid in particles:
            count = len(df[df["particle"] == pid])
            assert count == 3, f"Particle {pid} tracked {count} times, expected 3"

    def test_cells_disappear_then_reappear(self, tmp_dir, tracker):
        """Frames 0,1 have cells, frame 2 blank, frames 3,4 have cells.

        The tracker should link across the gap (memory=3 allows up to 3
        missing frames) and produce the same 2 particle IDs throughout.
        """
        mic = FakeMicroscope(CircleScene(blank_frames={2}))
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=False)
        ctrl = Controller(mic, pipeline)
        events = make_events(5)
        run_and_wait(ctrl, events)
        assert_no_background_errors(ctrl)
        tracks_dir = os.path.join(tmp_dir, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        # 4 observations per particle (frames 0,1,3,4)
        for pid in particles:
            count = len(df[df["particle"] == pid])
            assert count == 4, f"Particle {pid} tracked {count} times, expected 4"

    def test_cells_disappear_reappear_with_stim(self, tmp_dir, tracker):
        """Same gap pattern but with stim active — verifies stim_mask_queue
        doesn't deadlock or crash when pipeline has no labels to stim."""
        mic = FakeMicroscope(CircleScene(blank_frames={2}))
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, with_stim=True)
        ctrl = Controller(mic, pipeline)
        events = make_events(5, stim_frames=(1, 2, 3, 4))
        run_and_wait(ctrl, events, stim_mode="current")
        assert_no_background_errors(ctrl)
