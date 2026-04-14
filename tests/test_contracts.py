"""Contract tests for pipeline components.

Each component (segmentator, tracker, stimulator, feature extractor) must
return data with the correct structure.  These tests use small synthetic
images so they run fast and don't need ML models or hardware.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from faro.core.data_structures import FovState
from faro.feature_extraction.base import FeatureExtractor
from faro.feature_extraction.erk_ktr import FE_ErkKtr
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import (
    DummySegmentator,
    OtsuSegmentator,
    Segmentator,
    SegmentatorBinary,
)
from faro.stimulation.base import (
    Stim,
    StimNothing,
    StimTopEdgeMeta,
    StimWholeFOV,
    StimWithPipeline,
)
from faro.stimulation.center_circle import CenterCircle
from faro.stimulation.moving_line_20x import StimLine
from faro.tracking.trackpy import TrackerTrackpy

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 128, 128


def _circle_image(centers=((40, 40), (90, 90)), radii=(12, 10), value=200):
    """2D uint16 image with bright circles on a dark background."""
    img = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    yy, xx = np.ogrid[:IMG_H, :IMG_W]
    for (cy, cx), r in zip(centers, radii):
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        img[mask] = value
    return img


def _label_image(centers=((40, 40), (90, 90)), radii=(12, 10)):
    """Integer label image with two labeled circles (1, 2) on 0 background."""
    labels = np.zeros((IMG_H, IMG_W), dtype=np.int32)
    yy, xx = np.ogrid[:IMG_H, :IMG_W]
    for i, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        labels[mask] = i
    return labels


# =========================================================================
# Segmentation contracts
# =========================================================================


def _assert_valid_segmentation(result, input_image):
    """Check the contract every Segmentator.segment() must satisfy."""
    assert isinstance(
        result, np.ndarray
    ), f"Expected np.ndarray, got {type(result).__name__}"
    assert (
        result.ndim == 2
    ), f"Expected 2D label image, got {result.ndim}D (shape {result.shape})"
    assert (
        result.shape == input_image.shape[-2:]
    ), f"Shape mismatch: result {result.shape} vs input {input_image.shape[-2:]}"
    assert np.issubdtype(
        result.dtype, np.integer
    ), f"Expected integer labels, got dtype {result.dtype}"
    assert result.min() >= 0, f"Labels must be non-negative, got min={result.min()}"


class TestSegmentationContract:
    """Every built-in segmentator must return a valid label image."""

    @pytest.fixture(
        params=[
            SegmentatorBinary(),
            OtsuSegmentator(),
            DummySegmentator(),
        ],
        ids=["Binary", "Otsu", "Dummy"],
    )
    def segmentator(self, request):
        return request.param

    def test_returns_valid_labels(self, segmentator):
        img = _circle_image()
        result = segmentator.segment(img)
        _assert_valid_segmentation(result, img)

    def test_empty_image(self, segmentator):
        img = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
        result = segmentator.segment(img)
        _assert_valid_segmentation(result, img)

    def test_saturated_image(self, segmentator):
        img = np.full((IMG_H, IMG_W), 65535, dtype=np.uint16)
        result = segmentator.segment(img)
        _assert_valid_segmentation(result, img)

    def test_single_pixel_object(self, segmentator):
        img = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
        img[64, 64] = 200
        result = segmentator.segment(img)
        _assert_valid_segmentation(result, img)


class TestSegmentationSanity:
    """Basic sanity: known input → expected label count."""

    def test_binary_finds_two_circles(self):
        result = SegmentatorBinary().segment(_circle_image())
        assert result.max() == 2, f"Expected 2 labels, got {result.max()}"

    def test_otsu_finds_two_circles(self):
        result = OtsuSegmentator().segment(_circle_image())
        assert result.max() == 2, f"Expected 2 labels, got {result.max()}"

    def test_dummy_returns_single_label(self):
        result = DummySegmentator().segment(_circle_image())
        assert np.all(result == 1)

    def test_empty_image_returns_no_labels(self):
        img = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
        result = SegmentatorBinary().segment(img)
        assert result.max() == 0, "Empty image should produce no labels"


# =========================================================================
# Tracking contracts
# =========================================================================


def _assert_valid_tracking(result):
    """Check the contract every Tracker.track_cells() must satisfy."""
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected pd.DataFrame, got {type(result).__name__}"
    required = {"particle", "x", "y", "label"}
    missing = required - set(result.columns)
    assert not missing, f"Missing required columns: {missing}"
    assert "fov_timestep" in result.columns, "Missing 'fov_timestep' column"
    assert not result.duplicated(
        subset=["particle", "fov_timestep"]
    ).any(), "Duplicate (particle, fov_timestep) pairs found"


def _detections(positions, timestep_label_start=1):
    """Build a detections DataFrame like the pipeline produces."""
    rows = [
        {"x": y, "y": x, "label": i}
        for i, (y, x) in enumerate(positions, start=timestep_label_start)
    ]
    return pd.DataFrame(rows)


class TestTrackingContract:

    def test_first_frame_returns_valid_df(self):
        tracker = TrackerTrackpy(search_range=50, memory=3)
        fov = FovState()
        df_new = _detections([(40, 40), (90, 90)])
        result = tracker.track_cells(pd.DataFrame(), df_new, fov)
        _assert_valid_tracking(result)

    def test_two_frames_returns_valid_df(self):
        tracker = TrackerTrackpy(search_range=50, memory=3)
        fov = FovState()

        df1 = _detections([(40, 40), (90, 90)])
        df_tracked = tracker.track_cells(pd.DataFrame(), df1, fov)
        fov.fov_timestep_counter += 1

        df2 = _detections([(41, 41), (91, 91)])
        result = tracker.track_cells(df_tracked, df2, fov)
        _assert_valid_tracking(result)

    def test_consistent_particle_ids_across_frames(self):
        tracker = TrackerTrackpy(search_range=50, memory=3)
        fov = FovState()

        df1 = _detections([(40, 40), (90, 90)])
        df_tracked = tracker.track_cells(pd.DataFrame(), df1, fov)
        fov.fov_timestep_counter += 1

        # Small displacement — same particles
        df2 = _detections([(42, 42), (92, 92)])
        result = tracker.track_cells(df_tracked, df2, fov)

        assert (
            result["particle"].nunique() == 2
        ), "Two stationary particles should keep two unique IDs"

    def test_empty_second_frame(self):
        """No detections in a frame should not crash."""
        tracker = TrackerTrackpy(search_range=50, memory=3)
        fov = FovState()

        df1 = _detections([(40, 40)])
        df_tracked = tracker.track_cells(pd.DataFrame(), df1, fov)
        fov.fov_timestep_counter += 1

        df2 = _detections([])  # empty
        result = tracker.track_cells(df_tracked, df2, fov)
        _assert_valid_tracking(result)

    def test_particle_memory(self):
        """A particle missing for <= memory frames should be re-linked."""
        tracker = TrackerTrackpy(search_range=50, memory=3)
        fov = FovState()

        # Frame 0: particle at (40,40)
        df_tracked = tracker.track_cells(pd.DataFrame(), _detections([(40, 40)]), fov)
        fov.fov_timestep_counter += 1

        # Frame 1: particle disappears
        df_tracked = tracker.track_cells(df_tracked, _detections([]), fov)
        fov.fov_timestep_counter += 1

        # Frame 2: particle reappears nearby
        df_tracked = tracker.track_cells(df_tracked, _detections([(42, 42)]), fov)

        assert (
            df_tracked["particle"].nunique() == 1
        ), "Particle reappearing within memory window should keep its ID"


# =========================================================================
# Stimulation contracts
# =========================================================================


def _assert_valid_stim_mask(result, expected_shape):
    """Check the contract every stimulator.get_stim_mask() must satisfy."""
    assert (
        isinstance(result, tuple) and len(result) == 2
    ), f"Expected (mask, info) tuple, got {type(result).__name__}"
    mask = result[0]
    assert isinstance(
        mask, np.ndarray
    ), f"Mask must be np.ndarray, got {type(mask).__name__}"
    assert mask.ndim == 2, f"Mask must be 2D, got {mask.ndim}D (shape {mask.shape})"
    assert (
        mask.shape == expected_shape
    ), f"Mask shape {mask.shape} != expected {expected_shape}"
    assert mask.dtype == np.uint8, f"Mask must be uint8, got {mask.dtype}"


class TestStimContract:
    """Every built-in Stim must return a valid (mask, info) tuple."""

    def test_stim_whole_fov(self):
        meta = {"img_shape": (IMG_H, IMG_W)}
        result = StimWholeFOV().get_stim_mask(meta)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] is True, "StimWholeFOV should return True (whole FOV on)"

    def test_stim_nothing(self):
        meta = {"img_shape": (IMG_H, IMG_W)}
        result = StimNothing().get_stim_mask(meta)
        _assert_valid_stim_mask(result, (IMG_H, IMG_W))
        assert result[0].sum() == 0, "StimNothing mask should be all zeros"

    def test_stim_line(self):
        stim = StimLine(
            first_stim_frame=0,
            frames_for_1_loop=10,
            stripe_width=20,
            n_frames_total=30,
            mask_height=IMG_H,
            mask_width=IMG_W,
        )
        meta = {"img_shape": (IMG_H, IMG_W), "timestep": 3, "stim": True}
        result = stim.get_stim_mask(meta)
        _assert_valid_stim_mask(result, (IMG_H, IMG_W))

    def test_stim_line_no_stim_frame(self):
        stim = StimLine(mask_height=IMG_H, mask_width=IMG_W)
        meta = {"img_shape": (IMG_H, IMG_W), "timestep": 0, "stim": False}
        result = stim.get_stim_mask(meta)
        _assert_valid_stim_mask(result, (IMG_H, IMG_W))
        assert result[0].sum() == 0, "Non-stim frame should be empty mask"

    def test_center_circle(self):
        labels = _label_image()
        result = CenterCircle().get_stim_mask(
            label_images={"labels": labels},
            metadata={"img_shape": (IMG_H, IMG_W)},
        )
        _assert_valid_stim_mask(result, (IMG_H, IMG_W))
        assert result[0].sum() > 0, "CenterCircle mask should not be empty"

    def test_stim_top_edge_meta(self):
        labels = _label_image()
        meta = {"img_shape": (IMG_H, IMG_W), "stim_fraction": 0.5}
        result = StimTopEdgeMeta().get_stim_mask(
            label_images={"labels": labels},
            metadata=meta,
        )
        _assert_valid_stim_mask(result, (IMG_H, IMG_W))

    def test_stim_top_edge_meta_fraction_zero(self):
        labels = _label_image()
        meta = {"img_shape": (IMG_H, IMG_W), "stim_fraction": 0.0}
        result = StimTopEdgeMeta().get_stim_mask(
            label_images={"labels": labels},
            metadata=meta,
        )
        _assert_valid_stim_mask(result, (IMG_H, IMG_W))
        assert result[0].sum() == 0, "stim_fraction=0 should produce empty mask"

    def test_stim_top_edge_meta_empty_labels(self):
        labels = np.zeros((IMG_H, IMG_W), dtype=np.int32)
        meta = {"img_shape": (IMG_H, IMG_W), "stim_fraction": 0.5}
        result = StimTopEdgeMeta().get_stim_mask(
            label_images={"labels": labels},
            metadata=meta,
        )
        _assert_valid_stim_mask(result, (IMG_H, IMG_W))
        assert result[0].sum() == 0, "No labels should produce empty mask"


# =========================================================================
# Feature extraction contracts
# =========================================================================


def _assert_valid_positions(result):
    """Check the contract for FeatureExtractor.extract_positions()."""
    assert isinstance(
        result, pd.DataFrame
    ), f"Expected pd.DataFrame, got {type(result).__name__}"
    required = {"label", "x", "y"}
    missing = required - set(result.columns)
    assert not missing, f"Missing required columns: {missing}"


def _assert_valid_features(result):
    """Check the contract for FeatureExtractor.extract_features()."""
    assert (
        isinstance(result, tuple) and len(result) == 2
    ), f"Expected (table, extra_masks) tuple, got {type(result).__name__}"
    table = result[0]
    assert isinstance(
        table, pd.DataFrame
    ), f"Feature table must be pd.DataFrame, got {type(table).__name__}"
    assert "label" in table.columns, "Feature table must contain 'label' column"


class TestFeatureExtractionContract:

    def test_simple_fe_extract_positions(self):
        labels = {"labels": _label_image()}
        fe = SimpleFE(used_mask="labels")
        result = fe.extract_positions(labels)
        _assert_valid_positions(result)
        assert len(result) == 2, "Should find 2 objects"

    def test_simple_fe_extract_features(self):
        labels = {"labels": _label_image()}
        img = _circle_image()[np.newaxis, :, :]  # add channel dim
        fe = SimpleFE(used_mask="labels")
        result = fe.extract_features(labels, img)
        _assert_valid_features(result)
        assert "area" in result[0].columns, "SimpleFE must return 'area'"

    def test_simple_fe_empty_labels(self):
        labels = {"labels": np.zeros((IMG_H, IMG_W), dtype=np.int32)}
        img = np.zeros((1, IMG_H, IMG_W), dtype=np.uint16)
        fe = SimpleFE(used_mask="labels")
        result = fe.extract_features(labels, img)
        _assert_valid_features(result)
        assert len(result[0]) == 0, "No labels should produce empty table"

    def test_erk_ktr_extract_features(self):
        labels = {"labels": _label_image()}
        # ERK-KTR expects at least 2 channels (C, H, W)
        img = np.stack([_circle_image(), _circle_image(), _circle_image()])
        fe = FE_ErkKtr(used_mask="labels", margin=2, distance=4)
        result = fe.extract_features(labels, img)
        _assert_valid_features(result)
        assert "cnr" in result[0].columns, "ERK-KTR must return 'cnr'"
        assert "area_nuc" in result[0].columns, "ERK-KTR must return 'area_nuc'"

    def test_erk_ktr_ring_is_valid_labels(self):
        labels = _label_image()
        fe = FE_ErkKtr(used_mask="labels", margin=2, distance=4)
        ring = fe.extract_ring(labels)
        assert isinstance(ring, np.ndarray)
        assert ring.shape == labels.shape
        assert np.issubdtype(ring.dtype, np.integer)
        # Ring should not overlap with nuclei
        overlap = (ring > 0) & (labels > 0)
        assert not overlap.any(), "Ring should not overlap with nuclear labels"

    def test_extract_positions_centroids_in_bounds(self):
        labels = {"labels": _label_image()}
        fe = SimpleFE(used_mask="labels")
        result = fe.extract_positions(labels)
        assert (result["x"] >= 0).all() and (result["x"] < IMG_H).all()
        assert (result["y"] >= 0).all() and (result["y"] < IMG_W).all()
