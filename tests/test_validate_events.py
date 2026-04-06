"""Tests for pipeline.validate_pipeline() (and the validate_events wrapper).

Covers two validation axes:
1. **Signature checks** — subclass methods must accept the parameters
   declared by their base class.
2. **Required metadata checks** — events must carry the metadata keys
   declared by each pipeline component.
"""

from __future__ import annotations

import tempfile
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

from rtm_pymmcore.core.data_structures import (
    Channel,
    PowerChannel,
    RTMEvent,
    RTMSequence,
    SegmentationMethod,
)
from rtm_pymmcore.core.pipeline import ImageProcessingPipeline
from rtm_pymmcore.segmentation.base import Segmentator
from rtm_pymmcore.stimulation.base import Stim, StimWithImage, StimWithPipeline
from rtm_pymmcore.tracking.base import Tracker
from rtm_pymmcore.feature_extraction.base import FeatureExtractor


# ---------------------------------------------------------------------------
# Helpers: minimal concrete implementations for the happy path
# ---------------------------------------------------------------------------

class GoodSegmentator(Segmentator):
    def segment(self, image):
        return np.zeros_like(image)


class GoodTracker(Tracker):
    def track_cells(self, df_old, df_new, fov_state):
        return pd.concat([df_old, df_new], ignore_index=True)


class GoodFeatureExtractor(FeatureExtractor):
    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        return pd.DataFrame(), None


class GoodStimulator(StimWithPipeline):
    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        return np.zeros((10, 10), dtype=np.uint8), None


# ---------------------------------------------------------------------------
# Helpers: broken implementations (missing params)
# ---------------------------------------------------------------------------

class StimMissingTracks(StimWithPipeline):
    """Forgot to add the ``tracks`` kwarg."""
    def get_stim_mask(self, label_images, metadata=None, img=None):
        return np.zeros((10, 10), dtype=np.uint8), None


class SegMissingImage(Segmentator):
    """Forgot to accept ``image``."""
    def segment(self):
        return np.zeros((10, 10))


class TrackerMissingFovState(Tracker):
    """Forgot to accept ``fov_state``."""
    def track_cells(self, df_old, df_new):
        return pd.concat([df_old, df_new], ignore_index=True)


class FEMissingMetadata(FeatureExtractor):
    """Forgot to accept ``metadata``."""
    def extract_features(self, labels, image, df_tracked=None):
        return pd.DataFrame(), None


# ---------------------------------------------------------------------------
# Helpers: implementations that accept **kwargs (should always pass)
# ---------------------------------------------------------------------------

class StimWithKwargs(StimWithPipeline):
    def get_stim_mask(self, label_images, **kwargs):
        return np.zeros((10, 10), dtype=np.uint8), None


# ---------------------------------------------------------------------------
# Helpers: implementations with required_metadata
# ---------------------------------------------------------------------------

class StimNeedsFraction(StimWithPipeline):
    required_metadata: set[str] = {"stim_fraction"}

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        return np.zeros((10, 10), dtype=np.uint8), None


class TrackerNeedsCondition(Tracker):
    required_metadata: set[str] = {"condition"}

    def track_cells(self, df_old, df_new, fov_state):
        return pd.concat([df_old, df_new], ignore_index=True)


class SegNeedsChannel(Segmentator):
    required_metadata: set[str] = {"seg_channel"}

    def segment(self, image):
        return np.zeros_like(image)


class FENeedsThreshold(FeatureExtractor):
    required_metadata: set[str] = {"fe_threshold"}

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        return pd.DataFrame(), None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_path_cleanup():
    """Provide a temp directory that is cleaned up after the test."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _make_pipeline(path, *, segmentator=None, tracker=None, fe=None, stim=None):
    """Build a pipeline with the given components (defaults to Good* variants)."""
    seg = segmentator or GoodSegmentator()
    return ImageProcessingPipeline(
        storage_path=path,
        segmentators=[SegmentationMethod("labels", seg, 0, False)],
        tracker=tracker,
        feature_extractor=fe,
        stimulator=stim,
    )


def _make_events(*, stim=False, metadata=None):
    """Return a small list of RTMEvents for testing."""
    acq = RTMSequence(
        time_plan={"interval": 1.0, "loops": 3},
        stage_positions=[(0, 0, 0)],
        channels=[{"config": "phase-contrast", "exposure": 50}],
        stim_channels=(
            (PowerChannel("phase-contrast", 100, power=10),) if stim else ()
        ),
        stim_frames=range(3) if stim else range(0),
        rtm_metadata=metadata or {},
    )
    return list(acq)


# ===================================================================
# Signature validation tests
# ===================================================================

class TestSignatureValidation:
    """validate_pipeline checks that subclass methods match their base class."""

    def test_all_good_components_pass(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup,
            tracker=GoodTracker(),
            fe=GoodFeatureExtractor(),
            stim=GoodStimulator(),
        )
        events = _make_events(stim=True)
        assert pipeline.validate_pipeline(events) is True

    def test_stim_missing_tracks_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(tmp_path_cleanup, stim=StimMissingTracks())
        events = _make_events(stim=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("tracks" in str(warning.message) for warning in w)
        assert any("StimMissingTracks" in str(warning.message) for warning in w)

    def test_segmentator_missing_image_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup, segmentator=SegMissingImage(),
        )
        events = _make_events()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("image" in str(warning.message) for warning in w)
        assert any("SegMissingImage" in str(warning.message) for warning in w)

    def test_tracker_missing_fov_state_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup, tracker=TrackerMissingFovState(),
        )
        events = _make_events()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("fov_state" in str(warning.message) for warning in w)

    def test_fe_missing_metadata_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup, fe=FEMissingMetadata(),
        )
        events = _make_events()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("metadata" in str(warning.message) for warning in w)

    def test_kwargs_always_passes(self, tmp_path_cleanup):
        """A method accepting **kwargs satisfies any base class signature."""
        pipeline = _make_pipeline(tmp_path_cleanup, stim=StimWithKwargs())
        events = _make_events(stim=True)
        assert pipeline.validate_pipeline(events) is True

    def test_extra_params_are_fine(self, tmp_path_cleanup):
        """Subclass may accept MORE params than the base class."""

        class StimExtra(StimWithPipeline):
            def get_stim_mask(
                self, label_images, metadata=None, img=None,
                tracks=None, extra_param=None,
            ):
                return np.zeros((10, 10), dtype=np.uint8), None

        pipeline = _make_pipeline(tmp_path_cleanup, stim=StimExtra())
        events = _make_events(stim=True)
        assert pipeline.validate_pipeline(events) is True


# ===================================================================
# Required metadata tests
# ===================================================================

class TestRequiredMetadata:
    """validate_pipeline checks that events carry required metadata."""

    # --- Stimulator metadata (only checked on stim events) ---

    def test_stim_metadata_missing_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(tmp_path_cleanup, stim=StimNeedsFraction())
        events = _make_events(stim=True)  # no stim_fraction in metadata

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("stim_fraction" in str(warning.message) for warning in w)

    def test_stim_metadata_present_passes(self, tmp_path_cleanup):
        pipeline = _make_pipeline(tmp_path_cleanup, stim=StimNeedsFraction())
        events = _make_events(stim=True, metadata={"stim_fraction": 0.1})
        assert pipeline.validate_pipeline(events) is True

    def test_stim_metadata_not_checked_on_non_stim_events(self, tmp_path_cleanup):
        """Non-stim events should pass even without stim-required metadata."""
        pipeline = _make_pipeline(tmp_path_cleanup, stim=StimNeedsFraction())
        events = _make_events(stim=False)  # no stim channels → no check
        assert pipeline.validate_pipeline(events) is True

    # --- Tracker metadata (checked on all events) ---

    def test_tracker_metadata_missing_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup, tracker=TrackerNeedsCondition(),
        )
        events = _make_events()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("condition" in str(warning.message) for warning in w)

    def test_tracker_metadata_present_passes(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup, tracker=TrackerNeedsCondition(),
        )
        events = _make_events(metadata={"condition": "control"})
        assert pipeline.validate_pipeline(events) is True

    # --- Segmentator metadata (checked on all events) ---

    def test_seg_metadata_missing_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup, segmentator=SegNeedsChannel(),
        )
        events = _make_events()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("seg_channel" in str(warning.message) for warning in w)

    def test_seg_metadata_present_passes(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup, segmentator=SegNeedsChannel(),
        )
        events = _make_events(metadata={"seg_channel": "DAPI"})
        assert pipeline.validate_pipeline(events) is True

    # --- Feature extractor metadata (checked on all events) ---

    def test_fe_metadata_missing_fails(self, tmp_path_cleanup):
        pipeline = _make_pipeline(tmp_path_cleanup, fe=FENeedsThreshold())
        events = _make_events()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        assert any("fe_threshold" in str(warning.message) for warning in w)

    def test_fe_metadata_present_passes(self, tmp_path_cleanup):
        pipeline = _make_pipeline(tmp_path_cleanup, fe=FENeedsThreshold())
        events = _make_events(metadata={"fe_threshold": 0.5})
        assert pipeline.validate_pipeline(events) is True

    # --- Multiple components with metadata requirements ---

    def test_multiple_components_all_missing(self, tmp_path_cleanup):
        """All components require metadata, none provided."""
        pipeline = _make_pipeline(
            tmp_path_cleanup,
            segmentator=SegNeedsChannel(),
            tracker=TrackerNeedsCondition(),
            fe=FENeedsThreshold(),
            stim=StimNeedsFraction(),
        )
        events = _make_events(stim=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        messages = " ".join(str(warning.message) for warning in w)
        assert "seg_channel" in messages
        assert "condition" in messages
        assert "fe_threshold" in messages
        assert "stim_fraction" in messages

    def test_multiple_components_all_provided(self, tmp_path_cleanup):
        """All components require metadata, all provided."""
        pipeline = _make_pipeline(
            tmp_path_cleanup,
            segmentator=SegNeedsChannel(),
            tracker=TrackerNeedsCondition(),
            fe=FENeedsThreshold(),
            stim=StimNeedsFraction(),
        )
        events = _make_events(
            stim=True,
            metadata={
                "seg_channel": "DAPI",
                "condition": "control",
                "fe_threshold": 0.5,
                "stim_fraction": 0.1,
            },
        )
        assert pipeline.validate_pipeline(events) is True


# ===================================================================
# Combined: signature + metadata
# ===================================================================

class TestCombinedValidation:
    """Both signature and metadata problems are reported together."""

    def test_bad_signature_and_missing_metadata(self, tmp_path_cleanup):
        pipeline = _make_pipeline(
            tmp_path_cleanup,
            stim=StimMissingTracks(),
            tracker=TrackerNeedsCondition(),
        )
        events = _make_events(stim=True)  # missing 'condition' metadata

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pipeline.validate_pipeline(events)

        assert result is False
        messages = " ".join(str(warning.message) for warning in w)
        # Signature issue
        assert "tracks" in messages
        # Metadata issue
        assert "condition" in messages

    def test_no_components_passes(self, tmp_path_cleanup):
        """Pipeline with only a segmentator (no tracker/fe/stim) passes."""
        pipeline = _make_pipeline(tmp_path_cleanup)
        events = _make_events()
        assert pipeline.validate_pipeline(events) is True
