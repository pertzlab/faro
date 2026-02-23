"""Tests for validate_hardware().

Validates that RTMEvent channels exist on the microscope and that
exposure / device-property values are within the hardware's reported limits.
"""

from __future__ import annotations

import warnings

import pytest

from rtm_pymmcore.core.data_structures import Channel, RTMEvent
from rtm_pymmcore.core.utils import validate_hardware


# ---------------------------------------------------------------------------
# Fake CMMCorePlus — just enough API surface for validate_hardware
# ---------------------------------------------------------------------------

class FakeMMCore:
    """Minimal mock of CMMCorePlus for hardware validation tests."""

    def __init__(
        self,
        *,
        config_groups: dict[str, list[str]] | None = None,
        camera: str = "Camera",
        property_limits: dict[tuple[str, str], tuple[float, float]] | None = None,
    ):
        self._config_groups = config_groups or {
            "Channel": ["phase-contrast", "DAPI", "membrane"],
        }
        self._camera = camera
        # (device, property) → (lo, hi)
        self._property_limits = property_limits or {}

    def getAvailableConfigGroups(self):
        return list(self._config_groups.keys())

    def getAvailableConfigs(self, group):
        return self._config_groups.get(group, [])

    def getCameraDevice(self):
        return self._camera

    def hasPropertyLimits(self, device, prop):
        return (device, prop) in self._property_limits

    def getPropertyLowerLimit(self, device, prop):
        lims = self._property_limits.get((device, prop))
        return lims[0] if lims else 0.0

    def getPropertyUpperLimit(self, device, prop):
        lims = self._property_limits.get((device, prop))
        return lims[1] if lims else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(*, channels=None, stim_channels=None, n=3):
    """Return a list of RTMEvents with the given channels."""
    chs = channels or (Channel("phase-contrast", 50),)
    stim = stim_channels or ()
    return [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=tuple(chs),
            stim_channels=tuple(stim),
        )
        for t in range(n)
    ]


# ===================================================================
# Config existence checks
# ===================================================================

class TestChannelConfigExistence:

    def test_valid_config_passes(self):
        mmc = FakeMMCore()
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert validate_hardware(events, mmc) is True

    def test_unknown_config_fails(self):
        mmc = FakeMMCore()
        events = _make_events(channels=[Channel("GFP", 50)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("GFP" in str(warning.message) for warning in w)
        assert any("not found" in str(warning.message) for warning in w)

    def test_stim_channel_unknown_fails(self):
        mmc = FakeMMCore()
        events = _make_events(
            channels=[Channel("phase-contrast", 50)],
            stim_channels=[Channel("nonexistent-laser", 100)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("nonexistent-laser" in str(warning.message) for warning in w)

    def test_multiple_groups_searched(self):
        """Config can be in any group — not just 'Channel'."""
        mmc = FakeMMCore(config_groups={
            "Channel": ["phase-contrast"],
            "Laser": ["488nm", "561nm"],
        })
        events = _make_events(channels=[Channel("561nm", 50)])
        assert validate_hardware(events, mmc) is True

    def test_all_channels_checked(self):
        """All unique channel names across events are checked."""
        mmc = FakeMMCore()
        events = [
            RTMEvent(index={"t": 0, "p": 0}, channels=(Channel("phase-contrast", 50),)),
            RTMEvent(index={"t": 1, "p": 0}, channels=(Channel("DAPI", 30),)),
            RTMEvent(index={"t": 2, "p": 0}, channels=(Channel("MISSING", 50),)),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        messages = " ".join(str(x.message) for x in w)
        assert "MISSING" in messages
        # Only one warning (for MISSING), not for valid channels
        config_warnings = [x for x in w if "not found" in str(x.message)]
        assert len(config_warnings) == 1


# ===================================================================
# Exposure range checks
# ===================================================================

class TestExposureLimits:

    def test_exposure_within_range_passes(self):
        mmc = FakeMMCore(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert validate_hardware(events, mmc) is True

    def test_exposure_exceeds_max_fails(self):
        mmc = FakeMMCore(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 200)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("exceeds camera maximum" in str(x.message) for x in w)
        assert any("200" in str(x.message) for x in w)

    def test_exposure_below_min_fails(self):
        mmc = FakeMMCore(property_limits={("Camera", "Exposure"): (5.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 1)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("below camera minimum" in str(x.message) for x in w)

    def test_no_exposure_limits_skips_check(self):
        """When camera has no exposure limits, any value passes."""
        mmc = FakeMMCore()  # no property_limits
        events = _make_events(channels=[Channel("phase-contrast", 99999)])
        assert validate_hardware(events, mmc) is True

    def test_stim_channel_exposure_also_checked(self):
        mmc = FakeMMCore(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
        events = _make_events(
            channels=[Channel("phase-contrast", 50)],
            stim_channels=[Channel("phase-contrast", 500)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("500" in str(x.message) for x in w)

    def test_duplicate_exposures_not_repeated(self):
        """Same (name, exposure) across events should produce at most one warning."""
        mmc = FakeMMCore(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 200)], n=10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_hardware(events, mmc)

        exposure_warnings = [x for x in w if "exposure" in str(x.message).lower()]
        assert len(exposure_warnings) == 1


# ===================================================================
# Device property (power) limit checks
# ===================================================================

class TestDevicePropertyLimits:

    def test_power_within_range_passes(self):
        mmc = FakeMMCore(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        events = _make_events(
            channels=[Channel("phase-contrast", 50, power=50,
                              device_name="LED", property_name="Intensity")],
        )
        assert validate_hardware(events, mmc) is True

    def test_power_exceeds_max_fails(self):
        mmc = FakeMMCore(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        events = _make_events(
            channels=[Channel("phase-contrast", 50, power=150,
                              device_name="LED", property_name="Intensity")],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("exceeds device maximum" in str(x.message) for x in w)

    def test_power_below_min_fails(self):
        mmc = FakeMMCore(property_limits={("LED", "Intensity"): (10.0, 100.0)})
        events = _make_events(
            channels=[Channel("phase-contrast", 50, power=5,
                              device_name="LED", property_name="Intensity")],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("below device minimum" in str(x.message) for x in w)

    def test_no_device_limits_skips_check(self):
        """When device has no limits for the property, any value passes."""
        mmc = FakeMMCore()  # no property_limits
        events = _make_events(
            channels=[Channel("phase-contrast", 50, power=9999,
                              device_name="LED", property_name="Intensity")],
        )
        assert validate_hardware(events, mmc) is True

    def test_channel_without_power_skips_check(self):
        """Channels without power/device info skip the property check."""
        mmc = FakeMMCore(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert validate_hardware(events, mmc) is True

    def test_stim_channel_power_checked(self):
        mmc = FakeMMCore(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        events = _make_events(
            channels=[Channel("phase-contrast", 50)],
            stim_channels=[Channel("phase-contrast", 50, power=200,
                                   device_name="LED", property_name="Intensity")],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("200" in str(x.message) for x in w)


# ===================================================================
# Combined checks
# ===================================================================

class TestCombinedHardwareValidation:

    def test_all_good_passes(self):
        mmc = FakeMMCore(property_limits={
            ("Camera", "Exposure"): (0.0, 100.0),
            ("LED", "Intensity"): (0.0, 100.0),
        })
        events = _make_events(
            channels=[Channel("phase-contrast", 50, power=50,
                              device_name="LED", property_name="Intensity")],
        )
        assert validate_hardware(events, mmc) is True

    def test_multiple_problems_all_reported(self):
        """Bad config + bad exposure + bad power → three warnings."""
        mmc = FakeMMCore(property_limits={
            ("Camera", "Exposure"): (0.0, 100.0),
            ("LED", "Intensity"): (0.0, 50.0),
        })
        events = _make_events(
            channels=[Channel("MISSING-CHANNEL", 200, power=999,
                              device_name="LED", property_name="Intensity")],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        messages = " ".join(str(x.message) for x in w)
        assert "not found" in messages        # config check
        assert "exceeds camera maximum" in messages  # exposure check
        assert "exceeds device maximum" in messages  # power check

    def test_empty_events_passes(self):
        mmc = FakeMMCore()
        assert validate_hardware([], mmc) is True


# ===================================================================
# Microscope-level validation (mic.validate_events flow)
# ===================================================================

class TestAbstractMicroscopeValidateHardware:
    """AbstractMicroscope.validate_hardware is a no-op (returns True)."""

    def test_base_validate_hardware_returns_true(self):
        from rtm_pymmcore.microscope.base import AbstractMicroscope
        mic = AbstractMicroscope()
        events = _make_events(channels=[Channel("anything", 50)])
        assert mic.validate_hardware(events) is True

    def test_base_validate_events_without_pipeline(self):
        """With no pipeline set, validate_events still works (skips pipeline check)."""
        from rtm_pymmcore.microscope.base import AbstractMicroscope
        mic = AbstractMicroscope()
        events = _make_events(channels=[Channel("anything", 50)])
        assert mic.validate_events(events) is True


class TestPyMMCoreMicroscopeValidateHardware:
    """PyMMCoreMicroscope.validate_hardware delegates to utils.validate_hardware."""

    def test_no_mmc_returns_true(self):
        from rtm_pymmcore.microscope.pymmcore import PyMMCoreMicroscope
        mic = PyMMCoreMicroscope()
        events = _make_events(channels=[Channel("anything", 50)])
        assert mic.validate_hardware(events) is True

    def test_delegates_to_utils(self):
        from rtm_pymmcore.microscope.pymmcore import PyMMCoreMicroscope
        mic = PyMMCoreMicroscope()
        mic.mmc = FakeMMCore()
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert mic.validate_hardware(events) is True

    def test_delegates_detects_bad_channel(self):
        from rtm_pymmcore.microscope.pymmcore import PyMMCoreMicroscope
        mic = PyMMCoreMicroscope()
        mic.mmc = FakeMMCore()
        events = _make_events(channels=[Channel("MISSING", 50)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mic.validate_hardware(events)

        assert result is False
        assert any("MISSING" in str(x.message) for x in w)

    def test_validate_events_combines_pipeline_and_hardware(self):
        """mic.validate_events runs both pipeline and hardware checks."""
        import tempfile, shutil
        from rtm_pymmcore.microscope.pymmcore import PyMMCoreMicroscope
        from rtm_pymmcore.core.pipeline import ImageProcessingPipeline
        from rtm_pymmcore.core.data_structures import SegmentationMethod, RTMSequence

        # Minimal segmentator for pipeline
        from rtm_pymmcore.segmentation.base import Segmentator
        import numpy as np

        class DummySeg(Segmentator):
            def segment(self, image):
                return np.zeros_like(image)

        tmp = tempfile.mkdtemp()
        try:
            pipeline = ImageProcessingPipeline(
                storage_path=tmp,
                segmentators=[SegmentationMethod("labels", DummySeg(), 0, False)],
            )
            mic = PyMMCoreMicroscope()
            mic.mmc = FakeMMCore()
            mic.set_pipeline(pipeline)

            # Valid events — should pass
            events = _make_events(channels=[Channel("phase-contrast", 50)])
            assert mic.validate_events(events) is True

            # Bad channel — hardware check fails
            events_bad = _make_events(channels=[Channel("MISSING", 50)])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = mic.validate_events(events_bad)
            assert result is False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
