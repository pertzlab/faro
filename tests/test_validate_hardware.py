"""Tests for validate_hardware().

Validates that RTMEvent channels exist on the microscope and that
exposure / device-property values are within the hardware's reported limits.
"""

from __future__ import annotations

import warnings

import pytest

from faro.core.data_structures import Channel, PowerChannel, RTMEvent, wait
from faro.core.utils import validate_hardware

from tests.fake_mmc import build_validation_core as _core


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
        mmc = _core()
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert validate_hardware(events, mmc) is True

    def test_unknown_config_fails(self):
        mmc = _core()
        events = _make_events(channels=[Channel("GFP", 50)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("GFP" in str(warning.message) for warning in w)
        assert any("not found" in str(warning.message) for warning in w)

    def test_stim_channel_unknown_fails(self):
        mmc = _core()
        events = _make_events(
            channels=[Channel("phase-contrast", 50)],
            stim_channels=[Channel("nonexistent-laser", 100)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("nonexistent-laser" in str(warning.message) for warning in w)

    def test_wait_event_ignored(self):
        """A WaitEvent has no channels and must not trip config checks."""
        mmc = _core()
        events = _make_events(channels=[Channel("phase-contrast", 50)]) + [wait(5.0)]
        assert validate_hardware(events, mmc) is True

    def test_multiple_groups_searched(self):
        """Config can be in any group — not just 'Channel'."""
        mmc = _core(
            config_groups={
                "Channel": ["phase-contrast"],
                "Laser": ["488nm", "561nm"],
            }
        )
        events = _make_events(channels=[Channel("561nm", 50)])
        assert validate_hardware(events, mmc) is True

    def test_all_channels_checked(self):
        """All unique channel names across events are checked."""
        mmc = _core()
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
        mmc = _core(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert validate_hardware(events, mmc) is True

    def test_exposure_exceeds_max_fails(self):
        mmc = _core(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 200)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("exceeds camera maximum" in str(x.message) for x in w)
        assert any("200" in str(x.message) for x in w)

    def test_exposure_below_min_fails(self):
        mmc = _core(property_limits={("Camera", "Exposure"): (5.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 1)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc)

        assert result is False
        assert any("below camera minimum" in str(x.message) for x in w)

    def test_no_exposure_limits_skips_check(self):
        """When camera has no exposure limits, any value passes."""
        mmc = _core()  # no property_limits
        events = _make_events(channels=[Channel("phase-contrast", 99999)])
        assert validate_hardware(events, mmc) is True

    def test_stim_channel_exposure_also_checked(self):
        mmc = _core(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
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
        mmc = _core(property_limits={("Camera", "Exposure"): (0.0, 100.0)})
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
        mmc = _core(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        power_props = {"phase-contrast": ("LED", "Intensity")}
        events = _make_events(
            channels=[PowerChannel("phase-contrast", 50, power=50)],
        )
        assert validate_hardware(events, mmc, power_properties=power_props) is True

    def test_power_exceeds_max_fails(self):
        mmc = _core(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        power_props = {"phase-contrast": ("LED", "Intensity")}
        events = _make_events(
            channels=[PowerChannel("phase-contrast", 50, power=150)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc, power_properties=power_props)

        assert result is False
        assert any("exceeds device maximum" in str(x.message) for x in w)

    def test_power_below_min_fails(self):
        mmc = _core(property_limits={("LED", "Intensity"): (10.0, 100.0)})
        power_props = {"phase-contrast": ("LED", "Intensity")}
        events = _make_events(
            channels=[PowerChannel("phase-contrast", 50, power=5)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc, power_properties=power_props)

        assert result is False
        assert any("below device minimum" in str(x.message) for x in w)

    def test_no_device_limits_skips_check(self):
        """When device has no limits for the property, any value passes."""
        mmc = _core()  # no property_limits
        power_props = {"phase-contrast": ("LED", "Intensity")}
        events = _make_events(
            channels=[PowerChannel("phase-contrast", 50, power=9999)],
        )
        assert validate_hardware(events, mmc, power_properties=power_props) is True

    def test_channel_without_power_skips_check(self):
        """Channels without power skip the property check."""
        mmc = _core(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert validate_hardware(events, mmc) is True

    def test_stim_channel_power_checked(self):
        mmc = _core(property_limits={("LED", "Intensity"): (0.0, 100.0)})
        power_props = {"phase-contrast": ("LED", "Intensity")}
        events = _make_events(
            channels=[Channel("phase-contrast", 50)],
            stim_channels=[PowerChannel("phase-contrast", 50, power=200)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc, power_properties=power_props)

        assert result is False
        assert any("200" in str(x.message) for x in w)


# ===================================================================
# Combined checks
# ===================================================================


class TestCombinedHardwareValidation:

    def test_all_good_passes(self):
        mmc = _core(
            property_limits={
                ("Camera", "Exposure"): (0.0, 100.0),
                ("LED", "Intensity"): (0.0, 100.0),
            }
        )
        power_props = {"phase-contrast": ("LED", "Intensity")}
        events = _make_events(
            channels=[PowerChannel("phase-contrast", 50, power=50)],
        )
        assert validate_hardware(events, mmc, power_properties=power_props) is True

    def test_multiple_problems_all_reported(self):
        """Bad config + bad exposure + bad power → three warnings."""
        mmc = _core(
            property_limits={
                ("Camera", "Exposure"): (0.0, 100.0),
                ("LED", "Intensity"): (0.0, 50.0),
            }
        )
        power_props = {"MISSING-CHANNEL": ("LED", "Intensity")}
        events = _make_events(
            channels=[PowerChannel("MISSING-CHANNEL", 200, power=999)],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_hardware(events, mmc, power_properties=power_props)

        assert result is False
        messages = " ".join(str(x.message) for x in w)
        assert "not found" in messages  # config check
        assert "exceeds camera maximum" in messages  # exposure check
        assert "exceeds device maximum" in messages  # power check

    def test_empty_events_passes(self):
        mmc = _core()
        assert validate_hardware([], mmc) is True


# ===================================================================
# Microscope-level validation (mic.validate_events flow)
# ===================================================================


class TestAbstractMicroscopeValidateHardware:
    """AbstractMicroscope.validate_hardware is a no-op (returns True)."""

    def test_base_validate_hardware_returns_true(self):
        from faro.microscope.base import AbstractMicroscope

        mic = AbstractMicroscope()
        events = _make_events(channels=[Channel("anything", 50)])
        assert mic.validate_hardware(events) is True

    def test_base_validate_hardware_without_pipeline(self):
        """validate_hardware works standalone (no pipeline involved)."""
        from faro.microscope.base import AbstractMicroscope

        mic = AbstractMicroscope()
        events = _make_events(channels=[Channel("anything", 50)])
        assert mic.validate_hardware(events) is True


class TestPyMMCoreMicroscopeValidateHardware:
    """PyMMCoreMicroscope.validate_hardware delegates to utils.validate_hardware."""

    def test_no_mmc_returns_true(self):
        from faro.microscope.pymmcore import PyMMCoreMicroscope

        mic = PyMMCoreMicroscope()
        events = _make_events(channels=[Channel("anything", 50)])
        assert mic.validate_hardware(events) is True

    def test_delegates_to_utils(self):
        from faro.microscope.pymmcore import PyMMCoreMicroscope

        mic = PyMMCoreMicroscope()
        mic.mmc = _core()
        events = _make_events(channels=[Channel("phase-contrast", 50)])
        assert mic.validate_hardware(events) is True

    def test_delegates_detects_bad_channel(self):
        from faro.microscope.pymmcore import PyMMCoreMicroscope

        mic = PyMMCoreMicroscope()
        mic.mmc = _core()
        events = _make_events(channels=[Channel("MISSING", 50)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mic.validate_hardware(events)

        assert result is False
        assert any("MISSING" in str(x.message) for x in w)

    def test_pipeline_and_hardware_validation_separate(self):
        """pipeline.validate_pipeline + mic.validate_hardware work independently."""
        import tempfile, shutil
        from faro.microscope.pymmcore import PyMMCoreMicroscope
        from faro.core.pipeline import ImageProcessingPipeline
        from faro.core.data_structures import SegmentationMethod, RTMSequence

        # Minimal segmentator for pipeline
        from faro.segmentation.base import Segmentator
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
            mic.mmc = _core()

            # Valid events — both pass
            events = _make_events(channels=[Channel("phase-contrast", 50)])
            assert pipeline.validate_pipeline(events) is True
            assert mic.validate_hardware(events) is True

            # Bad channel — hardware check fails, pipeline still passes
            events_bad = _make_events(channels=[Channel("MISSING", 50)])
            assert pipeline.validate_pipeline(events_bad) is True
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = mic.validate_hardware(events_bad)
            assert result is False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
