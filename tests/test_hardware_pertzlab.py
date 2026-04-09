"""Tests for auto-detection of power properties from Micro-Manager configs.

These tests are specific to the Pertzlab microscope setups (Spectra / LedDMD
light sources with *_Level properties and DA TTL LED state devices).
"""

from __future__ import annotations

import pytest

from tests.test_validate_hardware import FakeMMCore
from faro.core.utils import detect_power_properties


# ===================================================================
# Auto-detection of power properties
# ===================================================================

class TestDetectPowerProperties:
    """detect_power_properties infers channel→power mappings from config."""

    def test_detects_spectra_cyan(self):
        """CyanStim config activating Cyan LED → (Spectra, Cyan_Level)."""
        mmc = FakeMMCore(
            config_groups={"TTL_ERK": ["CyanStim", "miRFP", "BF"]},
            devices={"Spectra": ["Cyan_Level", "Red_Level", "Green_Level"]},
            config_data={
                ("TTL_ERK", "CyanStim"): [
                    ("DA TTL LED", "Label", "Cyan"),
                    ("Wheel-C", "Label", "480/40"),
                ],
                ("TTL_ERK", "miRFP"): [
                    ("DA TTL LED", "Label", "Red"),
                    ("Wheel-C", "Label", "640/30"),
                ],
                ("TTL_ERK", "BF"): [
                    ("DA TTL LED", "Label", "OFF"),
                    ("DA TTL Bright", "Label", "On"),
                ],
            },
        )

        result = detect_power_properties(mmc)
        assert result["CyanStim"] == ("Spectra", "Cyan_Level")
        assert result["miRFP"] == ("Spectra", "Red_Level")
        assert "BF" not in result  # OFF doesn't match any color

    def test_greenyellow_matches_green_level(self):
        """GreenYellow LED label matches Green_Level via prefix."""
        mmc = FakeMMCore(
            config_groups={"Channel": ["mScarlet3"]},
            devices={"Spectra": ["Green_Level"]},
            config_data={
                ("Channel", "mScarlet3"): [
                    ("DA TTL LED", "Label", "GreenYellow"),
                ],
            },
        )
        result = detect_power_properties(mmc)
        assert result["mScarlet3"] == ("Spectra", "Green_Level")

    def test_scans_specific_group(self):
        """When group is specified, only that group is scanned."""
        mmc = FakeMMCore(
            config_groups={
                "TTL_ERK": ["CyanStim"],
                "Other": ["SomeConfig"],
            },
            devices={"Spectra": ["Cyan_Level"]},
            config_data={
                ("TTL_ERK", "CyanStim"): [("DA TTL LED", "Label", "Cyan")],
                ("Other", "SomeConfig"): [("DA TTL LED", "Label", "Cyan")],
            },
        )
        result = detect_power_properties(mmc, group="TTL_ERK")
        assert "CyanStim" in result
        assert "SomeConfig" not in result

    def test_no_level_devices_returns_empty(self):
        """No devices with *_Level properties → empty result."""
        mmc = FakeMMCore(
            config_groups={"Channel": ["GFP"]},
            devices={"Camera": ["Exposure", "Binning"]},
            config_data={("Channel", "GFP"): [("Filter", "Label", "GFP")]},
        )
        assert detect_power_properties(mmc) == {}

    def test_works_with_leddmd_device(self):
        """Works with LedDMD (Niesen) instead of Spectra."""
        mmc = FakeMMCore(
            config_groups={"WF_DMD": ["CyanStim"]},
            devices={"LedDMD": ["Cyan_Level", "Red_Level"]},
            config_data={
                ("WF_DMD", "CyanStim"): [("DA TTL LED", "Label", "Cyan")],
            },
        )
        result = detect_power_properties(mmc)
        assert result["CyanStim"] == ("LedDMD", "Cyan_Level")

    def test_microscope_auto_detect_merges_with_manual(self):
        """PyMMCoreMicroscope.get_power_properties merges detected + manual."""
        from faro.microscope.pymmcore import PyMMCoreMicroscope

        mmc = FakeMMCore(
            config_groups={"TTL_ERK": ["CyanStim", "miRFP"]},
            devices={"Spectra": ["Cyan_Level", "Red_Level"]},
            config_data={
                ("TTL_ERK", "CyanStim"): [("DA TTL LED", "Label", "Cyan")],
                ("TTL_ERK", "miRFP"): [("DA TTL LED", "Label", "Red")],
            },
        )

        mic = PyMMCoreMicroscope()
        mic.mmc = mmc
        # Manual override for CyanStim (should take priority)
        mic.POWER_PROPERTIES = {"CyanStim": ("CustomDev", "CustomProp")}
        mic.detect_power_properties()

        merged = mic.get_power_properties()
        # Manual override wins
        assert merged["CyanStim"] == ("CustomDev", "CustomProp")
        # Auto-detected still present
        assert merged["miRFP"] == ("Spectra", "Red_Level")


# ===================================================================
# SKIP_WAIT_DEVICES — per-microscope waitForDevice skip list
# ===================================================================


class _FakeWaitMMCore:
    """Minimal mmcore surface for _wait_for_system_excluding_xy tests."""

    def __init__(self, devices: list[str], xy_stage: str = ""):
        self._devices = devices
        self._xy_stage = xy_stage
        self.wait_calls: list[str] = []

    def getLoadedDevices(self):
        return self._devices

    def getXYStageDevice(self):
        return self._xy_stage

    def waitForDevice(self, dev: str):
        self.wait_calls.append(dev)

    def getXYPosition(self):
        return (0.0, 0.0)


class _FakeMoench:
    """Weakref-able stand-in for Moench carrying SKIP_WAIT_DEVICES."""

    def __init__(self, skip: tuple[str, ...]):
        self.SKIP_WAIT_DEVICES = skip


class TestSkipWaitDevices:
    """MoenchMDAEngine honors the microscope's SKIP_WAIT_DEVICES tuple.

    Regression guard against the 5 s-per-event Mosaic3 stuck-Busy wait
    (TODO.md #1): devices listed here must never hit waitForDevice().
    """

    def _make_engine(self, mmc, mic):
        import weakref

        from faro.microscope.pertzlab.moench import MoenchMDAEngine

        # Bypass MDAEngine.__init__ — the base class wants a real
        # CMMCorePlus, but this test only exercises the skip-filter
        # logic which reads self.mmcore and self.microscope.
        engine = MoenchMDAEngine.__new__(MoenchMDAEngine)
        engine._mmcore_ref = weakref.ref(mmc)
        engine._microscope_ref = weakref.ref(mic)
        return engine

    def test_skip_devices_bypass_wait(self):
        from useq import MDAEvent

        mmc = _FakeWaitMMCore(
            devices=["Core", "Camera", "Shutter", "Mosaic3", "XYStage"],
            xy_stage="XYStage",
        )
        mic = _FakeMoench(skip=("Mosaic3",))
        engine = self._make_engine(mmc, mic)

        engine._wait_for_system_excluding_xy(MDAEvent())

        assert "Mosaic3" not in mmc.wait_calls, "Mosaic3 should be skipped"
        assert "XYStage" not in mmc.wait_calls, "xy_stage handled separately"
        assert "Core" not in mmc.wait_calls, "Core is always skipped"
        assert "Camera" in mmc.wait_calls
        assert "Shutter" in mmc.wait_calls

    def test_missing_attribute_is_noop(self):
        """Microscopes without SKIP_WAIT_DEVICES fall through to the default."""
        from useq import MDAEvent

        mmc = _FakeWaitMMCore(
            devices=["Core", "Camera", "Mosaic3"],
            xy_stage="",
        )

        class _BareMic:
            pass

        engine = self._make_engine(mmc, _BareMic())
        engine._wait_for_system_excluding_xy(MDAEvent())

        # Without SKIP_WAIT_DEVICES, Mosaic3 is waited on as before.
        assert "Mosaic3" in mmc.wait_calls
        assert "Camera" in mmc.wait_calls
