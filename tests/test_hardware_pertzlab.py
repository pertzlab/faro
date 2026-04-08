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
