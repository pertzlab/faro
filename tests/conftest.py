"""Shared test fixtures/helpers for the FARO test suite."""

from __future__ import annotations


class FakeDMD:
    """Minimal stand-in for an SLM/DMD device.

    Real ``faro.core.dmd.DMD`` wraps a micro-manager device and requires
    calibration. Tests that need to exercise the controller's stim-event
    branch (``if self._mic.dmd:``) can attach this to a microscope
    instead. The ``affine_transform`` is identity so test assertions can
    compare the delivered SLM data against the mask produced by the
    stimulator directly.
    """

    name = "FakeDMD"

    def affine_transform(self, img):
        return img
