"""Fixtures for the hardware-in-the-loop test suite.

Each fixture is session-scoped so the microscope is initialized only
once per pytest run, regardless of how many hardware tests fire. The
``--scope`` CLI option (defined in ``tests/conftest.py``) selects which
Pertzlab microscope class to instantiate.

Safety:
- ``safe_positions`` reads the stage XY at session start and returns
  positions as RELATIVE offsets so we never move into uncalibrated
  territory or hit stage limits.
- The Z stage is read for record-keeping only — never written.
- The stage is restored to its original XY at session end.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Per-scope channel mapping
# ---------------------------------------------------------------------------
# Channel groups and channel names per scope are inferred from each
# microscope's Micro-Manager cfg in ``pertzlab_mic_configs/``. Update
# these tables when adding a new scope or when a cfg's groups change.
#
# Each scope provides:
#   channel_group     — Micro-Manager ConfigGroup that hosts the channels
#   imaging_channel   — primary imaging channel (passed to segmentation)
#   optocheck_channel — second channel acquired on the last frame as a ref
#   stim_channel      — stimulation channel (driven by the DMD path)
# Exposures are intentionally short so the smoke test stays under a minute.
SCOPE_CHANNELS = {
    "moench": {
        "channel_group": "TTL_ERK",
        "imaging_channel": "mScarlet3",
        "imaging_exposure": 100.0,
        "optocheck_channel": "mCitrine",
        "optocheck_exposure": 100.0,
        "stim_channel": "CyanStim",
        "stim_exposure": 100.0,
        "stim_power": 5,
    },
    "niesen": {
        "channel_group": "WF_DMD",
        "imaging_channel": "Red",
        "imaging_exposure": 100.0,
        "optocheck_channel": "Green",
        "optocheck_exposure": 100.0,
        "stim_channel": "CyanStim",
        "stim_exposure": 100.0,
        "stim_power": 5,
    },
    "jungfrau": {
        "channel_group": "TTL_ERK",
        "imaging_channel": "mScarlet3",
        "imaging_exposure": 100.0,
        "optocheck_channel": "mCitrine",
        "optocheck_exposure": 100.0,
        "stim_channel": "CyanStim",
        "stim_exposure": 100.0,
        "stim_power": 5,
    },
}


# Per-scope DMD calibration profile used to instantiate ``DMD`` where
# the microscope class doesn't already do so in its constructor (only
# Jungfrau today — its DMD isn't yet wired into the cfg, so this will
# raise loudly when the SLM device is missing, surfacing the gap).
SCOPE_DMD_PROFILES: dict[str, dict] = {
    "jungfrau": {
        "channel_group": "TTL_ERK",
        "channel_config": "CyanStim",
        "device_name": "LED",
        "property_name": "Cyan_Level",
        "power": 5,
    },
}


# ---------------------------------------------------------------------------
# Scope selection
# ---------------------------------------------------------------------------


def _resolve_scope(config: pytest.Config) -> str | None:
    return config.getoption("--scope") or os.environ.get("FARO_SCOPE")


@pytest.fixture(scope="session")
def scope_name(request: pytest.FixtureRequest) -> str:
    """Return the active scope name, skipping if none is selected."""
    name = _resolve_scope(request.config)
    if name is None:
        pytest.skip("hardware test — pass --scope or set FARO_SCOPE")
    return name


@pytest.fixture(scope="session")
def scope_config(scope_name: str) -> dict:
    """Return the per-scope channel mapping."""
    return SCOPE_CHANNELS[scope_name]


# ---------------------------------------------------------------------------
# Synthetic DMD affine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_affine() -> np.ndarray:
    """Identity 3x3 affine in skimage convention.

    Hardware tests don't validate optical alignment — they only need
    the DMD code path to run end-to-end without raising. The identity
    matrix lets ``skimage.transform.warp`` resample the camera image
    into DMD space by direct coordinate copy (with cropping/padding
    when DMD and camera differ in size). No interactive calibration
    or precomputed ``.npy`` is needed on disk.
    """
    return np.eye(3)


# ---------------------------------------------------------------------------
# Microscope factory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def microscope(scope_name: str, synthetic_affine: np.ndarray):
    """Instantiate the selected Pertzlab microscope once per session.

    Loads the Micro-Manager cfg, applies the per-scope ROI when the
    class requests it, and injects a synthetic DMD affine matrix so
    stim code paths run without an interactive calibration step.

    For Moench and Niesen the constructor accepts an
    ``affine_calibration_matrix`` and creates the DMD inside
    ``init_scope()``. Jungfrau today has no DMD setup in its
    constructor — we instantiate one manually after init. If the
    Jungfrau cfg lacks an SLM device this will raise with a clear
    error pointing at the gap, which is the desired signal.
    """
    if scope_name == "moench":
        from faro.microscope.pertzlab.moench import Moench

        mic = Moench(affine_calibration_matrix=synthetic_affine)
    elif scope_name == "niesen":
        from faro.microscope.pertzlab.niesen import Niesen

        mic = Niesen(
            affine_calibration_matrix=synthetic_affine, fast_init=True
        )
    elif scope_name == "jungfrau":
        from faro.core.dmd import DMD
        from faro.microscope.pertzlab.jungfrau import Jungfrau

        mic = Jungfrau()
        mic.dmd = DMD(
            mic.mmc,
            SCOPE_DMD_PROFILES["jungfrau"],
            affine_matrix=synthetic_affine,
        )
        mic.dmd_needs_to_be_waken = False
    else:
        raise ValueError(f"unknown scope: {scope_name!r}")

    # Apply per-scope ROI when production code does so.
    if getattr(mic, "SET_ROI_REQUIRED", False) and hasattr(mic, "set_roi"):
        mic.set_roi()

    # Force a known camera binning so the test's zarr store shape matches
    # what the camera actually delivers. Binning is intentionally a
    # standalone cfg group (not part of the light-path channel presets),
    # so we can set it once here and every subsequent frame comes back at
    # the same y/x. 2x2 keeps the test frame small and fast.
    try:
        mic.mmc.setConfig("Binning", "2x2")
    except Exception:
        pass

    yield mic

    # Release all hardware handles so the Python process can exit.
    # Cleanup lives on the microscope class itself, not in the fixture,
    # so notebooks / scripts get the same behavior.
    try:
        mic.shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Safe (relative) stage positions
# ---------------------------------------------------------------------------


def _parse_offset_env(raw: str) -> list[tuple[float, float]]:
    """Parse ``"dx0,dy0;dx1,dy1;..."`` into a list of (dx, dy) tuples."""
    offsets: list[tuple[float, float]] = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        dx_str, dy_str = pair.split(",")
        offsets.append((float(dx_str), float(dy_str)))
    return offsets


@pytest.fixture(scope="session")
def safe_positions(microscope) -> list[dict]:
    """Generate FOV positions as RELATIVE offsets from the stage's
    current XY so the test never moves into uncalibrated territory or
    crashes the objective.

    Workflow:
        1. The user manually parks the stage on a sample area before
           starting the test run.
        2. This fixture snapshots ``(start_x, start_y)`` from the
           current XY stage position and ``start_z`` from the focus
           drive (read-only — Z is never moved).
        3. Three nearby FOVs are returned as ``(start + offset)``
           tuples. Default offsets are ``(0,0)``, ``(40,0)``, ``(0,40)``
           microns — well inside a single 40x camera FOV (~130 µm)
           so cells overlap between FOVs but no large stage motion
           ever happens.
        4. At session end the stage is restored to ``(start_x, start_y)``
           so an interrupted manual session can resume seamlessly.

    Override the offsets via ``FARO_HW_TEST_OFFSETS_UM`` formatted as
    ``"dx0,dy0;dx1,dy1;..."`` (in microns). Set to a single ``"0,0"``
    pair to restrict the test to a single position.
    """
    mmc = microscope.mmc
    start_x, start_y = mmc.getXYPosition()
    try:
        start_z = mmc.getPosition()
    except Exception:
        start_z = 0.0

    raw_offsets = os.environ.get("FARO_HW_TEST_OFFSETS_UM")
    if raw_offsets:
        offsets = _parse_offset_env(raw_offsets)
    else:
        offsets = [(0.0, 0.0), (40.0, 0.0), (0.0, 40.0)]

    positions = [
        {
            "x": start_x + dx,
            "y": start_y + dy,
            "z": start_z,
            "name": f"fov_{i}",
        }
        for i, (dx, dy) in enumerate(offsets)
    ]

    yield positions

    # Best-effort: return the stage to where the user left it. Swallow
    # exceptions so a teardown failure can't mask a test failure above.
    try:
        mmc.setXYPosition(start_x, start_y)
    except Exception:
        pass
