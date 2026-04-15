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

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from faro.core.data_structures import SegmentationMethod
from faro.segmentation.cellpose_v4 import CellposeV4
from faro.stimulation.base import StimWithImage
from tests.conftest import resolve_scope


PREFLIGHT_PATH = Path(__file__).parent / ".preflight.json"


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


@pytest.fixture(scope="session")
def scope_name(request: pytest.FixtureRequest) -> str:
    """Return the active scope name, skipping if none is selected."""
    name = resolve_scope(request.config)
    if name is None:
        pytest.skip("hardware test — pass --scope or set FARO_SCOPE")
    return name


@pytest.fixture(scope="session")
def preflight(scope_name: str) -> dict | None:
    """Return the preflight JSON produced by ``tests/hardware/setup.ipynb``.

    None if the user hasn't run the notebook yet. Tests that need it
    will skip with a clear pointer to the notebook.
    """
    if not PREFLIGHT_PATH.is_file():
        return None
    data = json.loads(PREFLIGHT_PATH.read_text())
    if data.get("scope") != scope_name:
        pytest.skip(
            f".preflight.json is for scope {data.get('scope')!r}, "
            f"but --scope selects {scope_name!r}. Re-run setup.ipynb."
        )
    return data


@pytest.fixture(scope="session")
def scope_config(scope_name: str, preflight: dict | None) -> dict:
    """Return the per-scope channel mapping.

    Preflight JSON (from ``setup.ipynb``) takes precedence over the
    hand-maintained ``SCOPE_CHANNELS`` table so the test run reflects
    whatever the user just verified interactively.
    """
    base = dict(SCOPE_CHANNELS[scope_name])
    if preflight is not None:
        for key in (
            "channel_group",
            "imaging_channel",
            "imaging_exposure",
            "optocheck_channel",
            "optocheck_exposure",
            "stim_channel",
            "stim_exposure",
            "stim_power",
        ):
            if key in preflight:
                base[key] = preflight[key]
    return base


@pytest.fixture(scope="session")
def brightness_thresholds(preflight: dict | None) -> dict:
    """Per-session camera brightness thresholds for the stim-fire tests.

    Calibrated by ``setup.ipynb`` from an actual DMD-off / DMD-on snap
    so the thresholds track whatever LED power, exposure, binning, and
    sample the user picked. Falls back to conservative defaults if no
    preflight exists; individual tests may choose to skip in that case.
    """
    if preflight is None:
        return {"bright_min_p99": 500.0, "bright_vs_dark_ratio": 5.0}
    return {
        "bright_min_p99": float(preflight["bright_min_p99"]),
        "bright_vs_dark_ratio": float(preflight["bright_vs_dark_ratio"]),
    }


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

    # Force a known camera binning BEFORE applying the ROI — most MM
    # configs reset the ROI on a binning change, so the order matters.
    # 2x2 keeps the test frame small and fast.
    try:
        mic.mmc.setConfig("Binning", "2x2")
    except Exception as e:
        print(f"[hardware fixture] could not set Binning=2x2 on {scope_name}: {e}")

    # Apply per-scope ROI when production code does so.
    if getattr(mic, "SET_ROI_REQUIRED", False) and hasattr(mic, "set_roi"):
        mic.set_roi()

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


# ---------------------------------------------------------------------------
# Shared post-run assertions
# ---------------------------------------------------------------------------


def assert_clean_run(controller, tmp_path: Path, *, expect_tracks: bool) -> None:
    """Post-run smoke checks shared by every hardware test.

    Fails loudly on any background-thread error (experiments log-and-
    continue, but a hardware test is meaningless if it swallows them)
    and confirms the run produced a napari-loadable OME-Zarr store.
    """
    if controller.background_errors:
        # Print full tracebacks to stdout so they appear in pytest -s
        # output but don't bloat the assertion message.
        for e in controller.background_errors:
            print(f"\n--- [{e.source}] {e.exc_type} ---\n{e.traceback}")
        summary = "\n".join(
            f"  [{e.source}] {e.exc_type}: {e.message}"
            for e in controller.background_errors
        )
        raise AssertionError(
            f"Background errors during acquisition:\n{summary}\n"
            f"(full tracebacks printed above)"
        )

    zarr_path = tmp_path / "acquisition.ome.zarr"
    assert zarr_path.is_dir(), f"OME-Zarr store not created at {zarr_path}"

    grp = zarr.open_group(str(zarr_path), mode="r")
    assert "ome" in grp.attrs, (
        "OME metadata missing on root group — store will not load in napari"
    )

    assert (tmp_path / "events.json").is_file(), "events.json not written"

    if expect_tracks:
        tracks_dir = tmp_path / "tracks"
        assert tracks_dir.is_dir(), "tracks/ folder not created"
        assert list(tracks_dir.glob("*.parquet")), "no track parquet files written"


# ---------------------------------------------------------------------------
# Shared OME-Zarr readout helpers
# ---------------------------------------------------------------------------


def open_stim_channel_array(zarr_path):
    """Locate the first stim readout channel in a hardware-test OME-Zarr.

    Writer appends stim channels after imaging channels and labels them
    ``stim_0``, ``stim_1``, ... Multi-position runs use the direct-write
    path (single ``(t,p,c,y,x)`` array at ``"0"``); the ``omero``
    channel list is a sibling of ``multiscales`` under ``attrs["ome"]``.
    Returns ``(array, stim_channel_index)``.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    ome = root.attrs["ome"]
    multiscales = ome["multiscales"][0]
    channels = ome["omero"]["channels"]
    stim_idx = next(
        i for i, c in enumerate(channels) if c["label"].startswith("stim_")
    )
    arr = root[multiscales["datasets"][0]["path"]]
    return arr, stim_idx


# ---------------------------------------------------------------------------
# Shared test stubs
# ---------------------------------------------------------------------------


class DelayedMaskStim(StimWithImage):
    """StimWithImage stub that sleeps before returning a full-image mask.

    Used by lag / timeout tests that need the mask request to go
    through the Analyzer's async queue (not the synchronous base-Stim
    path) so sleep time directly translates into pipeline lag. The
    ``metadata["img_shape"]`` key isn't populated on the storage-worker
    path, so we derive shape from the raw image.
    """

    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def get_stim_mask(self, metadata: dict, img):
        time.sleep(self.delay_s)
        h, w = img.shape[-2], img.shape[-1]
        return np.ones((h, w), dtype=np.uint8), None


# ---------------------------------------------------------------------------
# Shared cellpose segmentator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cellpose_segmentator() -> SegmentationMethod:
    """One cellpose model shared across hardware tests in a session.

    Cellpose weight loading takes ~1.5 s; a session-scoped model saves
    that cost on every subsequent test. Hardware tests that need a
    different ``min_size`` or weight set should build their own
    ``SegmentationMethod`` instead of using this fixture.
    """
    return SegmentationMethod(
        name="labels",
        segmentation_class=CellposeV4(min_size=50, gpu=True),
        use_channel=0,
        save_tracked=True,
    )


# ---------------------------------------------------------------------------
# Tracks parquet helpers
# ---------------------------------------------------------------------------


def load_tracks_df(tmp_path: Path) -> pd.DataFrame:
    """Concatenate all per-FOV tracks parquets into a single DataFrame.

    Returns an empty DataFrame if no parquet files exist; the caller
    decides whether that's an error.
    """
    parts = sorted((tmp_path / "tracks").glob("*_latest.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def assert_timestep_ordering(df: pd.DataFrame) -> None:
    """PR #6 / TracksDispenser contract: ``timestep`` equals ``fov_timestep``.

    Under concurrent pipeline workers a thread-order race would desync
    the MDA timestep from the per-FOV counter. The fix (TracksDispenser)
    serializes handoff in acquisition order, making the two columns
    match row-for-row.
    """
    if "fov_timestep" not in df.columns:
        return
    mismatched = df[df["timestep"] != df["fov_timestep"]]
    assert mismatched.empty, (
        f"{len(mismatched)} rows where timestep != fov_timestep — "
        f"TracksDispenser ordering broken. Sample:\n"
        f"{mismatched[['fov', 'timestep', 'fov_timestep']].head()}"
    )
