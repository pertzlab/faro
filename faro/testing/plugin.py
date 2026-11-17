"""pytest plugin for hardware-in-the-loop tests.

Load it from the root ``conftest.py`` of a lab package::

    pytest_plugins = ["faro.testing.plugin"]

Hardware tests carry ``@pytest.mark.hardware`` and are gated as follows:

- ``pytest`` (no flag, no env var): hardware tests are collected but
  skipped. CI-safe by default.
- ``pytest --scope <name>``: hardware tests run against the microscope
  registered under ``<name>`` (see ``faro.microscope.registry``).
- ``FARO_SCOPE=<name> pytest``: same, through the environment.

pytest also puts directory names into a test's keywords, so every test in
a directory named ``hardware`` is gated the same way, marked or not.

Fixtures: ``scope_name``, ``microscope`` (builds the registered class with no
arguments; override it in the lab's conftest when a scope needs arguments),
``synthetic_affine``, ``safe_positions`` and ``cellpose_segmentator``.

Safety of ``safe_positions``: it reads the stage XY at session start and
returns positions as RELATIVE offsets, so a test never moves into
uncalibrated territory or hits stage limits. Z is read, never written. The
stage goes back to its start XY at session end.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from faro.testing.hardware import parse_offset_env


def resolve_scope(config: pytest.Config) -> str | None:
    """Return the active scope, honoring ``--scope`` and ``FARO_SCOPE``."""
    return config.getoption("--scope") or os.environ.get("FARO_SCOPE")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--scope`` for selecting which microscope to drive."""
    parser.addoption(
        "--scope",
        action="store",
        default=None,
        help=(
            "Registered microscope name (faro.microscopes entry point) to run "
            "hardware tests against. If omitted, hardware-marked tests are "
            "skipped. The FARO_SCOPE env var is honored as a fallback."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "hardware: requires a real microscope (selected with --scope or FARO_SCOPE). "
        "Skipped by default.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.hardware`` tests when no scope is selected."""
    if resolve_scope(config):
        return
    skip_marker = pytest.mark.skip(
        reason="hardware test — pass --scope or set FARO_SCOPE to enable"
    )
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def scope_name(request: pytest.FixtureRequest) -> str:
    """Return the active scope name, skipping if none is selected."""
    name = resolve_scope(request.config)
    if name is None:
        pytest.skip("hardware test — pass --scope or set FARO_SCOPE")
    return name


@pytest.fixture(scope="session")
def synthetic_affine() -> np.ndarray:
    """Identity 3x3 affine in skimage convention.

    Hardware tests don't validate optical alignment — they only need
    the DMD code path to run end-to-end without raising. The identity
    matrix lets ``skimage.transform.warp`` resample the camera image
    into DMD space by direct coordinate copy (with cropping/padding
    when DMD and camera differ in size).
    """
    return np.eye(3)


@pytest.fixture(scope="session")
def microscope(scope_name: str):
    """Build the selected microscope once per session.

    Uses the class registered under ``scope_name``, called without
    arguments. A lab conftest overrides this fixture when its scopes need
    arguments (e.g. a synthetic DMD affine).
    """
    from faro.microscope.registry import load_scope_class

    mic = load_scope_class(scope_name)()
    if getattr(mic, "SET_ROI_REQUIRED", False) and hasattr(mic, "set_roi"):
        mic.set_roi()
    yield mic
    try:
        mic.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="session")
def safe_positions(microscope) -> list[dict]:
    """FOV positions as RELATIVE offsets from the stage's current XY.

    Workflow:
        1. The user parks the stage on a sample area before the run.
        2. This fixture snapshots ``(start_x, start_y)`` from the XY stage
           and ``start_z`` from the focus drive (read-only).
        3. Nearby FOVs are returned as ``start + offset``. The default
           offsets are ``(0,0)``, ``(40,0)``, ``(0,40)`` microns, inside a
           single 40x camera FOV (~130 µm), so no large stage motion happens.
        4. At session end the stage goes back to ``(start_x, start_y)``.

    Override the offsets with ``FARO_HW_TEST_OFFSETS_UM`` as
    ``"dx0,dy0;dx1,dy1;..."`` (microns). A single ``"0,0"`` pair restricts
    the test to one position.
    """
    mmc = microscope.mmc
    start_x, start_y = mmc.getXYPosition()
    try:
        start_z = mmc.getPosition()
    except Exception:
        start_z = 0.0

    raw_offsets = os.environ.get("FARO_HW_TEST_OFFSETS_UM")
    if raw_offsets:
        offsets = parse_offset_env(raw_offsets)
    else:
        offsets = [(0.0, 0.0), (40.0, 0.0), (0.0, 40.0)]

    positions = [
        {"x": start_x + dx, "y": start_y + dy, "z": start_z, "name": f"fov_{i}"}
        for i, (dx, dy) in enumerate(offsets)
    ]

    yield positions

    # Best effort: a teardown failure must not mask a test failure above.
    try:
        mmc.setXYPosition(start_x, start_y)
    except Exception:
        pass


@pytest.fixture(scope="session")
def cellpose_segmentator():
    """One cellpose model shared across the hardware tests of a session.

    Loading the weights takes ~1.5 s. Tests that need another ``min_size``
    or weight set build their own ``SegmentationMethod``. Cellpose is
    imported lazily, so the plugin loads without the cellpose extra.
    """
    from faro.core.data_structures import SegmentationMethod
    from faro.segmentation.cellpose_v4 import CellposeV4

    return SegmentationMethod(
        name="labels",
        segmentation_class=CellposeV4(min_size=50, gpu=True),
        use_channel=0,
        save_tracked=True,
    )
