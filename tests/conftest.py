"""Top-level pytest configuration for the faro test suite.

Hosts the ``--scope`` CLI option used by hardware-in-the-loop tests so
that the option is recognized regardless of which subdirectory of
``tests/`` is being collected. Hardware tests live under
``tests/hardware/`` and are gated as follows:

- ``pytest`` (no flag, no env var): hardware tests are collected but
  auto-skipped. CI-safe by default.
- ``pytest --scope moench|niesen|jungfrau``: hardware tests run
  against the selected Pertzlab microscope.
- ``FARO_SCOPE=moench pytest``: same effect via env var (useful for
  CI runners that pre-configure the target scope).
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--scope`` for selecting which Pertzlab microscope to drive."""
    parser.addoption(
        "--scope",
        action="store",
        default=None,
        choices=("moench", "niesen", "jungfrau"),
        help=(
            "Pertzlab microscope to run hardware tests against. "
            "If omitted, hardware-marked tests are skipped. "
            "FARO_SCOPE env var is honored as a fallback."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip ``@pytest.mark.hardware`` tests when no scope is selected."""
    scope = config.getoption("--scope") or os.environ.get("FARO_SCOPE")
    if scope:
        return
    skip_marker = pytest.mark.skip(
        reason="hardware test — pass --scope or set FARO_SCOPE to enable"
    )
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_marker)
