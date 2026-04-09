"""Compatibility shim for useq-schema imports.

All FARO code that touches useq types goes through this module. When
useq-schema v2 lands and renames or moves symbols, only this file has
to change. See https://pymmcore-plus.github.io/useq-schema/v2-migration/.

Today (useq 0.9.x) ``SLMImage`` is exposed at the package top level,
but older snapshots only exported it from ``useq._mda_event``. We try
the public path first and fall back to the private one so older v1
pins continue to work.
"""

from __future__ import annotations

try:
    from useq import SLMImage
except ImportError:  # pragma: no cover — defensive fallback
    from useq._mda_event import SLMImage  # type: ignore[no-redef]

__all__ = ["SLMImage"]
