"""Find microscope classes by name.

faro ships no lab microscopes. A lab package (e.g. faro-pertzlab) registers
its classes in the ``faro.microscopes`` entry-point group of its
``pyproject.toml``::

    [project.entry-points."faro.microscopes"]
    moench = "faro_pertzlab.moench:Moench"

After ``uv sync``, ``faro-napari moench`` and the ``--scope`` option of the
hardware tests find the class by that name. An installed package sees a new
or renamed entry point only after it is installed again (``uv sync``; a
running Jupyter kernel also needs a restart).
"""

from __future__ import annotations

import logging
from importlib import import_module
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "faro.microscopes"

# Also listed as entry points in faro's pyproject.toml. Kept here so the
# names resolve when faro runs from a source tree that is not installed.
BUILTIN_SCOPES = {
    "demo": "faro.microscope.demo:MMDemo",
}


def available_scopes() -> dict[str, str]:
    """Return ``{name: "module:Class"}`` of all registered microscopes."""
    scopes = dict(BUILTIN_SCOPES)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        known = scopes.get(ep.name)
        if known is not None and known != ep.value:
            logger.warning(
                "Microscope name %r is registered twice (%s and %s); using %s.",
                ep.name,
                known,
                ep.value,
                ep.value,
            )
        scopes[ep.name] = ep.value
    return scopes


def load_scope_class(scope: str) -> type:
    """Return the microscope class for a registered name or ``"module:Class"``.

    Raises ``LookupError`` for an unknown name, with the registered names in
    the message.
    """
    target = available_scopes().get(scope, scope)
    if ":" not in target:
        names = ", ".join(sorted(available_scopes())) or "none"
        raise LookupError(
            f"Unknown microscope {scope!r}. Registered: {names}. Install the "
            "lab package that provides it, or pass 'module.path:ClassName'."
        )
    module_path, _, class_name = target.partition(":")
    return getattr(import_module(module_path), class_name)
