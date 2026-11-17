"""Top-level pytest configuration for the faro test suite.

Loads ``faro.testing.plugin``, which adds the ``--scope`` option and the
``hardware`` marker. faro has no hardware tests of its own: they live in the
lab packages (faro-pertzlab, faro-aoki, ...) next to their microscopes.
"""

from __future__ import annotations

import logging
import shutil
import tempfile

import pytest

pytest_plugins = ["faro.testing.plugin"]

# motile v0.4 emits a DEBUG record per variable with a ``%.3f`` format
# that blows up when pytest captures DEBUG (some numpy scalar subclasses
# do not implement __float__ cleanly in logging's lazy formatter).
# Silencing at INFO keeps the underlying solve intact.
logging.getLogger("motile").setLevel(logging.INFO)


@pytest.fixture()
def tmp_dir():
    """String tempdir cleaned up on teardown.

    Most faro storage APIs take ``str`` paths, so this fixture exists
    alongside pytest's built-in ``tmp_path`` (which yields ``pathlib.Path``).
    """
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)
