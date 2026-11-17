"""Test helpers for hardware-in-the-loop tests of faro microscopes.

Lab packages (faro-pertzlab, faro-aoki, ...) keep their hardware tests next
to their microscope classes and reuse these pieces:

- ``faro.testing.plugin``: pytest plugin with the ``--scope`` option, the
  ``hardware`` marker and the generic fixtures. Load it from the lab's root
  ``conftest.py`` with ``pytest_plugins = ["faro.testing.plugin"]``.
- ``faro.testing.hardware``: assertions and helpers for the output of a
  hardware run (``assert_clean_run``, ``load_tracks_df``, ...).
"""
