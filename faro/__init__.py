"""FARO -- Feedback Adaptive Real-time Optogenetics."""

# Force pymmcore-plus to use the synchronous psygnal signal backend BEFORE
# any submodule import can pull in pymmcore_widgets, which sets
# PYMM_SIGNALS_BACKEND='qt' via os.environ.setdefault at its own import.
# napari-micromanager transitively imports pymmcore_widgets, so a notebook
# that did `import napari_micromanager` before any faro import would
# otherwise lock the backend to 'qt' -- routing core.mda.events.frameReady
# through Qt's queued delivery and silently starving the controller's
# pipeline whenever the main thread is busy. faro's async controller
# (RunHandle / run_experiment) needs the data path direct + synchronous on
# the engine thread, so we set this *explicitly* and *hard* (not setdefault):
# pymmcore_widgets's own setdefault wins whenever it imports first, so faro
# can only get its way by overriding outright. pymmcore-plus reads this env
# lazily (in pymmcore_plus._util.signals_backend(), at each signaler
# construction in CMMCorePlus / MDARunner), so this line still takes effect
# as long as it runs before any CMMCorePlus is instantiated or any
# mmc.mda is first accessed -- which `import faro` (or `from faro...`)
# guarantees, since it precedes the faro-side Moench() call that does so.
#
# Headless paths are unaffected: with no Qt around, pymmcore-plus would
# pick psygnal anyway. Setting this is a no-op there.
import os

os.environ["PYMM_SIGNALS_BACKEND"] = "psygnal"

# Default console output for the "faro" logger tree, so hardware and
# controller messages are visible without any host setup. Runs mirror
# the same records into log.txt in the experiment folder (see
# Controller._run_mda_with_events). propagate=False prevents double
# printing when a host configures the root logger; hosts that want
# custom handling can swap the handler on logging.getLogger("faro").
import logging

_faro_logger = logging.getLogger("faro")
if not _faro_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    _faro_logger.addHandler(_handler)
    _faro_logger.setLevel(logging.INFO)
    _faro_logger.propagate = False
