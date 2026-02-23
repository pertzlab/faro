import os

LOG_FILE_PATH = "C:\\Users\\Jungfrau\\AppData\\Local\\pymmcore-plus\\pymmcore-plus\\logs\\pymmcore-plus.log"
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

import pymmcore_plus
import time
import weakref
import logging
from typing import Optional
from useq import MDAEvent
from pymmcore_plus.mda._engine import MDAEngine
from pymmcore_plus._logger import logger

from rtm_pymmcore.microscope.pymmcore import PyMMCoreMicroscope
from rtm_pymmcore.core.controller import Controller, Analyzer


class Jungfrau(PyMMCoreMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0_api74"
    MICROMANAGER_CONFIG = "E:\\pertzlab_mic_configs\\micromanager\\Jungfrau\\TiFluoroJungfrau_w_TTL_DIGITALIO.cfg"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = True

    def __init__(self):

        super().__init__()
        pymmcore_plus.configure_logging(
            stderr_level="CRITICAL",
            file_level="CRITICAL",
            file_rotation=160,
            file_retention=1,
        )
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus(self.MICROMANAGER_PATH)
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")
        # Register the MDA engine early so other components using `mmc.mda`
        # will already have the correct engine registered.
        self.register_engine()

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        # Enable analyzer debug to show live storage/pipeline stats in notebook output
        self.analyzer = Analyzer(self.pipeline)
        # Ensure engine is registered (idempotent).
        self.register_engine()

        self.controller = Controller(
            self.analyzer, self.mmc, self.queue, self.USE_AUTOFOCUS_EVENT
        )
        self.controller.run(df_acquire)

    def post_experiment(self):
        """Post-process the experiment."""

    def register_engine(self, force: bool = False) -> None:
        """Create and register the microscope-specific MDA engine.

        This is idempotent unless `force=True`. It will attach a weakref to
        this microscope on the engine and register the engine on `self.mmc.mda`.
        """
        # If engine already exists and caller doesn't want to force, do nothing
        if hasattr(self, "engine") and self.engine is not None and not force:
            return

        # Create the engine and attach this microscope (weakref)
        self.engine = JungfrauMDAEngine(self.mmc)
        try:
            self.engine.attach_microscope(self)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to attach microscope to engine"
            )

        # Register it on the MDARunner so acquisitions use it
        try:
            self.mmc.mda.set_engine(self.engine)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to register MDA engine on mmc.mda"
            )


class JungfrauMDAEngine(MDAEngine):
    """Microscope-specific MDA engine for Jungfrau.

    Override `setup_single_event` to add pre/post hooks for per-microscope
    behavior while preserving the base MDAEngine functionality by calling
    `super().setup_single_event(event)`.
    """

    def __init__(
        self,
        mmc,
        *,
        use_hardware_sequencing: bool = True,
        restore_initial_state: Optional[bool] = None,
    ):
        super().__init__(
            mmc,
            use_hardware_sequencing=use_hardware_sequencing,
            restore_initial_state=restore_initial_state,
        )
        self._microscope_ref: Optional[weakref.ref] = None
        self._log = logging.getLogger(self.__class__.__name__)

    def attach_microscope(self, mic) -> None:
        """Attach the microscope instance (weakref) so engine can consult it."""
        self._microscope_ref = weakref.ref(mic)

    @property
    def microscope(self):
        return None if self._microscope_ref is None else self._microscope_ref()

    def setup_single_event(self, event: MDAEvent) -> None:
        """Setup hardware for a single (non-sequenced) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        if event.keep_shutter_open:
            ...

        self._set_event_xy_position(event)
        if event.x_pos is not None or event.y_pos is not None:
            time.sleep(
                0.2
            )  # small delay to ensure XY stage has moved, as XY stage encore is broken on this microscope
        if event.z_pos is not None:
            self._set_event_z(event)
        if event.slm_image is not None:
            self._set_event_slm_image(event)

        self._set_event_channel(event)

        mmcore = self.mmcore
        if event.exposure is not None:
            try:
                mmcore.setExposure(event.exposure)
            except Exception as e:
                logger.warning("Failed to set exposure. %s", e)
        if event.properties is not None:
            try:
                for dev, prop, value in event.properties:
                    mmcore.setProperty(dev, prop, value)
            except Exception as e:
                logger.warning("Failed to set properties. %s", e)
        if (
            # (if autoshutter wasn't set at the beginning of the sequence
            # then it never matters...)
            self._autoshutter_was_set
            # if we want to leave the shutter open after this event, and autoshutter
            # is currently enabled...
            and event.keep_shutter_open
            and mmcore.getAutoShutter()
        ):
            # we have to disable autoshutter and open the shutter
            mmcore.setAutoShutter(False)
            mmcore.setShutterOpen(True)
