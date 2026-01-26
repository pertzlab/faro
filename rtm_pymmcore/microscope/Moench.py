import pymmcore_plus
import weakref

from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope
from rtm_pymmcore.controller import Controller, Analyzer
from rtm_pymmcore.dmd import DMD
from useq._mda_event import SLMImage
from pymmcore_plus.mda._engine import MDAEngine
from typing import Optional
from pymmcore_plus._logger import logger

from useq import MDAEvent
import os
import time
import threading
import locale
import logging
from pymmcore_plus.core._sequencing import SequencedEvent, iter_sequenced_events
from contextlib import suppress


os.environ["PYMM_PARALLEL_INIT"] = "0"


class KeepDMDAlive:
    def __init__(self, mmc):
        self.mmc = mmc
        self.thread = None
        self.last_wakeup = 0
        self.is_running = False

    def wakeup_dmd(self):
        self.mmc.setSLMExposure(self.mmc.getSLMDevice(), 200000.0)
        self.mmc.setSLMPixelsTo(self.mmc.getSLMDevice(), 255)
        self.mmc.displaySLMImage(self.mmc.getSLMDevice())

    def run(self):
        # Set locale to C/POSIX to ensure period as decimal separator
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            # If 'C' is not available, try 'en_US.UTF-8' or 'en_US'
            for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
                try:
                    locale.setlocale(locale.LC_NUMERIC, loc)
                    break
                except locale.Error:
                    continue

        self.is_running = True
        self.last_wakeup = 0
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_wakeup > 60:  # Wake up every minute
                self.wakeup_dmd()
                self.last_wakeup = current_time
            time.sleep(5)

    def stop(self):
        # Set locale to C/POSIX to ensure period as decimal separator
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            # If 'C' is not available, try 'en_US.UTF-8' or 'en_US'
            for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
                try:
                    locale.setlocale(locale.LC_NUMERIC, loc)
                    break
                except locale.Error:
                    continue

        self.is_running = False
        self.thread.join()
        time.sleep(5)
        self.mmc.setSLMExposure(self.mmc.getSLMDevice(), 100)
        self.mmc.displaySLMImage(self.mmc.getSLMDevice())


class Moench(AbstractMicroscope):
    MICROMANAGER_PATH = "C:\\micromanager_rtm_pymmcore\\Micro-Manager-2.0_api74"
    MICROMANAGER_CONFIG = "C:\\micromanager_rtm_pymmcore\\pertzlab_mic_configs\\micromanager\\Moench\\TiMoench_w_TTL_PrimeBSI.cfg"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = True
    DMD_NEEDS_TO_BE_WAKEN = True
    DMD_CHANNEL_GROUP = "TTL_ERK"
    DMD_CALIBRATION_PROFILE = {
        "channel_group": "TTL_ERK",
        "channel_config": "CyanStim",
        "device_name": "Spectra",
        "property_name": "Cyan_Level",
        "power": 10,
    }
    ROI_X = 0
    ROI_Y = 60
    ROI_WIDTH = 800
    ROI_HEIGHT = 800
    SET_ROI_REQUIRED = True

    def __init__(self, affine_calibration_matrix=None):
        super().__init__()

        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus(mm_path=self.MICROMANAGER_PATH)
        self.slm_dev = None
        self.slm_width = None
        self.slm_height = None

        self.affine_calibration_matrix = affine_calibration_matrix
        self.wakeup_dmd = None
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")
        self.register_engine()

        self.slm_dev = self.mmc.getSLMDevice()
        self.slm_width = self.mmc.getSLMWidth(self.slm_dev)
        self.slm_height = self.mmc.getSLMHeight(self.slm_dev)
        self.dmd = DMD(
            self.mmc,
            self.DMD_CALIBRATION_PROFILE,
            affine_matrix=self.affine_calibration_matrix,
        )
        self.wakeup_dmd = KeepDMDAlive(self.mmc)
        self.wakeup_dmd.run()

        self.image_height = self.mmc.getImageHeight()
        self.image_width = self.mmc.getImageWidth()

    def calibrate_dmd(
        self,
        verbous=False,
        n_points=15,
        radius=4,
        exposure=25,
        marker_style="x",
        calibration_points_DMD=None,
    ):
        self.disable_log_output()

        "Calibrate the DMD if it is not already calibrated." ""
        if self.dmd is not None and self.dmd.affine is None:
            self.wakeup_dmd.stop()
            self.dmd.calibrate(
                verbose=verbous,
                n_points=n_points,
                radius=radius,
                exposure=exposure,
                marker_style=marker_style,
                calibration_points_DMD=calibration_points_DMD,
            )
            self.wakeup_dmd.run()

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        self.mmc.setProperty("Core", "TimeoutMs", 9000)
        self.register_engine()
        self.disable_log_output()
        self.wakeup_dmd.stop()
        time.sleep(2)
        self.analyzer = Analyzer(self.pipeline)
        self.controller = Controller(
            self.analyzer,
            self.mmc,
            self.queue,
            use_autofocus_event=self.USE_AUTOFOCUS_EVENT,
            dmd=self.dmd,
            dmd_needs_to_be_waken=self.DMD_NEEDS_TO_BE_WAKEN,
        )
        self.controller.run(df_acquire)

    def set_roi(self):
        self.mmc.clearROI()
        self.mmc.setROI(self.ROI_X, self.ROI_Y, self.ROI_WIDTH, self.ROI_HEIGHT)

    def post_experiment(self):
        """Post-process the experiment."""
        self.wakeup_dmd.run()

    def register_engine(self, force: bool = False) -> None:
        """Create and register the microscope-specific MDA engine.

        This is idempotent unless `force=True`. It will attach a weakref to
        this microscope on the engine and register the engine on `self.mmc.mda`.
        """
        # If engine already exists and caller doesn't want to force, do nothing
        if hasattr(self, "engine") and self.engine is not None and not force:
            return

        # Create the engine and attach this microscope (weakref)
        self.engine = MoenchMDAEngine(self.mmc)
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

    def disable_log_output(self):
        pymmcore_plus.configure_logging(
            stderr_level="CRITICAL",
            file_level="CRITICAL",
        )
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
                for h in logger.handlers[:]:
                    logger.removeHandler(h)

        pymmcore_plus.configure_logging(stderr_level="WARNING")


class MoenchMDAEngine(MDAEngine):
    """Microscope-specific MDA engine for Moench.

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

    def _set_event_channel(self, event: MDAEvent, max_retry_attempts: int = 5) -> None:
        if (ch := event.channel) is None:
            return

        # comparison with _last_config is a fast/rough check ... which may miss subtle
        # differences if device properties have been individually set in the meantime.
        # could also compare to the system state, with:
        # data = self._mmc.getConfigData(ch.group, ch.config)
        # if self._mmc.getSystemStateCache().isConfigurationIncluded(data):
        #     ...
        if (ch.group, ch.config) != self.mmcore._last_config:  # noqa: SLF001
            # Try multiple times to set the configuration in case of transient failures.
            for attempt in range(1, max_retry_attempts + 1):
                try:
                    self.mmcore.setConfig(ch.group, ch.config)
                except Exception as e:
                    logger.warning(
                        "Failed to set channel (attempt %d/%d). %s",
                        attempt,
                        max_retry_attempts,
                        e,
                    )
                    print(
                        "Failed to set channel (attempt %d/%d). %s",
                        attempt,
                        max_retry_attempts,
                        e,
                    )
                    if attempt == max_retry_attempts:
                        logger.warning(
                            "Giving up after %d attempts to set channel.",
                            max_retry_attempts,
                        )
                    else:
                        time.sleep(0.1)
                else:
                    break

    def _set_event_xy_position(self, event: MDAEvent, max_retry_attempts=5) -> None:
        event_x, event_y = event.x_pos, event.y_pos
        # If neither coordinate is provided, do nothing.
        if event_x is None and event_y is None:
            return

        core = self.mmcore
        # skip if no XY stage device is found
        if not core.getXYStageDevice():
            logger.warning("No XY stage device found. Cannot set XY position.")
            return

        # Retrieve the last commanded XY position.
        last_x, last_y = core._last_xy_position.get(None) or (
            None,
            None,
        )  # noqa: SLF001
        if (
            not self.force_set_xy_position
            and (event_x is None or event_x == last_x)
            and (event_y is None or event_y == last_y)
        ):
            return

        if event_x is None or event_y is None:
            cur_x, cur_y = core.getXYPosition()
            event_x = cur_x if event_x is None else event_x
            event_y = cur_y if event_y is None else event_y

        for attempt in range(0, max_retry_attempts):
            try:
                core.setXYPosition(event_x, event_y)
                return
            except Exception as e:
                msg = str(e)
                if 'Wait for device "TIXYDrive" timed out' in msg:
                    if attempt == max_retry_attempts:
                        # all retries used, re-raise
                        raise
                    print(
                        f"[WARN] TIXYDrive wait timed out (attempt {attempt+1}/{max_retry_attempts+1}); "
                        f"retrying in {1} s..."
                    )
                    time.sleep(1)
                else:
                    # different error → don't hide it
                    logger.warning("Failed to set XY position. %s", e)
                    raise

    def setup_single_event(self, event: MDAEvent) -> None:
        """Setup hardware for a single (non-sequenced) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        if event.keep_shutter_open:
            ...

        max_retry_attempts = 5

        self._set_event_xy_position(event, max_retry_attempts=max_retry_attempts)

        if event.x_pos is not None or event.y_pos is not None:
            time.sleep(
                0.2
            )  # small delay to ensure XY stage has moved, as XY stage encore is broken on this microscope
        if event.z_pos is not None:
            self._set_event_z(event)
        if event.slm_image is not None:
            self._set_event_slm_image(event)

        self._set_event_channel(event, max_retry_attempts=max_retry_attempts)

        mmcore = self.mmcore
        if event.exposure is not None:
            try:
                mmcore.setExposure(event.exposure)
            except Exception as e:
                logger.warning("Failed to set exposure. %s", e)
        if event.properties is not None:
            for attempt in range(1, max_retry_attempts + 1):
                try:
                    for dev, prop, value in event.properties:
                        mmcore.setProperty(dev, prop, value)
                except Exception as e:
                    logger.warning("Failed to set properties. %s", e)
                    print(("Failed to set properties. %s", e))
                    if attempt == max_retry_attempts:
                        logger.warning(
                            "Giving up after %d attempts to set channel.",
                            max_retry_attempts,
                        )
                    else:
                        time.sleep(0.1)
                else:
                    break
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

    def _load_sequenced_event(
        self, event: SequencedEvent, max_retry_attempts: int = 0
    ) -> None:
        """Load a `SequencedEvent` into the core.

        `SequencedEvent` is a special pymmcore-plus specific subclass of
        `useq.MDAEvent`.
        """
        core = self.mmcore
        if event.exposure_sequence:
            cam_device = core.getCameraDevice()
            with suppress(RuntimeError):
                core.stopExposureSequence(cam_device)
            core.loadExposureSequence(cam_device, event.exposure_sequence)
        if event.x_sequence:  # y_sequence is implied and will be the same length
            stage = core.getXYStageDevice()
            with suppress(RuntimeError):
                core.stopXYStageSequence(stage)
            core.loadXYStageSequence(stage, event.x_sequence, event.y_sequence)
        if event.z_sequence:
            zstage = core.getFocusDevice()
            with suppress(RuntimeError):
                core.stopStageSequence(zstage)
            core.loadStageSequence(zstage, event.z_sequence)
        if event.slm_sequence:
            slm = core.getSLMDevice()
            with suppress(RuntimeError):
                core.stopSLMSequence(slm)
            core.loadSLMSequence(slm, event.slm_sequence)  # type: ignore[arg-type]
        if event.property_sequences:
            for (dev, prop), value_sequence in event.property_sequences.items():
                with suppress(RuntimeError):
                    core.stopPropertySequence(dev, prop)
                core.loadPropertySequence(dev, prop, value_sequence)

        # set all static properties, these won't change over the course of the sequence.
        if event.properties:
            for dev, prop, value in event.properties:
                for attempt in range(1, max_retry_attempts + 1):
                    try:
                        core.setProperty(dev, prop, value)
                    except Exception as e:
                        logger.warning(
                            "Failed to set property %s.%s (attempt %d/%d): %s",
                            dev,
                            prop,
                            attempt,
                            max_retry_attempts,
                            e,
                        )
                        if attempt == max_retry_attempts:
                            logger.warning(
                                "Giving up after %d attempts to set property %s.%s",
                                max_retry_attempts,
                                dev,
                                prop,
                            )
                        else:
                            time.sleep(0.1)
                    else:
                        break

    def setup_sequenced_event(
        self, event: SequencedEvent, max_retry_attempts: int = 5
    ) -> None:
        """Setup hardware for a sequenced (triggered) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        core = self.mmcore

        self._load_sequenced_event(event, max_retry_attempts=max_retry_attempts)

        # this is probably not necessary.  loadSequenceEvent will have already
        # set all the config properties individually/manually.  However, without
        # the call below, we won't be able to query `core.getCurrentConfig()`
        # not sure that's necessary; and this is here for tests to pass for now,
        # but this could be removed.
        self._set_event_channel(event, max_retry_attempts=max_retry_attempts)

        if event.slm_image:
            self._set_event_slm_image(event)

        # preparing a Sequence while another is running is dangerous.
        if core.isSequenceRunning():
            self._await_sequence_acquisition()
        core.prepareSequenceAcquisition(core.getCameraDevice())

        # start sequences or set non-sequenced values
        if event.x_sequence:
            core.startXYStageSequence(core.getXYStageDevice())
        else:
            self._set_event_xy_position(event)

        if event.z_sequence:
            core.startStageSequence(core.getFocusDevice())
        elif event.z_pos is not None:
            self._set_event_z(event)

        if event.exposure_sequence:
            core.startExposureSequence(core.getCameraDevice())
        elif event.exposure is not None:
            core.setExposure(event.exposure)

        if event.property_sequences:
            for dev, prop in event.property_sequences:
                core.startPropertySequence(dev, prop)
