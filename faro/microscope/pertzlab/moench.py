import pymmcore_plus
import weakref

import numpy as np

from faro.microscope.pymmcore import PyMMCoreMicroscope
from faro.core.data_structures import ImgType
from faro.core.dmd import DMD
from faro.core.qt import drain_posted_qt_events, pump_qt_events
from faro.core._useq_compat import SLMImage
from pymmcore_plus.mda._engine import MDAEngine
from typing import Optional

from useq import MDAEvent
import os
import time
import threading
import locale
import logging
from pymmcore_plus.core._sequencing import SequencedEvent, iter_sequenced_events
from contextlib import nullcontext, suppress


logger = logging.getLogger(__name__)

os.environ["PYMM_PARALLEL_INIT"] = "0"


def _set_c_numeric_locale():
    """Set locale to C/POSIX to ensure period as decimal separator."""
    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
    except locale.Error:
        for loc in ["en_US.UTF-8", "en_US", "English_United States.1252"]:
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
                break
            except locale.Error:
                continue


class MoenchCMMCorePlus(pymmcore_plus.CMMCorePlus):
    """CMMCorePlus for the Moench (Nikon Ti): confirm stage moves and cube changes.

    On the Nikon Ti the TIXYDrive/TIZDrive ``Busy()`` flag stays true for
    several seconds after every move, because Micro-Manager is never told the
    move finished, and the Mosaic3 DMD's ``Busy()`` is stuck the same way. So
    ``waitForDevice``/``waitForSystem`` wait ~5 s per move for nothing. The
    move itself finishes in under a second and the stage's reported position is
    always current, so this core waits for the XY and focus stages by reading
    their position until it reaches the commanded target, and does not wait on
    :attr:`SKIP_BUSY_DEVICES` at all.

    ``set(Relative)(XY/Z)Position`` stores the commanded target and marks that
    stage as "moving". ``waitForDevice``/``waitForSystem`` then wait for a
    moving stage by reading its position, and clear the mark once it arrives. A
    stage that was not commanded to move returns at once, because it is not
    moving and its stuck ``Busy()`` is never read. Any other device uses the
    normal wait.

    The same adapter silently skips a filter-cube (turret) change when its
    cached position already equals the target, so a genuinely needed change can
    be dropped and a frame acquired through the wrong cube. ``setConfig`` /
    ``setStateLabel`` / ``setState`` read the turret back after the change and,
    on a mismatch, force a real move by going to a neighbour position and back;
    the success path is a single read with no extra rotation, and the forced
    move fires only on a detected miss (capped at
    :attr:`FILTER_VERIFY_MAX_CORRECTIONS`).

    Sharing this core with napari-micromanager gives its interactive moves and
    channel changes the same handling, and the base engine's plain
    ``waitForSystem()`` / ``setConfig()`` calls keep :class:`MoenchMDAEngine`
    correct. napari-micromanager adopts the first ``CMMCorePlus`` ever
    constructed as its singleton, so it picks up this core automatically as
    long as :class:`Moench` is created before any napari-micromanager widget
    constructs a core of its own.
    """

    #: Devices with a stuck Busy() and no position to check. Never wait on them.
    SKIP_BUSY_DEVICES: frozenset = frozenset({"Mosaic3"})
    #: How close (µm) counts as "arrived", and how long (s) to wait before giving up.
    XY_TOLERANCE_UM: float = 1.0
    Z_TOLERANCE_UM: float = 0.5
    POSITION_CONFIRM_MAX_S: float = 5.0
    POSITION_CONFIRM_POLL_S: float = 0.05

    #: Filter-turret device to verify after a channel/state change (None = off),
    #: the property its preset sets, how many forced moves to try, and whether a
    #: persistent miss raises (default: log and continue).
    FILTER_VERIFY_DEVICE: str | None = "TIFilterBlock1"
    FILTER_VERIFY_PROPERTY: str = "Label"
    FILTER_VERIFY_MAX_CORRECTIONS: int = 3
    FILTER_VERIFY_RAISE_ON_FAILURE: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._xy_targets = {}  # device label -> (x, y)
        self._z_targets = {}  # device label -> z
        self._pending_moves = set()  # stages commanded to move, not yet arrived
        self._verifying_filter = False  # guards the verify's own turret moves
        self._dmd_live_hook = None  # set via set_dmd_live_hook
        self._last_dmd_arm = 0.0
        self._dmd_stale_connected = False

    # --- DMD live-pattern self-healing ------------------------------------
    # The Mosaic3 holds a pattern for at most its ExposureTime (200 s max)
    # and then parks the mirrors, which blacks out everything downstream of
    # the DMD. Interactive captures bypass the MDA engine (napari's snap and
    # live view call the core directly), so the core re-displays the DMD's
    # live-view pattern on demand before them. During an MDA the engine owns
    # the DMD (it injects a pattern into every event) and this must stay out
    # of the way.

    #: Re-arm the pattern when the last display is older than this. The
    #: Mosaic3 parks (goes dark) once its ExposureTime elapses after a
    #: display, camera triggers do NOT re-arm it, and the hold exposure is
    #: 200 s (Moench.DMD_HOLD_EXPOSURE_MS, datasheet max), so 150 s re-arms
    #: with margin. Displays are kept RARE on purpose: a display racing an
    #: open camera frame lights only part of that frame (measured on the
    #: scope), so per-capture re-displays cause flicker.
    DMD_ARM_STALE_S: float = 150.0
    #: Pause after re-arming so the mirror commit settles before the capture.
    DMD_REFRESH_SETTLE_S: float = 0.05

    def set_dmd_live_hook(self, hook) -> None:
        """Install the callable that re-displays the DMD live pattern.

        Pass ``dmd.display_livemode`` (or None to disable). The pattern is
        re-armed before a snap when stale, before every live-view start, and
        on the first capture after an MDA (the run leaves its own pattern,
        possibly a stim mask, on the mirrors).
        """
        self._dmd_live_hook = hook
        if hook is not None and not self._dmd_stale_connected:
            self._dmd_stale_connected = True
            self.mda.events.sequenceFinished.connect(self._mark_dmd_stale)

    def _mark_dmd_stale(self, *_) -> None:
        self._last_dmd_arm = 0.0

    def _ensure_dmd_pattern(self, force: bool = False) -> None:
        hook = self._dmd_live_hook
        if hook is None or self.mda.is_running():
            return
        now = time.monotonic()
        if not force and now - self._last_dmd_arm < self.DMD_ARM_STALE_S:
            return
        try:
            hook()
        except Exception as e:
            logger.warning("DMD live-pattern refresh failed: %s", e)
            return
        self._last_dmd_arm = now
        time.sleep(self.DMD_REFRESH_SETTLE_S)

    def snapImage(self) -> None:  # noqa: N802
        self._ensure_dmd_pattern()
        super().snapImage()

    def startContinuousSequenceAcquisition(self, *args) -> None:  # noqa: N802
        # always re-arm here: no frame is open yet, so the display is free
        self._ensure_dmd_pattern(force=True)
        super().startContinuousSequenceAcquisition(*args)

    def run_mda(
        self,
        events,
        *,
        output=None,
        block: bool = False,
        dimension_overrides=None,
    ) -> threading.Thread:
        """Run the MDA on a thread that drains its posted Qt events before exiting.

        Mirrors ``CMMCorePlus.run_mda`` except that the acquisition thread
        calls :func:`~faro.core.qt.drain_posted_qt_events` after the run, so
        no Qt events are left for Windows thread shutdown (see the helper's
        docstring for the deadlock this prevents). faro's Controller goes
        through ``PyMMCoreMicroscope.run_mda``, which drains as well; this
        override covers MDAs started directly on the core, e.g. from the
        napari-micromanager MDA widget.
        """
        if self.mda.is_running():
            raise ValueError(
                "Cannot start an MDA while the previous MDA is still running."
            )

        def _run_and_drain() -> None:
            try:
                self.mda.run(
                    events, output=output, dimension_overrides=dimension_overrides
                )
            finally:
                drain_posted_qt_events()

        th = threading.Thread(target=_run_and_drain)
        th.start()
        if block:
            th.join()
        return th

    def _is_managed_stage(self, label: str) -> bool:
        return bool(label) and label in (
            self.getXYStageDevice(),
            self.getFocusDevice(),
        )

    # Record the commanded target (both the (x, y) and (label, x, y) forms).
    def setXYPosition(self, *args) -> None:  # noqa: N802
        if len(args) == 2:
            label, x, y = "", args[0], args[1]
        elif len(args) == 3:
            label, x, y = args
        else:
            return super().setXYPosition(*args)
        dev = label or self.getXYStageDevice()
        if dev:
            self._xy_targets[dev] = (float(x), float(y))
            self._pending_moves.add(dev)
        return super().setXYPosition(*args)

    def setRelativeXYPosition(self, *args) -> None:  # noqa: N802
        dev = None
        try:
            if len(args) == 2:
                label, dx, dy = "", args[0], args[1]
            elif len(args) == 3:
                label, dx, dy = args
            else:
                raise ValueError
            dev = label or self.getXYStageDevice()
            cur = self.getXYPosition(dev) if label else self.getXYPosition()
            if dev:
                self._xy_targets[dev] = (cur[0] + float(dx), cur[1] + float(dy))
                self._pending_moves.add(dev)
        except Exception:
            if dev:
                self._pending_moves.discard(dev)  # target unknown -> don't wait
        return super().setRelativeXYPosition(*args)

    def setZPosition(self, val) -> None:  # noqa: N802
        dev = self.getFocusDevice()
        if dev:
            self._z_targets[dev] = float(val)
            self._pending_moves.add(dev)
        return super().setZPosition(val)

    def setPosition(self, *args) -> None:  # noqa: N802
        if len(args) == 1:
            label, z = "", args[0]
        elif len(args) == 2:
            label, z = args
        else:
            return super().setPosition(*args)
        dev = label or self.getFocusDevice()
        if dev:
            self._z_targets[dev] = float(z)
            self._pending_moves.add(dev)
        return super().setPosition(*args)

    def setRelativePosition(self, *args) -> None:  # noqa: N802
        dev = None
        try:
            if len(args) == 1:
                label, dz = "", args[0]
            elif len(args) == 2:
                label, dz = args
            else:
                raise ValueError
            dev = label or self.getFocusDevice()
            cur = self.getPosition(dev) if label else self.getPosition()
            if dev:
                self._z_targets[dev] = cur + float(dz)
                self._pending_moves.add(dev)
        except Exception:
            if dev:
                self._pending_moves.discard(dev)
        return super().setRelativePosition(*args)

    # Wait for a moving stage by reading its position.
    def waitForDevice(self, label: str) -> None:  # noqa: N802
        if label in self.SKIP_BUSY_DEVICES:
            return  # stuck Busy() and no position to check -> skip
        if self._is_managed_stage(label):
            if label not in self._pending_moves:
                return  # not commanded to move, so nothing to wait for
            self._pending_moves.discard(label)
            if label in self._xy_targets:
                return self._confirm_xy(label, self._xy_targets[label])
            if label in self._z_targets:
                return self._confirm_z(label, self._z_targets[label])
            return
        return self._base_wait_for_device(label)

    def _base_wait_for_device(self, label: str) -> None:
        super().waitForDevice(label)

    def waitForSystem(self) -> None:  # noqa: N802
        # The base waitForSystem() calls C++ waitForDevice directly, which
        # skips this class's waitForDevice override; loop over the devices here
        # so each one goes through the position check and skip logic above.
        for dev in self.getLoadedDevices():
            if dev == "Core":
                continue
            try:
                self.waitForDevice(dev)
            except RuntimeError as e:
                if "timed out" in str(e):
                    logger.warning(
                        "waitForDevice(%s) timed out, continuing: %s", dev, e
                    )
                else:
                    raise

    def _confirm_xy(self, dev, target) -> None:
        use_default = dev == self.getXYStageDevice()
        deadline = time.perf_counter() + self.POSITION_CONFIRM_MAX_S
        while time.perf_counter() < deadline:
            # A newer command supersedes the captured target, so an interactive
            # burst of moves (clicking through the position list) waits for
            # where the stage is actually headed instead of timing out on a
            # stale target.
            target = self._xy_targets.get(dev, target)
            tx, ty = target
            try:
                x, y = self.getXYPosition() if use_default else self.getXYPosition(dev)
            except Exception:
                return  # can't read position -> best-effort, don't hang
            if (
                abs(x - tx) < self.XY_TOLERANCE_UM
                and abs(y - ty) < self.XY_TOLERANCE_UM
            ):
                return
            time.sleep(self.POSITION_CONFIRM_POLL_S)
        logger.warning(
            "%s not within %.3g um of target %s after %.1f s",
            dev,
            self.XY_TOLERANCE_UM,
            target,
            self.POSITION_CONFIRM_MAX_S,
        )

    def _pfs_owns_focus(self) -> bool:
        """True when continuous focus (PFS) is driving the focus stage.

        While engaged, the PFS holds TIZDrive at whatever position keeps the
        sample in focus, so a commanded Z target is unreachable and waiting
        for it can only time out. With the unpatched Nikon adapter this flag
        is frozen at its config-load value; a stale True then skips a Z
        confirmation that would have been valid, which costs the check but
        never blocks a move. Best-effort: False when the read fails.
        """
        try:
            return bool(self.isContinuousFocusEnabled())
        except Exception:
            return False

    def _confirm_z(self, dev, target) -> None:
        deadline = time.perf_counter() + self.POSITION_CONFIRM_MAX_S
        while time.perf_counter() < deadline:
            if self._pfs_owns_focus():
                logger.debug(
                    "Skipping Z confirmation for %s: PFS is engaged and owns focus",
                    dev,
                )
                return
            target = self._z_targets.get(dev, target)
            try:
                z = self.getPosition(dev)
            except Exception:
                return
            if abs(z - target) < self.Z_TOLERANCE_UM:
                return
            time.sleep(self.POSITION_CONFIRM_POLL_S)
        logger.warning(
            "%s not within %.3g um of target %.3f after %.1f s",
            dev,
            self.Z_TOLERANCE_UM,
            target,
            self.POSITION_CONFIRM_MAX_S,
        )

    # Verify the filter turret actually reached the requested cube after a
    # channel/state change (the ``_verifying_filter`` guard keeps the verify's
    # own turret moves from triggering another verify).
    def setConfig(self, groupName, configName) -> None:  # noqa: N802, N803
        super().setConfig(groupName, configName)
        if not self._verifying_filter:
            target = self._filter_target_for_config(groupName, configName)
            if target is not None:
                self._verify_filter_reached(target)

    def setStateLabel(self, stateDeviceLabel, stateLabel) -> None:  # noqa: N802, N803
        super().setStateLabel(stateDeviceLabel, stateLabel)
        if not self._verifying_filter and stateDeviceLabel == self.FILTER_VERIFY_DEVICE:
            try:
                target = self.getStateFromLabel(stateDeviceLabel, stateLabel)
            except Exception:
                return
            self._verify_filter_reached(target)

    def setState(self, stateDeviceLabel, state) -> None:  # noqa: N802, N803
        super().setState(stateDeviceLabel, state)
        if not self._verifying_filter and stateDeviceLabel == self.FILTER_VERIFY_DEVICE:
            self._verify_filter_reached(int(state))

    def _filter_target_for_config(self, group, config):
        """State the filter device should reach for this preset, or None.

        None when filter verification is off, the device is not loaded, or the
        preset does not drive the filter device.
        """
        device = self.FILTER_VERIFY_DEVICE
        if not device or device not in self.getLoadedDevices():
            return None
        try:
            cfg = self.getConfigData(group, config)
        except Exception:
            return None
        for i in range(cfg.size()):
            s = cfg.getSetting(i)
            if (
                s.getDeviceLabel() == device
                and s.getPropertyName() == self.FILTER_VERIFY_PROPERTY
            ):
                try:
                    return self.getStateFromLabel(device, s.getPropertyValue())
                except Exception:
                    return None
        return None

    def _verify_filter_reached(self, target_state) -> None:
        """Confirm the Nikon Ti cube turret reached ``target_state``; on a
        detected mismatch, force a physical move and re-check.

        Why: the closed NikonTI adapter decides whether to move the turret by
        comparing the request against an internal, callback-fed position cache,
        and silently skips the move when they are equal ("Already at position;
        not moving", no exception, no error log). A missed position callback
        desyncs that cache, so a needed cube change can be dropped and the frame
        acquired through the wrong cube. Unlike XY/Z, the turret has no
        independent re-read or safety timeout in the adapter, so one is added
        here.

        Strategy (cheap by default; an extra rotation only on a miss):
          1. read the turret back; if it equals the target, return (the common
             case, one fast read, no extra movement);
          2. on mismatch, force a real move by first going to a neighbour
             position (which breaks the ``target == cache`` equality the adapter
             uses to suppress the move) and then to the target, re-checking each
             time, up to :attr:`FILTER_VERIFY_MAX_CORRECTIONS`;
          3. if it still will not land, log loudly (and, when
             :attr:`FILTER_VERIFY_RAISE_ON_FAILURE`, raise) so a long run
             surfaces the failure instead of silently collecting wrong-cube data.

        The read-back goes through the same cache the adapter compares against,
        so it cannot catch the rarer case where the cache wrongly equals the
        target. Forcing the move unconditionally would catch that case too,
        but it doubles turret wear on every change, which this rig's ~15 s
        imaging cadence cannot afford.
        """
        device = self.FILTER_VERIFY_DEVICE
        if target_state is None or not device:
            return
        max_corrections = int(self.FILTER_VERIFY_MAX_CORRECTIONS)

        failed_state = None
        self._verifying_filter = True
        try:
            if device not in self.getLoadedDevices():
                return
            n_states = self.getNumberOfStates(device)

            def _settled_state():
                # The filter turret's Busy() is reliable, so the plain wait
                # works and respects the configured FilterBlock Delay.
                try:
                    self.waitForDevice(device)
                except RuntimeError:
                    pass
                return self.getState(device)

            if _settled_state() == target_state:
                return  # fast path: turret is where we asked, no extra movement

            neighbour = (target_state + 1) % n_states
            if neighbour != target_state:  # guard a 1-position device
                for attempt in range(1, max_corrections + 1):
                    logger.warning(
                        "Filter turret missed state %d; forcing move (%d/%d).",
                        target_state,
                        attempt,
                        max_corrections,
                    )
                    print(
                        f"[WARN] Filter turret missed state {target_state}; "
                        f"forcing move ({attempt}/{max_corrections})."
                    )
                    # Go to a different cube first so target != cached position;
                    # this defeats the adapter's "Already at position" skip.
                    self.setState(device, neighbour)
                    try:
                        self.waitForDevice(device)
                    except RuntimeError:
                        pass
                    self.setState(device, target_state)
                    if _settled_state() == target_state:
                        logger.info(
                            "Filter turret recovered to state %d after %d "
                            "attempt(s).",
                            target_state,
                            attempt,
                        )
                        return

            failed_state = _settled_state()
        except Exception as e:  # never let verification crash an acquisition
            logger.warning("Filter-turret verify errored (ignored). %s", e)
            return
        finally:
            self._verifying_filter = False

        # Persistent mismatch: surface loudly so a long run can't silently
        # collect wrong-cube data.
        msg = (
            f"Filter turret FAILED to reach state {target_state} after "
            f"{max_corrections} forced moves (stuck at state {failed_state}); "
            f"frames may be acquired through the WRONG cube."
        )
        logger.error(msg)
        print(f"[ERROR] {msg}")
        if self.FILTER_VERIFY_RAISE_ON_FAILURE:
            raise RuntimeError(msg)


class KeepDMDAlive:
    #: Refresh cadence during live view. Must stay well under the 200 s hold
    #: exposure: the Mosaic3 parks (dark) once ExposureTime elapses after a
    #: display and camera triggers do not re-arm it, so long live sessions
    #: need a periodic re-display. Keep it long: a display racing an open
    #: frame lights only part of that frame, so every refresh risks one
    #: dimmed live frame.
    REFRESH_INTERVAL_S: float = 60.0

    def __init__(self, mmc, dmd):
        self.mmc = mmc
        self.dmd = dmd
        self.thread: threading.Thread | None = None
        self.last_wakeup = 0.0
        # daemon=True so interpreter shutdown doesn't block on this
        # thread holding COM3 (zombie python.exe on next session).
        self._stop_event = threading.Event()

    def wakeup_dmd(self):
        # Re-display the DMD's current live-view pattern (all-on by default,
        # or e.g. a checkerboard set for a focus check) so it survives the
        # periodic refresh instead of being forced back to all-on.
        self.dmd.display_livemode()

    def run(self, *_):
        """Start the refresh thread; no-op if it is already running.

        Connected to ``continuousSequenceAcquisitionStarted``, which fires
        again every time live view re-arms (napari does so on config and
        exposure changes), so repeated calls must not spawn extra threads.
        ``*_`` absorbs the signal's camera-label payload.
        """
        _set_c_numeric_locale()
        if self.thread is not None and self.thread.is_alive():
            return
        self._stop_event.clear()
        # the core re-arms the pattern right before the live start, so the
        # first periodic refresh can wait a full interval
        self.last_wakeup = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            current_time = time.time()
            # never touch the DMD while an MDA runs, whatever the signal
            # ordering: the engine drives it per event and a refresh could
            # overwrite a stim mask between upload and trigger
            if (
                current_time - self.last_wakeup > self.REFRESH_INTERVAL_S
                and not self.mmc.mda.is_running()
            ):
                self.wakeup_dmd()
                self.last_wakeup = current_time
            # Event.wait lets stop() break out immediately instead of
            # eating teardown time.
            if self._stop_event.wait(timeout=1):
                return

    def stop(self, *_):
        """Stop the refresh thread; no-op if not running.

        Connected to ``sequenceAcquisitionStopped``, which also fires when the
        MDA stops a live stream that was never running. The SLM is left
        holding its current pattern (the hold exposure keeps it up for
        200 s), so snaps right after a live stop still see light; parking it
        here made the first captures after every live stop dark.
        ``*_`` absorbs the signal's camera-label payload.
        """
        _set_c_numeric_locale()
        if self.thread is None:
            return
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        self.thread = None


class Moench(PyMMCoreMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0_api75"
    MICROMANAGER_CONFIG = (
        "C:\\faro\\pertzlab_mic_configs\\micromanager\\Moench\\TiMoench_UV.cfg"
    )
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = True
    DMD_NEEDS_TO_BE_WAKEN = True
    DMD_CHANNEL_GROUP = "TTL_ERK"

    # --- Mosaic3 DMD hold / settle ---
    # The Mosaic3 "SLM exposure" is the Andor ``ExposureTime`` feature: after a
    # ``displaySLMImage`` ("Expose") the micromirrors hold the displayed pattern
    # for exactly that long and then *park*. The Mosaic3 has no indefinite
    # "Mirror On" hold mode, so to keep the pattern on the mirrors for a whole
    # frame we set a long software exposure that outlasts any camera/LED window.
    # Light is gated in *time* by the camera-triggered LED (light path
    # LED -> DMD -> sample), not by the DMD, so holding the pattern between
    # frames delivers no extra dose and the stim dose is unchanged. Without this
    # a stale short ``ExposureTime`` left by another path (KeepDMDAlive.stop()
    # -> 100 ms, calibration -> 25 ms) parks the mirrors mid-frame and the tail
    # of any longer frame comes back dark.
    #
    # ``DMD_HOLD_EXPOSURE_MS`` is forced onto the SLM for every displayed pattern
    # (stim mask or all-on) by ``MoenchMDAEngine._set_event_slm_image``.
    # 200000 ms = 200 s is the Mosaic3 datasheet max and matches
    # ``DMD.display_livemode``. Set to None/0 to disable the override.
    DMD_HOLD_EXPOSURE_MS: float = 200000.0
    # Pause between committing the pattern (displaySLMImage / "Expose") and
    # snapping, so the mirror commit settles before the camera opens and the
    # camera-triggered LED fires. None/0 disables the wait.
    DMD_SETTLE_MS: float = 50.0
    # Manual config->(device, property) power mappings. These must be declared
    # explicitly: this config selects LED lines via numeric NIDAQ TTL states
    # (the preset stores e.g. State "16"; the color "GreenYellow" only lives in
    # the port0 state labels), so there's no reliable way to infer them. A
    # PowerChannel whose config is missing here raises in resolve_power()
    # instead of silently dropping the requested power.
    #
    # Derived from TiMoench.cfg: each preset sets NIDAQDO-Dev1/port0 State, and
    # the port's state labels give the color -> Spectra <Color>_Level:
    #   State  1 Violet      -> Violet_Level
    #   State  2 Blue        -> Blue_Level
    #   State  4 Cyan        -> Cyan_Level
    #   State  8 Teal        -> Teal_Level
    #   State 16 GreenYellow -> Green_Level
    #   State 32 Red         -> Red_Level
    POWER_PROPERTIES = {
        "CyanStim": ("LED", "Cyan_Level"),  # state 4
        "VioletStim": ("LED", "Violet_Level"),  # state 1, UV damage stim
        "mScarlet3": ("LED", "Green_Level"),  # state 16
        "miRFP": ("LED", "Red_Level"),  # state 32
        "mCitrine": ("LED", "Teal_Level"),  # state 8
        "mRuby2": ("LED", "Green_Level"),  # state 16
        "mTurquoise": ("LED", "Blue_Level"),  # state 2
        "mNeongreen": ("LED", "Cyan_Level"),  # state 4
    }
    BINNING = "2x2"
    # ROI applied as-is after binning, no centering recomputation.
    # Defaults below are for Prime BSI under 2x2 binning (1024 binned px wide).
    ROI_X = 0
    ROI_Y = 30
    ROI_WIDTH = 1024
    ROI_HEIGHT = 792
    SET_ROI_REQUIRED = True

    # ``Status`` values of the PFS device that mean "the PFS is driving Z".
    # Everything else ('Within range of focus search', 'Out of focus search
    # range', ...) means it is not engaged. See read_pfs_engaged().
    PFS_ENGAGED_STATUSES: tuple[str, ...] = ("Locked in focus", "Focusing")

    # ``Status`` values that indicate the PFS genuinely CANNOT hold focus:
    # a real fault when the run relies on the PFS. 'Out of focus search
    # range' / 'Within range of focus search' are deliberately NOT here:
    # those are normal when the PFS is simply switched off. A loss of lock is
    # detected separately, as a transition out of PFS_ENGAGED_STATUSES while
    # the PFS was engaged at the start of the run (see MoenchMDAEngine).
    #
    # Only the live ``Status`` read exposes these fault codes; the boolean
    # enabled flag cannot report *why* the PFS is not holding.
    PFS_FAILING_STATUSES: tuple[str, ...] = (
        "Focus lock failed",
        "Dichroic mirror not inserted",
        "Unsupported objective lens",
    )
    # If True, the engine warns when the PFS reports a fault at the start of a
    # run and, for runs that rely on the PFS to hold focus (PFS engaged and no
    # Z commanded), when it loses lock mid-acquisition. Monitoring is two-tier:
    # a cheap boolean probe (read_pfs_engaged_fast, no reload) runs on every
    # event, and the ~60 ms authoritative Status read (a non-destructive
    # AF-device reload) runs only when the probe disagrees with the expected
    # state or when PFS_HEALTH_PERIOD_S has elapsed since the last one. Set
    # False to disable all PFS health monitoring; the engine re-reads this
    # flag on every event, so it can be flipped mid-run.
    MONITOR_PFS_HEALTH: bool = True
    # Seconds between periodic authoritative Status reads while monitoring.
    # This is the safety net that keeps lost-lock detection working on an
    # UNPATCHED adapter DLL, where the cheap probe reads a frozen value and
    # never flags a transition. Time-based on purpose: it is immune to the
    # event order, so staggered schedules with interleaved timepoints pay the
    # same bounded cost as batch-ordered ones. None/0 disables the periodic
    # read; transitions seen by the cheap probe still trigger one.
    PFS_HEALTH_PERIOD_S: float = 300.0

    # Filter-turret verification (the "Already at position" cube-skip fix)
    # lives on MoenchCMMCorePlus, so it covers both MDA channel changes and
    # napari's interactive ones. Tune it via the FILTER_VERIFY_* class
    # attributes there.

    def __init__(self, affine_calibration_matrix=None, uncropped=False):
        """Load the Micro-Manager config and bring up the scope.

        Args:
            affine_calibration_matrix: DMD-to-camera affine from a previous
                ``calibrate_dmd()`` run, or None to start uncalibrated.
            uncropped: when True, skip the ROI crop and image the full
                sensor (sets ``SET_ROI_REQUIRED = False`` on this instance).
        """
        super().__init__()

        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = MoenchCMMCorePlus(mm_path=self.MICROMANAGER_PATH)
        # Only one thread at a time may reload the PFS device or read/write the
        # PFS. Reloading (unload + load + init) destroys and rebuilds the
        # device's C++ object, so if another thread reads or writes the PFS at
        # the same moment (e.g. the MDA runner during a Z move while a PFS
        # health read is reloading) it touches freed memory and the process
        # crashes. This lock keeps those apart.
        self._pfs_lock = threading.RLock()
        # faro's pipeline (and its per-event PFS handling) runs on the engine
        # thread. That only works with the "psygnal" MDA signal backend; the
        # "qt" backend instead hands callbacks to the main thread, which both
        # drops frames and lets a PFS reload run at the same time as PFS access
        # on another thread. Fail loudly if the backend is "qt"; this call also
        # builds the runner now, locking in the "psygnal" backend faro set at
        # import.
        self._check_signal_backend()
        self.slm_dev = None
        self.slm_width = None
        self.slm_height = None

        self.affine_calibration_matrix = affine_calibration_matrix
        self.wakeup_dmd = None
        self.dmd_needs_to_be_waken = self.DMD_NEEDS_TO_BE_WAKEN
        if uncropped:
            self.SET_ROI_REQUIRED = False
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        # isContinuousFocusEnabled()/State are trustworthy only right after a
        # fresh config load (mid-session they stay frozen at this value).
        # Snapshot it as a fallback for read_pfs_engaged(); the live read
        # (reload + Status) is what the engine actually uses.
        self.pfs_on_at_init = bool(self.mmc.isContinuousFocusEnabled())
        self.mmc.setConfig(groupName="System", configName="Startup")
        # Pin camera binning before set_roi(): MM camera drivers reset
        # the ROI on a binning change, so binning must come first.
        self.mmc.setConfig("Binning", self.BINNING)
        if self.SET_ROI_REQUIRED:
            self.set_roi()
        self.register_engine()

        self.slm_dev = self.mmc.getSLMDevice()
        self.slm_width = self.mmc.getSLMWidth(self.slm_dev)
        self.slm_height = self.mmc.getSLMHeight(self.slm_dev)
        self.dmd = DMD(
            self.mmc,
            resolve_power=self.resolve_power,
            affine_matrix=self.affine_calibration_matrix,
        )
        self.wakeup_dmd = KeepDMDAlive(self.mmc, self.dmd)
        # The keep-alive refresh only matters while live view is running.
        # During an MDA the engine drives the DMD on every event and the hold
        # exposure spans the inter-frame gaps, so tying the thread to the
        # live-acquisition signals keeps it off for the whole run.
        self.mmc.events.continuousSequenceAcquisitionStarted.connect(
            self.wakeup_dmd.run
        )
        self.mmc.events.sequenceAcquisitionStopped.connect(self.wakeup_dmd.stop)
        # Interactive snaps and live starts bypass the engine, so the core
        # re-displays the live pattern on demand (parked mirrors otherwise
        # black out any capture until a manual all_on()).
        if hasattr(self.mmc, "set_dmd_live_hook"):
            self.mmc.set_dmd_live_hook(self.dmd.display_livemode)

        self.image_height = self.mmc.getImageHeight()
        self.image_width = self.mmc.getImageWidth()

    def calibrate_dmd(
        self,
        calibration_channel,
        verbose=False,
        n_points=15,
        radius=4,
        exposure=25,
        marker_style="x",
        calibration_points_DMD=None,
        background=True,
    ):
        """Calibrate the DMD against the camera. Always runs when called
        (re-call to retune, e.g. with a different channel or power).

        Args:
            calibration_channel: the light path (Channel/PowerChannel) to image
                the DMD spots with; pass per experiment (e.g. UV vs cyan).
            background: When True (default), the calibration MDAs run on a
                worker thread while this call pumps the Qt event loop, so
                napari stays responsive and previews the calibration spots
                live. The call still blocks until calibration finishes;
                it just doesn't freeze the GUI. Set False to run
                synchronously on the calling thread.

        Note:
            With ``background=True`` and ``verbose=True`` the matplotlib
            diagnostic plots are created from the worker thread. With the
            Jupyter inline backend this routes to the running cell fine;
            if plots misbehave, use ``background=False`` for verbose runs.
        """
        if self.dmd is None:
            return

        def _do_calibration() -> None:
            # The calibration MDA's setup_sequence stops live acquisition,
            # which stops KeepDMDAlive, and the engine then drives the DMD
            # itself for every calibration event.
            self.dmd.calibrate(
                calibration_channel,
                verbose=verbose,
                n_points=n_points,
                radius=radius,
                exposure=exposure,
                marker_style=marker_style,
                calibration_points_DMD=calibration_points_DMD,
            )

        if not background:
            _do_calibration()
            return

        # Run on a worker thread; pump Qt here so napari keeps repainting
        # and previewing the calibration frames. The call still blocks
        # until calibration is done; it just doesn't starve the GUI.
        done = threading.Event()
        box: list[BaseException] = []

        def _worker() -> None:
            try:
                _do_calibration()
            except BaseException as exc:  # noqa: BLE001
                box.append(exc)
            finally:
                # the calibration MDAs ran on this thread
                drain_posted_qt_events()
                done.set()

        threading.Thread(target=_worker, name="DMDCalibration", daemon=True).start()
        while not done.wait(timeout=0.05):
            pump_qt_events()
        if box:
            raise box[0]

    def set_roi(self):
        """Apply the class's ROI_* settings as-is (after binning is set)."""
        self.mmc.clearROI()
        self.mmc.setROI(self.ROI_X, self.ROI_Y, self.ROI_WIDTH, self.ROI_HEIGHT)

    def post_experiment(self):
        """Post-process the experiment."""
        pass

    def shutdown(self):
        """Tear down hardware state so the microscope can be discarded.

        Stops the DMD wakeup loop and unloads all Micro-Manager devices
        so COM ports (notably the LED on COM3) and the SLM handle are
        released. Without this, pymmcore's native threads keep the
        Python process alive after the main thread exits, leaving a
        zombie that blocks the next session with
        ``Error in device "COM3"`` when MM tries to initialize.

        Idempotent with the atexit hook registered in
        :class:`PyMMCoreMicroscope`: calling ``shutdown`` explicitly just
        runs the same teardown earlier; if it's never called, the hook
        runs it at interpreter exit.
        """
        self._teardown_hardware()

    def _teardown_hardware(self) -> None:
        """Stop the DMD wakeup thread, then delegate to the base teardown.

        The wakeup thread keeps a reference to the SLM device; stopping
        it before ``unloadAllDevices`` avoids the unload racing the
        thread's next ``displaySLMImage`` call. Its live-acquisition
        connections go first so no late signal touches the SLM during unload.
        """
        wakeup = getattr(self, "wakeup_dmd", None)
        if wakeup is not None:
            with suppress(Exception):
                self.mmc.events.continuousSequenceAcquisitionStarted.disconnect(
                    wakeup.run
                )
                self.mmc.events.sequenceAcquisitionStopped.disconnect(wakeup.stop)
            with suppress(Exception):
                wakeup.stop()
        # a late snap must not call into the DMD while devices unload
        if hasattr(self.mmc, "set_dmd_live_hook"):
            with suppress(Exception):
                self.mmc.set_dmd_live_hook(None)
        super()._teardown_hardware()

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

    def reload_autofocus_device(self) -> bool:
        """Unload and re-load the autofocus (PFS) device.

        Reconstructing the device clears the stale state that causes the
        10 s-per-Z-move block (see ``MoenchMDAEngine._set_event_z``) and
        refreshes the device's ``Status`` property from the hardware, the
        basis of ``read_pfs_engaged()``. It does NOT refresh
        ``isContinuousFocusEnabled()``/``State``: those stay frozen at their
        config-load values (they appear to live in the TIScope hub's COM
        objects, which survive a child-device reload). Costs ~60 ms and does
        not disturb the PFS itself: a locked PFS stays locked across a reload.

        Why it is needed: ``TiCOMPFS::IsEnabled()`` issues a real COM call on
        every read (the Micro-Manager adapter does not cache anything), but
        Nikon's MIP COM layer returns a client-side cached parameter value that
        is only refreshed by ``IMipParameterEvents`` notifications, which are
        not delivered for the PFS parameter in-process. Constructing a new
        device (and hence a new parameter object) is what forces a fresh query;
        ``initializeDevice()`` on a live device is refused, and neither
        ``TIPFSStatus`` nor the ``TIScope`` hub exposes a refresh knob.
        """
        mmc = self.mmc
        label = mmc.getAutoFocusDevice()
        if not label:
            return False
        with self._pfs_lock:
            try:
                library = mmc.getDeviceLibrary(label)
                name = mmc.getDeviceName(label)
                parent = mmc.getParentLabel(label)
            except Exception as e:
                logger.warning("Cannot introspect %s for reload: %s", label, e)
                return False
            try:
                mmc.unloadDevice(label)
                mmc.loadDevice(label, library, name)
                if parent:
                    mmc.setParentLabel(label, parent)
                mmc.initializeDevice(label)
                mmc.setAutoFocusDevice(label)
            except Exception as e:
                logger.error("Failed to reload autofocus device %s: %s", label, e)
                return False
        return True

    def read_pfs_status_raw(self) -> str | None:
        """Live PFS ``Status`` string (e.g. 'Locked in focus'), or None.

        Reloads the autofocus device (~60 ms) and reads its ``Status``
        property. Returns ``None`` when the read path is unavailable (no AF
        device, reload or read failed).

        Why the reload: on the Nikon Ti, ``isContinuousFocusEnabled()`` /
        ``State`` are truthful only right after a full config load and are
        frozen afterwards, even across a reload of the AF device. ``Status``,
        however, is re-read from the hardware when the device object is
        constructed, so reload-then-read-``Status`` is the one in-session
        read that tracks reality, including changes made at the hardware
        PFS button, which are otherwise completely invisible to software.
        """
        # Hold the lock across both the reload and the read, so the read never
        # runs while another thread is reloading (and destroying) the device.
        with self._pfs_lock:
            if not self.reload_autofocus_device():
                return None
            label = self.mmc.getAutoFocusDevice()
            if not label:
                return None
            try:
                return self.mmc.getProperty(label, "Status")
            except Exception as e:
                logger.warning("Cannot read PFS Status: %s", e)
                return None

    def read_pfs_engaged(self) -> bool | None:
        """Live, truthful read of whether the PFS is currently engaged.

        Returns True/False from the live ``Status`` (see
        ``read_pfs_status_raw``), or ``None`` when the read path is
        unavailable; fall back to ``pfs_was_on()`` then.
        """
        status = self.read_pfs_status_raw()
        if status is None:
            return None
        return status in self.PFS_ENGAGED_STATUSES

    def read_pfs_engaged_fast(self) -> bool | None:
        """Cheap engaged probe: ``isContinuousFocusEnabled()``, no reload.

        With the patched Nikon adapter DLL, ``TiCOMPFS::IsEnabled()`` reports
        the live focus status (enabled = Focusing or Locked), so this read is
        truthful and matches the ``PFS_ENGAGED_STATUSES`` predicate at the
        cost of one COM call. WITHOUT the patch the value is frozen at its
        config-load state, so it never changes; callers must then rely on a
        periodic ``pfs_health()`` read instead (see
        ``MoenchMDAEngine._monitor_pfs_health``). Unlike ``pfs_health()``,
        this cannot report *why* the PFS is not holding, only whether it is.

        Takes the PFS lock so the read cannot hit the autofocus device while
        another thread is reloading (and destroying) it. Returns ``None``
        when the read fails.
        """
        with self._pfs_lock:
            try:
                return bool(self.mmc.isContinuousFocusEnabled())
            except Exception as e:
                logger.warning("Fast PFS engaged read failed: %s", e)
                return None

    def pfs_health(self) -> tuple[str, str]:
        """Live PFS health as ``(state, status_string)``.

        ``state`` is one of:
          * ``'engaged'``: actively focusing or locked ('Locked in focus',
            'Focusing');
          * ``'failing'``: a genuine fault the PFS reports it cannot recover
            from ('Focus lock failed', 'Dichroic mirror not inserted',
            'Unsupported objective lens');
          * ``'idle'``: not driving focus, no fault ('Within range of
            focus search', 'Out of focus search range'), normal when the PFS
            is simply off;
          * ``'unknown'``: the live read was unavailable.

        Only the live ``Status`` read exposes the failing states at all; the
        boolean enabled flag cannot. Costs one ~60 ms AF-device reload.
        """
        status = self.read_pfs_status_raw()
        if status is None:
            return ("unknown", "")
        if status in self.PFS_ENGAGED_STATUSES:
            return ("engaged", status)
        if status in self.PFS_FAILING_STATUSES:
            return ("failing", status)
        return ("idle", status)

    def pfs_was_on(self) -> bool:
        """Load-time PFS snapshot; fallback for ``read_pfs_engaged()``.

        Only consulted when the live read path is unavailable (no autofocus
        device, or a reload/read error). In normal operation prefer
        ``read_pfs_engaged()``, which reads the true current state, including
        changes made at the hardware PFS button.
        """
        return bool(getattr(self, "pfs_on_at_init", False))

    def disable_log_output(self):
        """Opt-in: silence noisy third-party loggers (matplotlib, ...).

        No longer called automatically. The baseline is: everything flows
        to stderr and the standard pymmcore-plus logfile from startup, and
        each experiment additionally mirrors records into its own log.txt
        while it runs (see Controller._run_mda_with_events). The ``faro``
        tree and ``pymmcore-plus`` are therefore never muted here — doing
        so would blind both the base log and the experiment log.
        """
        for name, other in logging.Logger.manager.loggerDict.items():
            if name == "faro" or name.startswith("faro.") or name == "pymmcore-plus":
                continue
            if isinstance(other, logging.Logger):
                other.setLevel(logging.CRITICAL)
                other.propagate = False
                for h in other.handlers[:]:
                    other.removeHandler(h)


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
        self._log = logging.getLogger(__name__)
        # Per-run PFS bookkeeping (see _set_event_z).
        self._af_handled_for_run = False
        self._af_reengage_after_run = False
        self._warned_slow_z = False
        # Per-run PFS health monitoring (see _monitor_pfs_health).
        self._pfs_started_engaged = False
        self._pfs_last_health_check = None  # time.monotonic() of last Status read
        self._pfs_lost_lock_warned = False

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
            # Try multiple times to set the configuration in case of transient
            # failures. The core (MoenchCMMCorePlus.setConfig) then verifies the
            # filter turret reached the requested cube and force-moves on a miss.
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
                        f"Failed to set channel (attempt {attempt}/"
                        f"{max_retry_attempts}). {e}"
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

        for attempt in range(1, max_retry_attempts + 1):
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
                        f"[WARN] TIXYDrive wait timed out (attempt {attempt}/{max_retry_attempts}); "
                        "retrying in 1 s..."
                    )
                    time.sleep(1)
                else:
                    # different error -> don't hide it
                    logger.warning("Failed to set XY position. %s", e)
                    raise

    def _set_pfs(self, on: bool) -> bool:
        """Enable or disable continuous focus (PFS).

        Returns True when the enable/disable call raises no error, not that the
        hardware confirmed. Takes the microscope's ``_pfs_lock`` so the write
        cannot hit the device while another thread reloads it. When the Nikon
        adapter DLL is patched to report the PFS on/off state from the hardware,
        both enable and disable take effect; otherwise only the first write per
        session does, so check the result with ``read_pfs_engaged()`` when it
        matters.
        """
        mic = self.microscope
        lock = getattr(mic, "_pfs_lock", None) if mic is not None else None
        with lock or nullcontext():
            try:
                self.mmcore.enableContinuousFocus(bool(on))
            except Exception as e:
                logger.warning("Failed to %s PFS: %s", "enable" if on else "disable", e)
                return False
        return True

    def _set_event_z(self, event: MDAEvent) -> None:
        """Disengage the PFS once, on the first Z move of the run.

        When a Z move is commanded while the PFS is on, TIZDrive's SetPosition
        turns the PFS off and then waits ~10 s for a confirmation that never
        comes, so every Z move costs ~10 s. Turning the PFS off in software
        once, before the first Z move, avoids this for the rest of the run, as
        long as the Nikon adapter DLL is patched to report the PFS on/off state
        from the hardware. So this reads the PFS state once (one device reload,
        on the first event that has a Z position), turns the PFS off if it was
        on, and turns it back on in ``teardown_sequence``. Events with no Z
        position never reach this method and leave the PFS alone.

        This depends on the patched adapter. Without the patch, Micro-Manager
        keeps reporting the PFS as on even after we turn it off, so the 10 s
        wait comes back; the slow-Z check below then warns.
        """
        mic = self.microscope
        if mic is not None and not self._af_handled_for_run:
            self._af_handled_for_run = True
            engaged = None
            try:
                engaged = mic.read_pfs_engaged()  # one reload, first Z event
            except Exception as e:
                logger.warning("Live PFS read failed: %s", e)
            if engaged is None:
                engaged = bool(getattr(mic, "pfs_on_at_init", False))
            if engaged:
                msg = (
                    "Z move requested while continuous focus (PFS) is "
                    "engaged: disengaging it for this run; it will be "
                    "re-engaged when the run ends."
                )
                logger.info(msg)
                print(f"[INFO] {msg}")
                if self._set_pfs(False):
                    self._af_reengage_after_run = True
                    time.sleep(0.5)  # let the PFS physically release Z

        # Safety check: if the Z move is still slow, Micro-Manager still thinks
        # the PFS is on despite the disable above, so warn.
        t0 = time.perf_counter()
        super()._set_event_z(event)
        dt = time.perf_counter() - t0
        if dt > 5.0 and not self._warned_slow_z:
            self._warned_slow_z = True
            msg = (
                f"Z move took {dt:.1f} s; the Nikon adapter still believed "
                "the PFS was engaged despite the pre-run handling. Check the "
                "PFS state at the microscope."
            )
            logger.warning(msg)
            print(f"[WARN] {msg}")

    def setup_sequence(self, sequence):
        """Stop live acquisition and reset per-run PFS state before the MDA.

        The live stop runs before the first event, so it only ever stops a
        live preview, never a real hardware sequence. It is not gated on
        ``isSequenceRunning()``: napari-micromanager can hold a live timer
        while the stream is already stopped, and that timer restarts live
        acquisition on the per-frame configSet/exposureChanged signals this
        engine emits, which then fights every snap. The emitted
        ``sequenceAcquisitionStopped`` is what clears that timer, and it also
        stops ``KeepDMDAlive`` for the duration of the run. The Controller
        stops live too; doing it here also covers calibration and bare
        ``mmc.mda.run()`` calls.

        The PFS bookkeeping resets the once-per-run Z handling (see
        ``_set_event_z``) and takes one live health read at run start (see
        ``_monitor_pfs_health``).
        """
        core = getattr(self, "mmcore", None)
        if core is not None:
            try:
                core.stopSequenceAcquisition()
            except Exception:
                self._log.exception("Failed to stop live acquisition before MDA")

        mic = self.microscope
        self._af_handled_for_run = False
        self._af_reengage_after_run = False
        self._warned_slow_z = False

        # PFS health: one live read at run start. Records whether the PFS is
        # holding focus (so _monitor_pfs_health can watch for a lost lock on
        # runs that rely on it), and warns immediately if it reports a fault.
        # It also starts the periodic-read clock, so the first event does not
        # repeat this read.
        self._pfs_started_engaged = False
        self._pfs_last_health_check = None
        self._pfs_lost_lock_warned = False
        if mic is not None and getattr(mic, "MONITOR_PFS_HEALTH", False):
            try:
                state, status = mic.pfs_health()
                self._pfs_last_health_check = time.monotonic()
            except Exception as e:
                state, status = "unknown", ""
                logger.warning("PFS health read failed at run start: %s", e)
            self._pfs_started_engaged = state == "engaged"
            if state == "failing":
                msg = (
                    f"PFS reports a fault at the start of this run "
                    f"(status: {status!r}); focus may be unreliable."
                )
                logger.warning(msg)
                print(f"[WARN] {msg}")

        return super().setup_sequence(sequence)

    def teardown_sequence(self, sequence) -> None:
        # The base teardown's state restoration calls waitForSystem(), which
        # can raise (e.g. a device wait timing out while TIZDrive's Busy()
        # flag is stuck), so the PFS re-engage below must run in a ``finally``
        # or the PFS would silently stay disengaged.
        #
        # KeepDMDAlive needs no restart here: it is tied to live acquisition
        # (started/stopped by the core's continuous-acquisition events), so it
        # starts again on its own when the user resumes live view.
        try:
            super().teardown_sequence(sequence)
        finally:
            mic = self.microscope
            # Turn the PFS back on only if _set_event_z turned it off. With the
            # patched Nikon adapter DLL this reaches the hardware and normally
            # works; we check the result and warn on failure (without the patch
            # only the first PFS on/off per session takes effect, so the user
            # then needs a fresh config load or the hardware button).
            if self._af_reengage_after_run:
                self._af_reengage_after_run = False
                self._set_pfs(True)
                time.sleep(1.5)
                try:
                    engaged = mic.read_pfs_engaged() if mic is not None else None
                except Exception:
                    engaged = None
                if engaged is False:
                    msg = (
                        "PFS could NOT be re-engaged from software. If the "
                        "Nikon adapter DLL is not patched, only the first "
                        "continuous-focus write per session reaches the "
                        "hardware; re-lock the PFS at the microscope button."
                    )
                    logger.warning(msg)
                    print(f"[WARN] {msg}")
                else:
                    logger.info("Re-engaged continuous focus (PFS) after run.")
                    print("[INFO] Re-engaged continuous focus (PFS) after run.")

    def setup_event(self, event: MDAEvent) -> None:
        """Set up the event, wait for the system, and monitor PFS health.

        Adds an all-on DMD-wake SLM injection before setup and a PFS-health
        check after the wait. The wait itself is the plain
        ``mmcore.waitForSystem()``; MoenchCMMCorePlus makes it fast on this
        scope by confirming the stages via position instead of their stuck
        ``Busy()`` flags.
        """
        event = self._maybe_inject_dmd_wake_slm(event)
        if isinstance(event, SequencedEvent):
            self.setup_sequenced_event(event)
        else:
            self.setup_single_event(event)

        # MoenchCMMCorePlus.waitForSystem() confirms the XY/Z stages by position
        # and skips the DMD's stuck Busy(), so this returns promptly.
        self.mmcore.waitForSystem()
        self._monitor_pfs_health(event)

    def _monitor_pfs_health(self, event: MDAEvent) -> None:
        """Warn if a PFS the run relies on loses focus mid-acquisition.

        Only active when the PFS was engaged at run start AND the run has not
        disengaged it for Z moves (i.e. the acquisition is trusting the PFS to
        hold focus). Runs that never engaged the PFS, or that disabled it for
        Z, are not monitored. Disable via ``Moench.MONITOR_PFS_HEALTH =
        False``; the flag is re-read on every event, so it works mid-run.

        Two-tier cost model. Every event pays only the cheap boolean probe
        (``read_pfs_engaged_fast``, one COM call, no reload). The ~60 ms
        authoritative ``Status`` read (an AF-device reload, which also carries
        the fault reason) runs only when the probe disagrees with the expected
        state, or when ``Moench.PFS_HEALTH_PERIOD_S`` has elapsed since the
        last one. The periodic read is the safety net for unpatched adapter
        DLLs, where the probe reads a frozen value and never flags anything.
        The schedule is time-based, not timepoint-based, so staggered runs
        with interleaved ``t`` indices pay the same bounded cost as
        batch-ordered ones.

        A drop out of the engaged states, whether to a fault ('Focus lock
        failed', ...) or simply out of lock ('Within range of focus search'),
        is reported once, and a recovery is noted.
        """
        mic = self.microscope
        if mic is None or not getattr(mic, "MONITOR_PFS_HEALTH", False):
            return
        if not self._pfs_started_engaged:
            return  # PFS was not holding focus at run start; nothing to watch
        if self._af_reengage_after_run:
            return  # we intentionally disengaged it for this (Z) run

        now = time.monotonic()
        period = float(getattr(mic, "PFS_HEALTH_PERIOD_S", 0) or 0)
        due = self._pfs_last_health_check is None or (
            period > 0 and now - self._pfs_last_health_check >= period
        )
        if not due:
            # Cheap probe: authoritative read only on an apparent transition.
            expected_engaged = not self._pfs_lost_lock_warned
            try:
                fast = mic.read_pfs_engaged_fast()
            except Exception:
                fast = None
            if fast is None or fast == expected_engaged:
                return

        self._pfs_last_health_check = now
        try:
            state, status = mic.pfs_health()
        except Exception as e:
            logger.warning("PFS health read failed mid-run: %s", e)
            return

        if state == "engaged":
            if self._pfs_lost_lock_warned:
                self._pfs_lost_lock_warned = False
                logger.info("PFS re-locked (status: %r).", status)
                print(f"[INFO] PFS re-locked (status: {status!r}).")
        elif state in ("failing", "idle"):
            # Engaged at start, now not locked/focusing -> lost lock or fault.
            if not self._pfs_lost_lock_warned:
                self._pfs_lost_lock_warned = True
                kind = "fault" if state == "failing" else "lost lock"
                msg = (
                    f"PFS {kind} during acquisition (status: {status!r}); "
                    "frames from here may be out of focus."
                )
                logger.warning(msg)
                print(f"[WARN] {msg}")

    def exec_event(self, event: MDAEvent):
        """Override to inject the all-on SLM on non-stim events.

        Must mirror ``setup_event``'s injection: ``MDARunner`` keeps its
        own reference to the original event, so a ``model_copy`` inside
        ``setup_event`` doesn't propagate. Without this override
        ``_exec_event_slm_image`` (the ``displaySLMImage`` that arms the
        pattern for the next camera TTL) never fires for non-stim
        events, leaving the previously latched stim pattern to pulse
        instead.
        """
        event = self._maybe_inject_dmd_wake_slm(event)
        yield from super().exec_event(event)

    def _maybe_inject_dmd_wake_slm(self, event: MDAEvent) -> MDAEvent:
        """Hold the DMD all-on for non-stim captures.

        Under OverlapMode=On the DMD re-pulses its currently loaded
        pattern on every camera TTL. After a stim event the stim
        pattern stays latched, so subsequent imaging events at the
        same timepoint (e.g. other FOVs in an FOV-batched burst)
        would pulse the stim pattern instead of an all-on frame and
        come back dark. KeepDMDAlive's 60 s refresh is too slow to
        catch that burst.
        """
        if event.slm_image is not None:
            return event
        mic = self.microscope
        if mic is None or not getattr(mic, "dmd_needs_to_be_waken", False):
            return event
        dmd = getattr(mic, "dmd", None)
        if dmd is None:
            return event
        if event.metadata.get("img_type") == ImgType.IMG_STIM:
            return event
        return event.model_copy(
            update={
                "slm_image": SLMImage(
                    data=True, device=dmd.name, exposure=event.exposure
                )
            }
        )

    def _set_event_slm_image(self, event: MDAEvent) -> None:
        """Upload the SLM pattern as an array, then force a long *hold* exposure.

        The base method uploads the image and, if the ``SLMImage`` carries an
        exposure, writes it via ``setSLMExposure``. On the Mosaic3 that value is
        the Andor ``ExposureTime``: the micromirrors hold the displayed pattern
        for exactly that long after the "Expose" and then park. We override it
        to ``Moench.DMD_HOLD_EXPOSURE_MS`` so the mirrors stay in the pattern
        across the whole camera/LED window (the Mosaic3 has no "Mirror On"
        mode). The stim *dose* is unaffected; it is gated by the
        camera-triggered LED, not the DMD.
        """
        core = self.mmcore
        # A scalar-bool SLMImage (all-on/all-off) reaches the base engine as
        # setSLMPixelsTo, which the Mosaic3 ignores without raising, leaving
        # whatever pattern was last latched on the mirrors. Expanding the
        # scalar to a uint8 array routes it through setSLMImage instead, the
        # only path this DMD reliably applies.
        if event.slm_image is not None:
            data = np.asarray(event.slm_image.data)
            if data.ndim == 0:
                slm_dev = event.slm_image.device or core.getSLMDevice()
                full = np.full(
                    (core.getSLMHeight(slm_dev), core.getSLMWidth(slm_dev)),
                    255 if bool(data.item()) else 0,
                    dtype=np.uint8,
                )
                event = event.model_copy(
                    update={
                        "slm_image": event.slm_image.model_copy(update={"data": full})
                    }
                )
        super()._set_event_slm_image(event)
        if event.slm_image is None:
            return
        mic = self.microscope
        hold_ms = (
            getattr(mic, "DMD_HOLD_EXPOSURE_MS", None) if mic is not None else None
        )
        if not hold_ms:
            return
        slm_device = event.slm_image.device or core.getSLMDevice()
        if not slm_device:
            return
        try:
            core.setSLMExposure(slm_device, float(hold_ms))
        except Exception as e:
            logger.warning("Failed to set DMD hold exposure. %s", e)

    def _exec_event_slm_image(self, img) -> None:
        """Display the pattern, then settle before the camera snaps.

        ``displaySLMImage`` ("Expose") commits the mask to the micromirrors; the
        short settle lets that commit finish before ``snapImage`` opens the
        camera and the camera-triggered LED fires, so the first part of the
        frame isn't integrated against a not-yet-committed pattern. Gated by
        ``Moench.DMD_SETTLE_MS`` (None/0 disables).
        """
        super()._exec_event_slm_image(img)
        mic = self.microscope
        settle_ms = getattr(mic, "DMD_SETTLE_MS", None) if mic is not None else None
        if settle_ms:
            time.sleep(float(settle_ms) / 1000.0)

    def setup_single_event(self, event: MDAEvent) -> None:
        """Setup hardware for a single (non-sequenced) event.

        This method is not part of the PMDAEngine protocol (it is called by
        `setup_event`, which *is* part of the protocol), but it is made public
        in case a user wants to subclass this engine and override this method.
        """
        if event.keep_shutter_open:
            ...

        max_retry_attempts = 10

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
                    print(f"Failed to set properties. {e}")
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
