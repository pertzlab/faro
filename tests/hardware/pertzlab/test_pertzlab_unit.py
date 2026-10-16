"""Pure-Python unit tests for Pertzlab-specific faro code.

Lives under ``tests/hardware/pertzlab/`` because its subjects are
Pertzlab-scope-only: per-microscope power-property mappings (declared
manually; an unmapped ``PowerChannel`` fails loud rather than silently
dropping the requested power),
:class:`faro.microscope.pertzlab.moench.MoenchCMMCorePlus`'s
position-confirmed stage waits and filter-turret verification, and the
Moench engine's DMD/SLM handling (scalar-image expansion, live-stop,
keep-alive lifecycle).

The tests use fakes and drive no hardware, but living under
``tests/hardware/`` puts them behind the ``--scope`` gate (any test there is
skipped without ``--scope``/``FARO_SCOPE``), so in practice they run on the
Moench alongside the real hardware tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from useq import MDAEvent, MDASequence

from faro.core._useq_compat import SLMImage
from faro.core.data_structures import Channel, PowerChannel
from faro.microscope.pertzlab.moench import KeepDMDAlive, MoenchMDAEngine

from tests.fake_mmc import build_core


# ===================================================================
# Power-property mapping (manual-only; unmapped power fails loud)
# ===================================================================


class TestPowerPropertyMapping:
    """Power mappings are declared on the microscope; no auto-detection."""

    def _mic(self, mapping):
        from faro.microscope.pymmcore import PyMMCoreMicroscope

        mic = PyMMCoreMicroscope()
        mic.POWER_PROPERTIES = mapping
        return mic

    def test_get_power_properties_is_manual_only(self):
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        assert mic.get_power_properties() == {"CyanStim": ("LED", "Cyan_Level")}

    def test_resolve_power_mapped(self):
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        ch = PowerChannel(config="CyanStim", exposure=10, power=25)
        assert mic.resolve_power(ch) == ("LED", "Cyan_Level", 25)

    def test_resolve_power_none_without_power(self):
        """Plain Channels and power-less PowerChannels resolve to None."""
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        assert mic.resolve_power(Channel(config="BF", exposure=10)) is None
        assert mic.resolve_power(PowerChannel(config="x", exposure=10)) is None

    def test_resolve_power_raises_when_unmapped(self):
        """A PowerChannel with power set but no mapping must fail loud."""
        mic = self._mic({"CyanStim": ("LED", "Cyan_Level")})
        ch = PowerChannel(config="mScarlet3", exposure=10, power=10)
        with pytest.raises(ValueError, match="mScarlet3"):
            mic.resolve_power(ch)

    def test_validate_hardware_flags_unmapped_power(self):
        """validate_hardware warns + fails when a power channel has no mapping."""
        from faro.core.utils import validate_hardware

        class _StubMMC:
            def getAvailableConfigGroups(self):
                return ["TTL_ERK"]

            def getAvailableConfigs(self, group):
                return ["mScarlet3"]

            def getCameraDevice(self):
                return ""  # skip the exposure-limit block

        events = [
            SimpleNamespace(
                channels=[PowerChannel(config="mScarlet3", exposure=200, power=10)],
                stim_channels=[],
                ref_channels=[],
            )
        ]
        with pytest.warns(UserWarning, match="no power-property mapping"):
            ok = validate_hardware(events, _StubMMC(), power_properties={})
        assert ok is False


# ===================================================================
# MoenchCMMCorePlus: position-confirmed stage waits
# ===================================================================


class _DuckCore:
    """Duck-typed self that runs the real MoenchCMMCorePlus wait logic.

    MoenchCMMCorePlus subclasses the SWIG ``CMMCorePlus``, whose C++ methods
    can't be monkeypatched and which segfaults if constructed without its
    real core. So instead we bind the real wait methods onto a plain object
    and stub the (few) core-boundary methods they call. ``_base_wait_for_device``
    stands in for the ``super().waitForDevice`` delegation and records the call,
    letting us assert which devices actually hit the base wait.
    """

    from faro.microscope.pertzlab.moench import MoenchCMMCorePlus as _M

    SKIP_BUSY_DEVICES = _M.SKIP_BUSY_DEVICES
    XY_TOLERANCE_UM = _M.XY_TOLERANCE_UM
    Z_TOLERANCE_UM = _M.Z_TOLERANCE_UM
    POSITION_CONFIRM_MAX_S = 0.5
    POSITION_CONFIRM_POLL_S = 0.0

    # the methods under test, bound to this duck
    waitForDevice = _M.waitForDevice
    waitForSystem = _M.waitForSystem
    _is_managed_stage = _M._is_managed_stage
    _confirm_xy = _M._confirm_xy
    _confirm_z = _M._confirm_z

    def __init__(
        self, devices, xy="TIXYDrive", z="TIZDrive", xy_pos=(0.0, 0.0), z_pos=0.0
    ):
        self._devices = devices
        self._xy = xy
        self._z = z
        self._xy_pos = xy_pos
        self._z_pos = z_pos
        self._xy_targets = {}
        self._z_targets = {}
        self._pending_moves = set()
        self.base_waits = []

    # --- core boundary the wait logic calls ---
    def getLoadedDevices(self):
        return self._devices

    def getXYStageDevice(self):
        return self._xy

    def getFocusDevice(self):
        return self._z

    def getXYPosition(self, *a):
        return self._xy_pos

    def getPosition(self, *a):
        return self._z_pos

    def _base_wait_for_device(self, label):
        self.base_waits.append(label)


class TestMoenchCorePositionConfirm:
    """MoenchCMMCorePlus confirms stages by position and skips the stuck DMD.

    The stuck-Busy devices (Mosaic3 DMD, TIXY/TIZDrive) must never hit the
    base waitForDevice, and a commanded stage move must be confirmed by
    reading the stage position.
    """

    def test_dmd_busy_skipped_others_waited(self):
        core = _DuckCore(["Core", "Camera", "Shutter", "Mosaic3", "TIXYDrive"])
        core.waitForSystem()
        assert "Mosaic3" not in core.base_waits, "DMD Busy must be skipped"
        assert "Core" not in core.base_waits, "Core is always skipped"
        assert "TIXYDrive" not in core.base_waits, "stage handled by position"
        assert "Camera" in core.base_waits
        assert "Shutter" in core.base_waits

    def test_commanded_stage_confirmed_by_position(self):
        core = _DuckCore(
            ["Core", "TIXYDrive", "TIZDrive"], xy_pos=(100.0, 50.0), z_pos=10.0
        )
        core._xy_targets["TIXYDrive"] = (100.0, 50.0)
        core._z_targets["TIZDrive"] = 10.0
        core._pending_moves.update({"TIXYDrive", "TIZDrive"})
        core.waitForSystem()
        assert "TIXYDrive" not in core.base_waits
        assert "TIZDrive" not in core.base_waits
        # a pending move is one-shot: cleared once confirmed
        assert not core._pending_moves

    def test_stage_without_pending_move_not_waited(self):
        # No move commanded -> the stuck Busy() is never consulted.
        core = _DuckCore(["Core", "TIXYDrive", "TIZDrive"])
        core.waitForSystem()
        assert "TIXYDrive" not in core.base_waits
        assert "TIZDrive" not in core.base_waits

    def test_nonstage_device_delegates_to_base(self):
        core = _DuckCore(["Core", "Camera"])
        core.waitForDevice("Camera")
        assert core.base_waits == ["Camera"]


# ===================================================================
# Filter-turret verify / force-move (NikonTI "Already at position" bug)
# ===================================================================


class _Setting:
    def __init__(self, dev, prop, val):
        self._dev, self._prop, self._val = dev, prop, val

    def getDeviceLabel(self):
        return self._dev

    def getPropertyName(self):
        return self._prop

    def getPropertyValue(self):
        return self._val


class _Config:
    def __init__(self, settings):
        self._s = settings

    def size(self):
        return len(self._s)

    def getSetting(self, i):
        return self._s[i]


class _DuckFilterCore:
    """Duck-typed self that runs the real MoenchCMMCorePlus filter-verify logic.

    Binds the real target-extraction and verify methods onto a plain object and
    stubs the core boundary they call. ``getState`` returns a scripted sequence
    so the test controls exactly what the read-back sees, decoupled from the
    set* calls, which is what lets us assert the control flow precisely. The
    force-move calls (``setState``) are recorders here, so the real verify code
    drives them without needing a live device.
    """

    from faro.microscope.pertzlab.moench import MoenchCMMCorePlus as _M

    FILTER_VERIFY_DEVICE = _M.FILTER_VERIFY_DEVICE
    FILTER_VERIFY_PROPERTY = _M.FILTER_VERIFY_PROPERTY
    FILTER_VERIFY_MAX_CORRECTIONS = _M.FILTER_VERIFY_MAX_CORRECTIONS
    FILTER_VERIFY_RAISE_ON_FAILURE = _M.FILTER_VERIFY_RAISE_ON_FAILURE

    # the methods under test, bound to this duck
    _filter_target_for_config = _M._filter_target_for_config
    _verify_filter_reached = _M._verify_filter_reached

    DEVICE = "TIFilterBlock1"
    TARGET_LABEL = "cube_T"
    TARGET_STATE = 2
    N_STATES = 6

    def __init__(self, read_states, *, loaded=True, config_has_filter=True):
        self._reads = list(read_states)
        self._stuck = self._reads[-1] if self._reads else 0
        self._loaded = [self.DEVICE] if loaded else ["Camera"]
        self._config_has_filter = config_has_filter
        self._verifying_filter = False
        self.setState_calls: list[int] = []
        self.getState_calls = 0

    # --- the override body, minus super() (super is exercised on hardware) ---
    def verify_config(self, group, config):
        target = self._filter_target_for_config(group, config)
        if target is not None:
            self._verify_filter_reached(target)

    # --- core boundary the verify logic calls ---
    def getLoadedDevices(self):
        return self._loaded

    def getConfigData(self, group, config):
        settings = [_Setting("Wheel-A", "Label", "x")]
        if self._config_has_filter:
            settings.append(_Setting(self.DEVICE, "Label", self.TARGET_LABEL))
        return _Config(settings)

    def getStateFromLabel(self, device, label):
        assert label == self.TARGET_LABEL
        return self.TARGET_STATE

    def getNumberOfStates(self, device):
        return self.N_STATES

    def waitForDevice(self, device):
        pass

    def getState(self, device):
        self.getState_calls += 1
        return self._reads.pop(0) if self._reads else self._stuck

    def setState(self, device, n):
        self.setState_calls.append(n)


class TestFilterVerify:
    """MoenchCMMCorePlus confirms the cube turret landed and force-moves on a miss.

    Guards the silent "Already at position; not moving" skip in the closed
    NikonTI adapter. The verify lives on the core, so interactive napari
    channel changes are covered too; the force-move goes neighbour -> target
    by state.
    """

    def test_fast_path_no_extra_move(self):
        # Turret already reads the target -> no rotation, no correction.
        core = _DuckFilterCore(read_states=[_DuckFilterCore.TARGET_STATE])
        core.verify_config("TTL_ERK", "mScarlet3")
        assert core.setState_calls == []

    def test_recovers_on_detected_mismatch(self):
        # First read wrong (suppressed move), recovers after one force-move.
        core = _DuckFilterCore(read_states=[0, _DuckFilterCore.TARGET_STATE])
        core.verify_config("TTL_ERK", "mScarlet3")
        # neighbour = (2 + 1) % 6 = 3, then back to the target state 2.
        assert core.setState_calls == [3, 2]

    def test_persistent_failure_logs_but_does_not_raise(self):
        # Always wrong: exhaust corrections, log loudly, do not raise (default).
        core = _DuckFilterCore(read_states=[0])  # stuck at 0 forever
        core.verify_config("TTL_ERK", "mScarlet3")
        # neighbour -> target, MAX_CORRECTIONS (3) times.
        assert core.setState_calls == [3, 2, 3, 2, 3, 2]

    def test_persistent_failure_raises_when_flagged(self):
        core = _DuckFilterCore(read_states=[0])
        core.FILTER_VERIFY_RAISE_ON_FAILURE = True
        with pytest.raises(RuntimeError, match="WRONG cube"):
            core.verify_config("TTL_ERK", "mScarlet3")

    def test_interactive_setstatelabel_path_verifies(self):
        # The direct state-label path (napari's interactive change) recovers too.
        core = _DuckFilterCore(read_states=[0, _DuckFilterCore.TARGET_STATE])
        target = core.getStateFromLabel(core.DEVICE, core.TARGET_LABEL)
        core._verify_filter_reached(target)
        assert core.setState_calls == [3, 2]

    def test_channel_without_turret_is_noop(self):
        # A channel whose preset doesn't drive the turret is left alone.
        core = _DuckFilterCore(read_states=[0], config_has_filter=False)
        core.verify_config("Binning", "2x2")
        assert core.getState_calls == 0
        assert core.setState_calls == []

    def test_device_not_loaded_is_noop(self):
        core = _DuckFilterCore(read_states=[0], loaded=False)
        core.verify_config("TTL_ERK", "mScarlet3")
        assert core.getState_calls == 0
        assert core.setState_calls == []


# ===================================================================
# DMD/SLM uploads, live-stop, and keep-alive lifecycle
# ===================================================================


class _SLMScene:
    """Minimal scene declaring a camera and an SLM; never renders."""

    image_height = image_width = 64
    channels = ("phase-contrast",)
    slm_name = "SLM"
    slm_shape = (64, 64)

    def render(self, event):
        return np.zeros((self.image_height, self.image_width), dtype=np.uint16)


@pytest.fixture()
def slm_core():
    return build_core(_SLMScene())


def _record_slm_calls(core):
    """Replace the core's two SLM upload paths with recorders."""
    calls = []
    core.setSLMImage = lambda label, image: calls.append(("setSLMImage", image))
    core.setSLMPixelsTo = lambda *args: calls.append(("setSLMPixelsTo", args))
    return calls


class TestScalarSLMImageExpansion:
    """Every SLM command reaches the core as an array, never as a scalar.

    A scalar all-on/all-off goes out via ``setSLMPixelsTo``, which some DMDs
    ignore without raising, leaving the last pattern on the mirrors.
    """

    @pytest.mark.parametrize(
        ("data", "expected_value"), [(True, 255), (False, 0)], ids=["all-on", "all-off"]
    )
    def test_scalar_routed_through_set_slm_image(self, slm_core, data, expected_value):
        engine = MoenchMDAEngine(slm_core)
        calls = _record_slm_calls(slm_core)

        engine._set_event_slm_image(
            MDAEvent(slm_image=SLMImage(data=data, device="SLM"))
        )

        assert [name for name, _ in calls] == ["setSLMImage"]
        image = calls[0][1]
        assert image.shape == _SLMScene.slm_shape
        assert image.dtype == np.uint8
        assert (image == expected_value).all()

    def test_array_is_passed_through_unchanged(self, slm_core):
        engine = MoenchMDAEngine(slm_core)
        calls = _record_slm_calls(slm_core)
        mask = np.zeros(_SLMScene.slm_shape, dtype=np.uint8)
        mask[10:20, 10:20] = 255

        engine._set_event_slm_image(
            MDAEvent(slm_image=SLMImage(data=mask, device="SLM"))
        )

        assert [name for name, _ in calls] == ["setSLMImage"]
        assert np.array_equal(calls[0][1], mask)


class TestSetupSequenceStopsLive:
    """Every MDA stops live acquisition first, whether or not it is running.

    A viewer can keep a live timer armed after the stream itself has stopped,
    and only the ``sequenceAcquisitionStopped`` signal clears it.
    """

    def test_stops_live_when_nothing_is_running(self, slm_core):
        stopped = []
        slm_core.events.sequenceAcquisitionStopped.connect(
            lambda *args: stopped.append(args)
        )
        engine = MoenchMDAEngine(slm_core)

        assert not slm_core.isSequenceRunning()
        engine.setup_sequence(MDASequence())

        assert stopped, "sequenceAcquisitionStopped must fire even when idle"


class TestKeepDMDAliveLifecycle:
    """The keep-alive thread runs while live acquisition does, and not longer.

    It refreshes the DMD so the mirrors do not park during live view. An MDA
    drives the DMD itself, so the thread must be off for the whole run.
    """

    def _keep_alive(self, core):
        displays = []
        dmd = SimpleNamespace(display_livemode=lambda: displays.append(1))
        return KeepDMDAlive(core, dmd), displays

    def test_repeated_run_does_not_spawn_a_second_thread(self, slm_core):
        keep_alive, _ = self._keep_alive(slm_core)
        try:
            keep_alive.run()
            first = keep_alive.thread
            keep_alive.run()
            assert keep_alive.thread is first
        finally:
            keep_alive.stop()

    def test_stop_while_idle_leaves_the_slm_alone(self, slm_core):
        keep_alive, _ = self._keep_alive(slm_core)
        displayed = []
        slm_core.displaySLMImage = lambda *args: displayed.append(args)

        keep_alive.stop()

        assert displayed == []

    def test_live_signals_drive_the_thread(self, slm_core):
        """Live start runs the thread, live stop stops it."""
        keep_alive, _ = self._keep_alive(slm_core)
        slm_core.events.continuousSequenceAcquisitionStarted.connect(keep_alive.run)
        slm_core.events.sequenceAcquisitionStopped.connect(keep_alive.stop)
        try:
            slm_core.events.continuousSequenceAcquisitionStarted.emit("Camera")
            assert keep_alive.thread is not None

            slm_core.events.sequenceAcquisitionStopped.emit("Camera")
            assert keep_alive.thread is None
        finally:
            keep_alive.stop()


# ===================================================================
# PFS health monitoring: two-tier probe (cheap boolean + periodic Status)
# ===================================================================


class _PFSMic:
    """Weakref-able Moench stand-in recording which PFS reads the monitor pays.

    ``read_pfs_engaged_fast`` returns scripted values (sticking at the last
    one, which models the frozen read of an unpatched adapter DLL);
    ``pfs_health`` returns a fixed (state, status) tuple. The call counters
    are what the tests assert on: the design goal is that a quiet event costs
    one fast read and zero reloads.
    """

    MONITOR_PFS_HEALTH = True
    PFS_HEALTH_PERIOD_S = 300.0

    def __init__(self, fast=(True,), health=("engaged", "Locked in focus")):
        self._fast = list(fast)
        self._health = health
        self.fast_calls = 0
        self.health_calls = 0

    def read_pfs_engaged_fast(self):
        self.fast_calls += 1
        return self._fast.pop(0) if len(self._fast) > 1 else self._fast[0]

    def pfs_health(self):
        self.health_calls += 1
        return self._health


def _monitor_engine(mic, *, started_engaged=True, warned=False, last_check=None):
    """Engine via __new__ with just the monitor's per-run state set."""
    import time
    import weakref

    eng = MoenchMDAEngine.__new__(MoenchMDAEngine)
    eng._microscope_ref = weakref.ref(mic)
    eng._pfs_started_engaged = started_engaged
    eng._pfs_lost_lock_warned = warned
    eng._af_reengage_after_run = False
    # "recent" = the periodic read is not due; None = due immediately.
    eng._pfs_last_health_check = (
        time.monotonic() if last_check == "recent" else last_check
    )
    return eng


class TestPFSHealthMonitor:
    """The monitor pays the reload only on a transition or on the period.

    Guards the staggered-schedule regression where the old once-per-timepoint
    dedup keyed on ``t`` degenerated to one ~60 ms AF-device reload per event
    when timepoints interleave across FOVs.
    """

    def test_quiet_event_costs_only_the_fast_probe(self):
        mic = _PFSMic(fast=(True,))
        eng = _monitor_engine(mic, last_check="recent")
        eng._monitor_pfs_health(MDAEvent())
        assert mic.fast_calls == 1
        assert mic.health_calls == 0

    def test_fast_flip_triggers_authoritative_read_and_warns(self):
        mic = _PFSMic(fast=(False,), health=("idle", "Within range of focus search"))
        eng = _monitor_engine(mic, last_check="recent")
        eng._monitor_pfs_health(MDAEvent())
        assert mic.health_calls == 1
        assert eng._pfs_lost_lock_warned is True

    def test_recovery_seen_by_probe_clears_the_warning(self):
        mic = _PFSMic(fast=(True,), health=("engaged", "Locked in focus"))
        eng = _monitor_engine(mic, warned=True, last_check="recent")
        eng._monitor_pfs_health(MDAEvent())
        assert mic.health_calls == 1
        assert eng._pfs_lost_lock_warned is False

    def test_periodic_read_catches_frozen_probe(self):
        # Unpatched DLL: the fast probe would read a frozen True forever, so
        # only the elapsed period gets the authoritative read to run and see
        # the lost lock.
        import time

        mic = _PFSMic(fast=(True,), health=("idle", "Out of focus search range"))
        eng = _monitor_engine(
            mic, last_check=time.monotonic() - mic.PFS_HEALTH_PERIOD_S - 1
        )
        eng._monitor_pfs_health(MDAEvent())
        assert mic.health_calls == 1
        assert eng._pfs_lost_lock_warned is True
        # the periodic clock advanced: the next event is quiet again
        mic.health_calls = 0
        mic._health = ("engaged", "Locked in focus")
        eng._monitor_pfs_health(MDAEvent())
        assert mic.health_calls == 1  # probe now disagrees with warned state
        assert eng._pfs_lost_lock_warned is False

    def test_period_zero_disables_the_periodic_read(self):
        import time

        mic = _PFSMic(fast=(True,))
        mic.PFS_HEALTH_PERIOD_S = 0
        eng = _monitor_engine(mic, last_check=time.monotonic() - 10_000)
        eng._monitor_pfs_health(MDAEvent())
        assert mic.health_calls == 0  # probe agreed; no periodic fallback

    def test_gates_suppress_all_reads(self):
        for kwargs in (
            {"started_engaged": False},  # PFS never held focus this run
            {},  # disabled via the flag below
        ):
            mic = _PFSMic()
            if not kwargs:
                mic.MONITOR_PFS_HEALTH = False
            eng = _monitor_engine(mic, last_check=None, **kwargs)
            eng._monitor_pfs_health(MDAEvent())
            assert mic.fast_calls == 0
            assert mic.health_calls == 0

        # a Z run that deliberately disengaged the PFS is not monitored
        mic = _PFSMic()
        eng = _monitor_engine(mic, last_check=None)
        eng._af_reengage_after_run = True
        eng._monitor_pfs_health(MDAEvent())
        assert mic.fast_calls == 0
        assert mic.health_calls == 0
