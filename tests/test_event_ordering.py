"""Tests for event iteration ordering and MDAEvent emission order.

Covers:
1. axis_order respected in RTMSequence.iter_events() — time-first vs position-first.
2. Stim assigned to correct frames regardless of axis_order.
3. to_mda_events() emits imaging → stim.
4. Channel ordering preserved within each (t, p) group.
5. Positions carry correct XY coordinates in every ordering.
6. Ref as a separate phase via RTMSequence concatenation.
"""

from __future__ import annotations

import pytest

from faro.core.data_structures import (
    Channel,
    ImgType,
    RTMEvent,
    RTMSequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tp(ev: RTMEvent) -> tuple[int, int]:
    """Extract (t, p) tuple from an RTMEvent."""
    return (ev.index["t"], ev.index["p"])


def _make_seq(*, n_time=3, n_pos=2, n_ch=1, axis_order="tpcz",
              stim_frames=frozenset()):
    """Build an RTMSequence with sensible defaults."""
    channels = [{"config": f"ch{i}", "exposure": 50} for i in range(n_ch)]
    positions = [(i * 100.0, i * 100.0, 0.0) for i in range(n_pos)]

    stim_channels = (Channel(config="stim-405", exposure=100),) if stim_frames else ()

    return RTMSequence(
        time_plan={"interval": 1.0, "loops": n_time},
        stage_positions=positions,
        channels=channels,
        axis_order=axis_order,
        stim_channels=stim_channels,
        stim_frames=stim_frames,
    )


# ===================================================================
# axis_order: iteration ordering
# ===================================================================

class TestAxisOrder:
    """iter_events() respects axis_order for (t, p) nesting."""

    def test_tpcz_time_outermost(self):
        """Default tpcz: all positions at t=0, then all at t=1, etc."""
        events = list(_make_seq(n_time=3, n_pos=2, axis_order="tpcz"))
        tp_order = [_tp(e) for e in events]
        assert tp_order == [
            (0, 0), (0, 1),
            (1, 0), (1, 1),
            (2, 0), (2, 1),
        ]

    def test_ptcz_position_outermost(self):
        """ptcz: all timepoints at p=0, then all at p=1."""
        events = list(_make_seq(n_time=3, n_pos=2, axis_order="ptcz"))
        tp_order = [_tp(e) for e in events]
        assert tp_order == [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
        ]

    def test_tpcz_three_positions(self):
        events = list(_make_seq(n_time=2, n_pos=3, axis_order="tpcz"))
        tp_order = [_tp(e) for e in events]
        assert tp_order == [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
        ]

    def test_ptcz_three_positions(self):
        events = list(_make_seq(n_time=2, n_pos=3, axis_order="ptcz"))
        tp_order = [_tp(e) for e in events]
        assert tp_order == [
            (0, 0), (1, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2),
        ]

    def test_single_position_order_identical(self):
        """With 1 position, tpcz and ptcz produce the same order."""
        ev_t = list(_make_seq(n_time=4, n_pos=1, axis_order="tpcz"))
        ev_p = list(_make_seq(n_time=4, n_pos=1, axis_order="ptcz"))
        assert [_tp(e) for e in ev_t] == [_tp(e) for e in ev_p]

    def test_single_timepoint_order_identical(self):
        """With 1 timepoint, tpcz and ptcz produce the same order."""
        ev_t = list(_make_seq(n_time=1, n_pos=4, axis_order="tpcz"))
        ev_p = list(_make_seq(n_time=1, n_pos=4, axis_order="ptcz"))
        assert [_tp(e) for e in ev_t] == [_tp(e) for e in ev_p]

    def test_event_count_same_regardless_of_order(self):
        ev_t = list(_make_seq(n_time=3, n_pos=3, axis_order="tpcz"))
        ev_p = list(_make_seq(n_time=3, n_pos=3, axis_order="ptcz"))
        assert len(ev_t) == len(ev_p) == 9

    def test_same_tp_pairs_regardless_of_order(self):
        """Both orderings visit the same set of (t, p) pairs."""
        ev_t = list(_make_seq(n_time=3, n_pos=2, axis_order="tpcz"))
        ev_p = list(_make_seq(n_time=3, n_pos=2, axis_order="ptcz"))
        assert set(_tp(e) for e in ev_t) == set(_tp(e) for e in ev_p)


# ===================================================================
# Stim assignment
# ===================================================================

class TestStimAssignment:
    """Stim flags follow the correct frames in any axis_order."""

    @pytest.mark.parametrize("axis_order", ["tpcz", "ptcz"])
    def test_stim_only_on_specified_frames(self, axis_order):
        stim_frames = {1, 3}
        events = list(_make_seq(
            n_time=5, n_pos=2, axis_order=axis_order,
            stim_frames=stim_frames,
        ))
        for ev in events:
            t = ev.index["t"]
            if t in stim_frames:
                assert len(ev.stim_channels) > 0, f"t={t} should have stim"
            else:
                assert len(ev.stim_channels) == 0, f"t={t} should not have stim"

    @pytest.mark.parametrize("axis_order", ["tpcz", "ptcz"])
    def test_all_positions_get_stim_at_stim_frame(self, axis_order):
        """Every position at a stim frame gets stim channels."""
        n_pos = 3
        events = list(_make_seq(
            n_time=4, n_pos=n_pos, axis_order=axis_order,
            stim_frames={2},
        ))
        stim_events = [ev for ev in events if len(ev.stim_channels) > 0]
        stim_positions = {ev.index["p"] for ev in stim_events}
        assert stim_positions == set(range(n_pos))
        assert all(ev.index["t"] == 2 for ev in stim_events)


# ===================================================================
# Channel ordering within events
# ===================================================================

class TestChannelOrdering:
    """Channels within each RTMEvent are ordered correctly."""

    @pytest.mark.parametrize("axis_order", ["tpcz", "ptcz"])
    def test_multichannel_order_preserved(self, axis_order):
        """Channel order within (t, p) matches the channels list."""
        events = list(_make_seq(n_time=2, n_pos=2, n_ch=3, axis_order=axis_order))
        for ev in events:
            ch_names = [c.config for c in ev.channels]
            assert ch_names == ["ch0", "ch1", "ch2"]

    @pytest.mark.parametrize("axis_order", ["tpcz", "ptcz"])
    def test_channel_count_matches(self, axis_order):
        n_ch = 3
        events = list(_make_seq(n_time=2, n_pos=2, n_ch=n_ch, axis_order=axis_order))
        for ev in events:
            assert len(ev.channels) == n_ch


# ===================================================================
# Position coordinates
# ===================================================================

class TestPositionCoordinates:
    """XY coordinates are correctly assigned regardless of axis_order."""

    @pytest.mark.parametrize("axis_order", ["tpcz", "ptcz"])
    def test_positions_carry_correct_xy(self, axis_order):
        events = list(_make_seq(n_time=3, n_pos=3, axis_order=axis_order))
        for ev in events:
            p = ev.index["p"]
            expected_x = p * 100.0
            expected_y = p * 100.0
            assert ev.x_pos == expected_x, f"p={p} x_pos={ev.x_pos} != {expected_x}"
            assert ev.y_pos == expected_y, f"p={p} y_pos={ev.y_pos} != {expected_y}"


# ===================================================================
# to_mda_events: emission order
# ===================================================================

class TestToMdaEventsOrder:
    """to_mda_events() emits imaging channels then stim channels."""

    def test_imaging_only(self):
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50), Channel(config="ch1", exposure=200)),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0, metadata={},
        )
        mda = ev.to_mda_events()
        types = [m.metadata["img_type"] for m in mda]
        assert types == [ImgType.IMG_RAW, ImgType.IMG_RAW]

    def test_imaging_then_stim(self):
        ev = RTMEvent(
            index={"t": 1, "p": 0},
            channels=(Channel(config="ch0", exposure=50),),
            stim_channels=(Channel(config="stim-405", exposure=100),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=1,
            metadata={"stim": True},
        )
        mda = ev.to_mda_events()
        types = [m.metadata["img_type"] for m in mda]
        assert types == [ImgType.IMG_RAW, ImgType.IMG_STIM]

    def test_stim_has_no_c_index(self):
        """Stim MDAEvents should not carry a 'c' index."""
        ev = RTMEvent(
            index={"t": 1, "p": 0},
            channels=(Channel(config="ch0", exposure=50),),
            stim_channels=(Channel(config="stim-405", exposure=100),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=1,
            metadata={"stim": True},
        )
        mda = ev.to_mda_events()
        img_ev, stim_ev = mda[0], mda[1]
        assert "c" in img_ev.index
        assert "c" not in stim_ev.index

    def test_multichannel_imaging_stim(self):
        """2 imaging + 1 stim → correct order and c-indices."""
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50), Channel(config="ch1", exposure=200)),
            stim_channels=(Channel(config="stim-405", exposure=100),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0,
            metadata={"stim": True},
        )
        mda = ev.to_mda_events()
        channels = [m.channel.config for m in mda]
        assert channels == ["ch0", "ch1", "stim-405"]

        c_indices = [m.index.get("c") for m in mda]
        assert c_indices == [0, 1, None]

    def test_xy_only_on_first_imaging_channel(self):
        """Only the first imaging MDAEvent carries x/y position."""
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50), Channel(config="ch1", exposure=50)),
            stim_channels=(Channel(config="stim-405", exposure=100),),
            x_pos=42.0, y_pos=99.0, z_pos=0,
            min_start_time=0, metadata={"stim": True},
        )
        mda = ev.to_mda_events()
        assert mda[0].x_pos == 42.0
        assert mda[0].y_pos == 99.0
        for m in mda[1:]:
            assert m.x_pos is None
            assert m.y_pos is None

    def test_img_type_from_metadata(self):
        """img_type is read from metadata, defaulting to IMG_RAW."""
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="mCitrine", exposure=600),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0,
            metadata={"img_type": ImgType.IMG_REF},
        )
        mda = ev.to_mda_events()
        assert mda[0].metadata["img_type"] == ImgType.IMG_REF


# ===================================================================
# plan_events: per-RTMEvent dispatch ordering (current vs previous mode)
# ===================================================================

class TestPlanEvents:
    """RTMEvent.plan_events orders imaging/stim within a single (t, p)."""

    def _event_with_stim(self):
        return RTMEvent(
            index={"t": 1, "p": 0},
            channels=(Channel(config="ch0", exposure=50),),
            stim_channels=(Channel(config="stim-405", exposure=100),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=1,
            metadata={"stim": True},
        )

    def test_current_mode_orders_imaging_then_stim(self):
        ev = self._event_with_stim()
        planned = ev.plan_events(stim_mode="current")
        types = [e.metadata["img_type"] for e in planned]
        assert types == [ImgType.IMG_RAW, ImgType.IMG_STIM]

    def test_previous_mode_orders_stim_then_imaging(self):
        ev = self._event_with_stim()
        planned = ev.plan_events(stim_mode="previous")
        types = [e.metadata["img_type"] for e in planned]
        assert types == [ImgType.IMG_STIM, ImgType.IMG_RAW]

    def test_no_stim_channels_returns_imaging_only(self):
        """Without stim, both modes return the same imaging events."""
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0, metadata={},
        )
        cur = ev.plan_events(stim_mode="current")
        prev = ev.plan_events(stim_mode="previous")
        assert len(cur) == 1
        assert len(prev) == 1
        assert cur[0].metadata["img_type"] == ImgType.IMG_RAW
        assert prev[0].metadata["img_type"] == ImgType.IMG_RAW

    def test_build_slm_is_attached_to_stim_events(self):
        """The build_slm callback's return value lands on stim events."""
        sentinel = object()
        ev = self._event_with_stim()
        planned = ev.plan_events(
            stim_mode="current",
            build_slm=lambda _ev: sentinel,
        )
        stim_events = [e for e in planned if e.metadata["img_type"] == ImgType.IMG_STIM]
        assert len(stim_events) == 1
        assert stim_events[0].slm_image is sentinel

    def test_build_slm_not_called_without_stim(self):
        """build_slm is skipped when the event has no stim channels."""
        called = []
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0, metadata={},
        )
        ev.plan_events(
            stim_mode="current",
            build_slm=lambda e: called.append(e) or None,
        )
        assert called == []

    def test_build_slm_returning_none_leaves_stim_unchanged(self):
        """A None SLM return value doesn't overwrite slm_image."""
        ev = self._event_with_stim()
        planned = ev.plan_events(
            stim_mode="current",
            build_slm=lambda _ev: None,
        )
        stim_events = [e for e in planned if e.metadata["img_type"] == ImgType.IMG_STIM]
        assert stim_events[0].slm_image is None


# ===================================================================
# Ref as a separate phase
# ===================================================================

class TestRefPhase:
    """Ref is a separate RTMSequence phase, not a special channel type."""

    def test_ref_phase_via_concatenation(self):
        """phase1 + phase2 produces experiment events then ref events."""
        phase1 = RTMSequence(
            time_plan={"interval": 1.0, "loops": 3},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        phase2 = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "mCitrine", "exposure": 600}],
            rtm_metadata={"img_type": ImgType.IMG_REF},
        )
        events = list(phase1 + phase2)

        assert len(events) == 4  # 3 imaging + 1 ref
        # First 3: regular imaging
        for ev in events[:3]:
            assert ev.channels[0].config == "phase-contrast"
            assert ev.metadata.get("img_type", ImgType.IMG_RAW) == ImgType.IMG_RAW
        # Last 1: ref
        assert events[3].channels[0].config == "mCitrine"
        assert events[3].metadata["img_type"] == ImgType.IMG_REF

    def test_ref_phase_timepoints_offset(self):
        """Ref phase timepoints are offset after the main experiment."""
        phase1 = RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        phase2 = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "mCitrine", "exposure": 600}],
            rtm_metadata={"img_type": ImgType.IMG_REF},
        )
        events = list(phase1 + phase2)
        opto_event = events[-1]
        assert opto_event.index["t"] == 5  # offset after last imaging t=4

    def test_ref_phase_multi_position(self):
        """Ref phase runs at all positions."""
        positions = [(0, 0, 0), (100, 100, 0), (200, 200, 0)]
        phase1 = RTMSequence(
            time_plan={"interval": 1.0, "loops": 3},
            stage_positions=positions,
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        phase2 = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=positions,
            channels=[{"config": "mCitrine", "exposure": 600}],
            rtm_metadata={"img_type": ImgType.IMG_REF},
        )
        events = list(phase1 + phase2)

        assert len(events) == 12  # 3t * 3p + 1t * 3p
        opto_events = [e for e in events if e.metadata.get("img_type") == ImgType.IMG_REF]
        assert len(opto_events) == 3
        opto_positions = {e.index["p"] for e in opto_events}
        assert opto_positions == {0, 1, 2}

    def test_ref_phase_multiple_channels(self):
        """Ref phase can have multiple verification channels."""
        phase1 = RTMSequence(
            time_plan={"interval": 1.0, "loops": 2},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        phase2 = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=[(0, 0, 0)],
            channels=[
                {"config": "mCitrine", "exposure": 600},
                {"config": "mCherry", "exposure": 400},
            ],
            rtm_metadata={"img_type": ImgType.IMG_REF},
        )
        events = list(phase1 + phase2)
        opto_event = events[-1]
        ch_names = [c.config for c in opto_event.channels]
        assert ch_names == ["mCitrine", "mCherry"]

    def test_ref_to_mda_events_tagged(self):
        """Ref phase MDAEvents carry IMG_REF type."""
        phase2 = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "mCitrine", "exposure": 600}],
            rtm_metadata={"img_type": ImgType.IMG_REF},
        )
        events = list(phase2)
        mda = events[0].to_mda_events()
        assert len(mda) == 1
        assert mda[0].metadata["img_type"] == ImgType.IMG_REF

    def test_stim_then_ref_phases(self):
        """Experiment with stim phase then ref phase."""
        phase1 = RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
            stim_channels=(Channel(config="stim-405", exposure=100),),
            stim_frames={2, 3, 4},
        )
        phase2 = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "mCitrine", "exposure": 600}],
            rtm_metadata={"img_type": ImgType.IMG_REF},
        )
        events = list(phase1 + phase2)

        # 5 imaging + 1 ref
        assert len(events) == 6
        stim_events = [e for e in events if len(e.stim_channels) > 0]
        assert len(stim_events) == 3
        opto_events = [e for e in events if e.metadata.get("img_type") == ImgType.IMG_REF]
        assert len(opto_events) == 1
        # Ref comes after all stim events
        assert opto_events[0].index["t"] > max(e.index["t"] for e in stim_events)


# ===================================================================
# ref_channels / ref_frames on RTMSequence (inline, like stim)
# ===================================================================

class TestRefChannelsInline:
    """ref_channels + ref_frames on RTMSequence (not via phase concatenation)."""

    def test_ref_channels_assigned_to_ref_frames(self):
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            ref_frames={0, 4},
        )
        events = list(seq)
        for ev in events:
            t = ev.index["t"]
            if t in {0, 4}:
                assert len(ev.ref_channels) == 1
                assert ev.ref_channels[0].config == "mCitrine"
            else:
                assert len(ev.ref_channels) == 0

    def test_ref_channels_absent_when_no_ref_frames(self):
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 3},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            ref_frames=frozenset(),  # no ref frames
        )
        events = list(seq)
        for ev in events:
            assert len(ev.ref_channels) == 0

    def test_ref_to_mda_events_emits_ref_type(self):
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50),),
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0, metadata={},
        )
        mda = ev.to_mda_events()
        assert len(mda) == 2
        assert mda[0].metadata["img_type"] == ImgType.IMG_RAW
        assert mda[1].metadata["img_type"] == ImgType.IMG_REF

    def test_ref_mda_events_have_c_index(self):
        """Ref MDAEvents carry a c-index (unlike stim)."""
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50), Channel(config="ch1", exposure=50)),
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0, metadata={},
        )
        mda = ev.to_mda_events()
        c_indices = [m.index.get("c") for m in mda]
        assert c_indices == [0, 1, 2]  # 2 imaging + 1 ref

    def test_stim_and_ref_together(self):
        """RTMEvent with both stim and ref channels."""
        ev = RTMEvent(
            index={"t": 0, "p": 0},
            channels=(Channel(config="ch0", exposure=50),),
            stim_channels=(Channel(config="stim-405", exposure=100),),
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            x_pos=0, y_pos=0, z_pos=0, min_start_time=0,
            metadata={"stim": True},
        )
        mda = ev.to_mda_events()
        types = [m.metadata["img_type"] for m in mda]
        assert types == [ImgType.IMG_RAW, ImgType.IMG_STIM, ImgType.IMG_REF]

    def test_ref_sequence_with_stim_and_ref(self):
        """RTMSequence with both stim and ref frames."""
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
            stim_channels=(Channel(config="stim-405", exposure=100),),
            stim_frames={2, 3},
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            ref_frames={0},
        )
        events = list(seq)
        ref_events = [e for e in events if len(e.ref_channels) > 0]
        stim_events = [e for e in events if len(e.stim_channels) > 0]
        assert len(ref_events) == 1
        assert ref_events[0].index["t"] == 0
        assert len(stim_events) == 2


# ===================================================================
# Negative indexing for stim_frames and ref_frames
# ===================================================================

class TestNegativeIndexing:
    """Negative indices in stim_frames and ref_frames resolve correctly."""

    def test_ref_frames_negative_one(self):
        """ref_frames={-1} resolves to last frame."""
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "ch0", "exposure": 50}],
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            ref_frames={-1},
        )
        events = list(seq)
        ref_events = [e for e in events if len(e.ref_channels) > 0]
        assert len(ref_events) == 1
        assert ref_events[0].index["t"] == 4  # last of 0..4

    def test_stim_frames_negative_one(self):
        """stim_frames={-1} resolves to last frame."""
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 3},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "ch0", "exposure": 50}],
            stim_channels=(Channel(config="stim-405", exposure=100),),
            stim_frames={-1},
        )
        events = list(seq)
        stim_events = [e for e in events if len(e.stim_channels) > 0]
        assert len(stim_events) == 1
        assert stim_events[0].index["t"] == 2

    def test_ref_frames_negative_two(self):
        """ref_frames={-2} resolves to second-to-last frame."""
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 10},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "ch0", "exposure": 50}],
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            ref_frames={-2},
        )
        events = list(seq)
        ref_events = [e for e in events if len(e.ref_channels) > 0]
        assert len(ref_events) == 1
        assert ref_events[0].index["t"] == 8

    def test_mixed_positive_and_negative(self):
        """Mix of positive and negative indices."""
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "ch0", "exposure": 50}],
            ref_channels=(Channel(config="mCitrine", exposure=600),),
            ref_frames={0, -1},
        )
        events = list(seq)
        ref_events = [e for e in events if len(e.ref_channels) > 0]
        ref_times = {e.index["t"] for e in ref_events}
        assert ref_times == {0, 4}

    def test_range_in_stim_frames(self):
        """range() objects work in stim_frames."""
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 10},
            stage_positions=[(0, 0, 0)],
            channels=[{"config": "ch0", "exposure": 50}],
            stim_channels=(Channel(config="stim-405", exposure=100),),
            stim_frames=set(range(0, 10, 2)),  # every other frame
        )
        events = list(seq)
        stim_events = [e for e in events if len(e.stim_channels) > 0]
        stim_times = {e.index["t"] for e in stim_events}
        assert stim_times == {0, 2, 4, 6, 8}
