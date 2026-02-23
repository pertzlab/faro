"""Tests for events_to_dataframe() and RTMSequence/MDASequence compatibility."""

from __future__ import annotations

from useq import MDASequence

from rtm_pymmcore.core.data_structures import Channel, RTMSequence
from rtm_pymmcore.core.utils import events_to_dataframe


class TestMDASequenceCompatibility:
    """A simple timelapse should yield equivalent events from MDASequence and RTMSequence."""

    def test_simple_timelapse_same_event_count(self):
        params = dict(
            time_plan={"interval": 1.0, "loops": 50},
            stage_positions=[(256.0, 256.0, 0.0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        events_rtm = list(RTMSequence(**params))
        events_mda = list(MDASequence(**params))

        assert len(events_rtm) == len(events_mda)

    def test_simple_timelapse_same_positions_and_timing(self):
        params = dict(
            time_plan={"interval": 1.0, "loops": 10},
            stage_positions=[(256.0, 256.0, 0.0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        events_rtm = list(RTMSequence(**params))
        events_mda = list(MDASequence(**params))

        for rtm_ev, mda_ev in zip(events_rtm, events_mda):
            assert rtm_ev.index.get("t") == mda_ev.index.get("t")
            assert rtm_ev.x_pos == mda_ev.x_pos
            assert rtm_ev.y_pos == mda_ev.y_pos
            assert rtm_ev.min_start_time == mda_ev.min_start_time

    def test_events_to_dataframe_works_for_both(self):
        params = dict(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0.0, 0.0, 0.0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        df_rtm = events_to_dataframe(list(RTMSequence(**params)))
        df_mda = events_to_dataframe(list(MDASequence(**params)))

        assert df_rtm.shape == df_mda.shape
        assert (df_rtm["channels"] == df_mda["channels"]).all()
        assert (df_rtm["timestep"] == df_mda["timestep"]).all()


class TestOptocheckColumns:
    """events_to_dataframe correctly reports optocheck status."""

    def test_no_optocheck_all_false(self):
        events = list(RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0.0, 0.0, 0.0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        ))
        df = events_to_dataframe(events)

        assert "optocheck" in df.columns
        assert not df["optocheck"].any()

    def test_optocheck_at_specific_frames(self):
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 10},
            stage_positions=[(0.0, 0.0, 0.0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
            optocheck_channels=(Channel(config="mCitrine", exposure=600),),
            optocheck_frames={4, 9},
        )
        df = events_to_dataframe(list(seq))

        assert df.loc[df["timestep"] == 4, "optocheck"].iloc[0] == True  # noqa: E712
        assert df.loc[df["timestep"] == 9, "optocheck"].iloc[0] == True  # noqa: E712
        assert df.loc[df["timestep"] == 0, "optocheck"].iloc[0] == False  # noqa: E712
        assert df["optocheck"].sum() == 2
