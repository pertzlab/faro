"""Tests for events_to_dataframe() and RTMSequence/MDASequence compatibility."""

from __future__ import annotations

from useq import MDASequence

from rtm_pymmcore.core.data_structures import Channel, ImgType, RTMSequence
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


class TestRefPhaseInDataframe:
    """Ref as a separate phase shows up correctly in events_to_dataframe."""

    def test_ref_phase_has_img_type_in_metadata(self):
        phase1 = RTMSequence(
            time_plan={"interval": 1.0, "loops": 5},
            stage_positions=[(0.0, 0.0, 0.0)],
            channels=[{"config": "phase-contrast", "exposure": 50}],
        )
        phase2 = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=[(0.0, 0.0, 0.0)],
            channels=[{"config": "mCitrine", "exposure": 600}],
            rtm_metadata={"img_type": ImgType.IMG_REF},
        )
        events = list(phase1 + phase2)
        df = events_to_dataframe(events)

        assert len(df) == 6  # 5 imaging + 1 ref
        # Last row is ref
        assert df.iloc[-1]["img_type"] == ImgType.IMG_REF
        # First 5 rows: img_type is NaN (no img_type in their metadata)
        import pandas as pd
        for _, row in df.iloc[:5].iterrows():
            assert pd.isna(row["img_type"])
