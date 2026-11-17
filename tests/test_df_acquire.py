"""The legacy df_acquire builder and its conversion to events."""

from __future__ import annotations

from types import SimpleNamespace

from faro.core.conversion import df_to_events
from faro.core.data_structures import Channel
from faro.core.utils import generate_df_acquire, generate_fov_positions_from_list

CHANNELS = [Channel(config="miRFP", exposure=100)]
OPTOCHECK = Channel(config="mCitrine", exposure=600)
MIC = SimpleNamespace(USE_ONLY_PFS=False)


def _fovs(n):
    return generate_fov_positions_from_list(
        MIC, [{"x": float(i), "y": 0.0, "z": 1.0} for i in range(n)]
    )


def test_fov_positions_get_list_indices():
    # FovPosition is a namedtuple: its .index is the tuple method, not a field.
    df = generate_df_acquire(
        _fovs(3),
        n_frames=2,
        time_between_timesteps=60,
        time_per_fov=2,
        channels=CHANNELS,
    )
    assert sorted(df["fov"].unique()) == [0, 1, 2]
    assert set(df["fname"]) == {f"{f:03d}_{t:05d}" for f in range(3) for t in range(2)}


def test_first_fov_index_keeps_groups_apart():
    fovs = _fovs(4)
    kwargs = dict(
        n_frames=2, time_between_timesteps=60, time_per_fov=2, channels=CHANNELS
    )
    group_1 = generate_df_acquire(fovs[:2], **kwargs)
    group_2 = generate_df_acquire(fovs[2:], first_fov_index=2, **kwargs)
    assert sorted(group_1["fov"].unique()) == [0, 1]
    assert sorted(group_2["fov"].unique()) == [2, 3]
    assert group_2.loc[group_2["fov"] == 2, "fov_x"].iloc[0] == 2.0
    # Both groups start at the same time: the batching counts within a group.
    assert group_2["time"].min() == group_1["time"].min() == 0


def test_fovs_beyond_one_interval_start_in_a_later_batch():
    # 10 s per FOV in a 20 s interval: 2 FOVs per batch.
    df = generate_df_acquire(
        _fovs(3),
        n_frames=2,
        time_between_timesteps=20,
        time_per_fov=10,
        channels=CHANNELS,
    )
    start = df.groupby("fov")["time"].min()
    assert start[0] == start[1] == 0
    assert start[2] == 2 * 20  # after the 2 frames of the first batch


def test_optocheck_becomes_ref_channels_on_its_timepoints():
    df = generate_df_acquire(
        _fovs(2),
        n_frames=3,
        time_between_timesteps=60,
        time_per_fov=2,
        channels=CHANNELS,
        channel_optocheck=OPTOCHECK,
        optocheck_timepoints=(2,),
    )
    events = df_to_events(df)
    with_ref = {(e.index["p"], e.index["t"]) for e in events if e.ref_channels}
    assert with_ref == {(0, 2), (1, 2)}
    ref = next(e for e in events if e.ref_channels).ref_channels
    assert [ch.config for ch in ref] == ["mCitrine"]
    assert ref[0].exposure == 600
    assert all("optocheck_channels" not in e.metadata for e in events)


def test_optocheck_defaults_to_the_last_frame():
    df = generate_df_acquire(
        _fovs(1),
        n_frames=4,
        time_between_timesteps=60,
        time_per_fov=2,
        channels=CHANNELS,
        channel_optocheck=[OPTOCHECK],
    )
    events = df_to_events(df)
    assert [e.index["t"] for e in events if e.ref_channels] == [3]


def test_no_optocheck_no_ref_channels():
    df = generate_df_acquire(
        _fovs(1),
        n_frames=2,
        time_between_timesteps=60,
        time_per_fov=2,
        channels=CHANNELS,
    )
    assert not any(e.ref_channels for e in df_to_events(df))
