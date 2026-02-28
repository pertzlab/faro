"""Convert legacy df_acquire DataFrames to RTMEvent lists."""

from __future__ import annotations

import pandas as pd

from rtm_pymmcore.core.data_structures import Channel, PowerChannel, RTMEvent


def _dict_to_channel(d: dict) -> Channel | PowerChannel:
    """Convert a channel dict to a Channel or PowerChannel dataclass.

    Returns PowerChannel if the dict contains a ``power`` key with a non-None
    value, otherwise returns a plain Channel.
    """
    if d.get("power") is not None:
        return PowerChannel(
            config=d["config"],
            exposure=d.get("exposure"),
            group=d.get("group"),
            power=d["power"],
        )
    return Channel(
        config=d["config"],
        exposure=d.get("exposure"),
        group=d.get("group"),
    )


def df_to_events(df_acquire: pd.DataFrame) -> list[RTMEvent]:
    """Convert a legacy *df_acquire* DataFrame to a list of :class:`RTMEvent`.

    One DataFrame row produces one RTMEvent.  Column mapping:

    =========== ====================================
    Column      RTMEvent field
    =========== ====================================
    fov         index["p"]
    timestep    index["t"]
    time        min_start_time
    fov_x       x_pos
    fov_y       y_pos
    fov_z       z_pos
    channels    channels (tuple of Channel dicts)
    stim=True   stim_channels built from stim_* cols
    optocheck   optocheck_channels from column
    =========== ====================================

    Remaining columns are forwarded as ``metadata``.
    """
    events: list[RTMEvent] = []

    # Columns that map directly to RTMEvent fields (not metadata)
    _SKIP_META = {
        "fov", "timestep", "time",
        "fov_x", "fov_y", "fov_z",
        "channels",
        "stim", "stim_channel_name", "stim_channel_group",
        "stim_channel_device_name", "stim_channel_power_property_name",
        "stim_power", "stim_exposure",
        "stim_timestep", "stim_exposure_list",
        "optocheck", "optocheck_channels",
        "device_name", "property_name",
    }

    for _, row in df_acquire.iterrows():
        fov = int(row["fov"])
        timestep = int(row["timestep"])

        # --- imaging channels ---
        channels_raw = row.get("channels", ())
        channels = tuple(_dict_to_channel(d) for d in channels_raw)

        # --- stim channels ---
        stim_channels: tuple[Channel, ...] = ()
        if row.get("stim", False) and row.get("stim_exposure") and row.get("stim_power"):
            stim_ch_name = row.get("stim_channel_name", "")
            stim_ch_group = row.get("stim_channel_group")
            stim_power = row.get("stim_power")
            stim_exposure = row.get("stim_exposure")
            stim_channels = (
                PowerChannel(
                    config=stim_ch_name,
                    exposure=stim_exposure,
                    group=stim_ch_group,
                    power=int(stim_power),
                ),
            )

        # --- optocheck channels ---
        optocheck_channels: tuple[Channel, ...] = ()
        if row.get("optocheck", False):
            oc_raw = row.get("optocheck_channels", ())
            if oc_raw is not None and len(oc_raw) > 0:
                optocheck_channels = tuple(_dict_to_channel(d) for d in oc_raw)

        # --- metadata: everything not consumed above ---
        metadata = {
            k: v for k, v in row.items()
            if k not in _SKIP_META
        }

        events.append(RTMEvent(
            index={"t": timestep, "p": fov},
            channels=channels,
            stim_channels=stim_channels,
            optocheck_channels=optocheck_channels,
            x_pos=row.get("fov_x"),
            y_pos=row.get("fov_y"),
            z_pos=row.get("fov_z"),
            min_start_time=float(row.get("time", 0)),
            metadata=metadata,
        ))

    return events
