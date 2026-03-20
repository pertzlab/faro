from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
import enum
from typing import Any, Iterator
import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from rtm_pymmcore.segmentation.base import Segmentator
from dataclasses import dataclass, InitVar


class FovState:
    """Per-FOV mutable state for tracking and stimulation."""

    def __init__(self):
        self.stim_mask_queue = queue.SimpleQueue()
        self.tracks_queue = queue.SimpleQueue()
        self.parquet_lock = threading.Lock()
        self.linker = None
        self.tracks_queue.put(pd.DataFrame())  # initial empty dataframe
        self.fov_timestep_counter = 0
        self.n_cells_latest = 0


@dataclass
class SegmentationMethod:
    name: str
    segmentation_class: Segmentator
    use_channel: int = 0
    save_tracked: bool = False


@dataclass
class Channel:
    """Acquisition channel (useq-compatible field names)."""
    config: str
    exposure: float = None
    group: str = None


@dataclass
class PowerChannel(Channel):
    """Channel with light-source power control.

    Only ``power`` is needed — the microscope resolves which device/property
    to set based on its hardware configuration.
    """
    power: int = None


@dataclass
class StimTreatment:
    treatment_name: str
    stim_timestep: tuple
    stim_exposure_list: tuple
    stim_power: int
    stim_channel_name: str
    stim_channel_group: str
    stim_channel_device_name: str
    stim_channel_power_property_name: str
    auto_repeat_stim_exposure: InitVar[bool] = False

    def __post_init__(self, *args, **kwargs):
        # dataclasses passes InitVar values to __post_init__ either as
        # positional args (older behavior) or keyword args. Be permissive
        # so callers can't trigger a TypeError if the InitVar isn't passed
        # the way we expect.
        if "auto_repeat_stim_exposure" in kwargs:
            auto_repeat_stim_exposure = kwargs.pop("auto_repeat_stim_exposure")
        elif len(args) > 0:
            auto_repeat_stim_exposure = args[0]
        else:
            auto_repeat_stim_exposure = False

        """Normalize stim-related fields on initialization.

        Accepts ranges, lists, numpy arrays, tuples or scalar ints and converts
        them to tuples (None is preserved).
        """
        if auto_repeat_stim_exposure and isinstance(self.stim_exposure_list, int):
            # If a single int was given and auto-repeat is requested, expand it
            # to match the length of stim_timestep (if present).
            try:
                length = (
                    len(self.stim_timestep) if self.stim_timestep is not None else 0
                )
            except TypeError:
                # stim_timestep might be an int (not normalized yet) -> treat as 1
                length = 1
            if length > 0:
                self.stim_exposure_list = (self.stim_exposure_list,) * length

        self.stim_timestep = _normalize_to_tuple(self.stim_timestep)
        self.stim_exposure_list = _normalize_to_tuple(self.stim_exposure_list)
        # Safety: ensure the init-only variable is not present on the instance
        # Remove the entry from the instance dict if present (guaranteed removal).
        if "auto_repeat_stim_exposure" in getattr(self, "__dict__", {}):
            try:
                self.__dict__.pop("auto_repeat_stim_exposure", None)
            except Exception:
                # best-effort cleanup; ignore any error
                pass


def _normalize_to_tuple(value):
    """Normalize a value to a tuple.

    Rules:
      - range -> tuple(range)
      - list/ndarray -> tuple(value)
      - tuple -> unchanged
      - scalar (int/float/str) -> (value,)
      - None -> None
    """
    if value is None:
        return None
    if isinstance(value, range):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    if isinstance(value, (list, np.ndarray)):
        return tuple(value)
    return (value,)


class ImgType(enum.Enum):
    IMG_RAW = enum.auto()
    IMG_STIM = enum.auto()
    IMG_REF = enum.auto()


# ---------------------------------------------------------------------------
# RTMEvent — extended MDAEvent with multi-channel + stimulation support
# ---------------------------------------------------------------------------
from useq import MDAEvent, MDASequence
from useq._mda_event import SLMImage
from useq._mda_sequence import iter_sequence


class RTMEvent(MDAEvent):
    """Extended acquisition event: imaging channels + optional stim/ref channels.

    Inherits all MDAEvent fields (index, x_pos, y_pos, z_pos, min_start_time,
    metadata, etc.). Adds multi-channel, stimulation, and reference support.

    The parent ``channel``/``exposure`` fields are not used directly — use
    ``channels``, ``stim_channels``, and ``ref_channels`` instead. Call
    ``to_mda_events()`` to convert to standard MDAEvents for the microscope.
    """

    channels: tuple[Channel, ...] = ()
    stim_channels: tuple[Channel, ...] = ()
    ref_channels: tuple[Channel, ...] = ()

    class Config:
        arbitrary_types_allowed = True

    def to_mda_events(self, *, resolve_group=None, resolve_power=None,
                      dmd=None, stim_slm_image=None) -> list[MDAEvent]:
        """Convert to standard useq MDAEvents.

        Args:
            resolve_group: callable(config_name) -> group name
            resolve_power: callable(channel) -> (device, property, power) or None
            dmd: DMD object for SLM image creation (optional)
            stim_slm_image: pre-computed SLMImage for stim (optional)

        Returns:
            List of standard MDAEvents (NOT RTMEvents).
        """
        events = []
        fov = self.index.get("p", 0)
        timestep = self.index.get("t", 0)
        has_stim = len(self.stim_channels) > 0
        fname = self.metadata.get("fname", f"{fov:03d}_{timestep:05d}")

        channel_names = [ch.config for ch in self.channels]

        base_meta = {
            **self.metadata,
            "fov": fov,
            "timestep": timestep,
            "fname": fname,
            "time": self.min_start_time or 0,
            "stim": has_stim,
            "channels": channel_names,
        }
        if has_stim:
            sch = self.stim_channels[0]
            base_meta["stim_power"] = getattr(sch, "power", None)
            base_meta["stim_exposure"] = sch.exposure

        # img_type from metadata (e.g. IMG_REF for ref phases)
        img_type = self.metadata.get("img_type", ImgType.IMG_RAW)

        def _resolve_ch(ch):
            """Build channel dict and properties for a Channel or PowerChannel."""
            ch_dict = {"config": ch.config}
            if ch.group:
                ch_dict["group"] = ch.group
            elif resolve_group:
                ch_dict["group"] = resolve_group(ch.config)
            props = None
            if resolve_power:
                props = resolve_power(ch)
                if props is not None:
                    props = [props]
            return ch_dict, props

        # Imaging channels
        for i, ch in enumerate(self.channels):
            ch_dict, props = _resolve_ch(ch)
            events.append(MDAEvent(
                index={**dict(self.index), "c": i},
                channel=ch_dict,
                exposure=ch.exposure,
                x_pos=self.x_pos if i == 0 else None,
                y_pos=self.y_pos if i == 0 else None,
                z_pos=self.z_pos,
                min_start_time=self.min_start_time,
                metadata={**base_meta, "img_type": img_type},
                properties=props,
            ))

        # Stim channels
        if has_stim:
            for ch in self.stim_channels:
                ch_dict, props = _resolve_ch(ch)
                events.append(MDAEvent(
                    index=dict(self.index),  # no "c" for stim
                    channel=ch_dict,
                    exposure=ch.exposure,
                    min_start_time=self.min_start_time,
                    metadata={**base_meta, "img_type": ImgType.IMG_STIM},
                    properties=props,
                    slm_image=stim_slm_image,  # filled by Controller
                ))

        # Ref channels
        if self.ref_channels:
            n_img = len(self.channels)
            for j, ch in enumerate(self.ref_channels):
                ch_dict, props = _resolve_ch(ch)
                events.append(MDAEvent(
                    index={**dict(self.index), "c": n_img + j},
                    channel=ch_dict,
                    exposure=ch.exposure,
                    min_start_time=self.min_start_time,
                    metadata={**base_meta, "img_type": ImgType.IMG_REF},
                    properties=props,
                ))

        return events


# ---------------------------------------------------------------------------
# Frame-set helpers
# ---------------------------------------------------------------------------


def _resolve_frame_set(frames, n_timepoints: int) -> set[int]:
    """Resolve a frame specification to a concrete set of non-negative indices.

    Accepts sets, frozensets, ranges, or any iterable of ints.
    Negative indices are resolved relative to *n_timepoints*
    (e.g. ``-1`` → ``n_timepoints - 1``).
    """
    resolved: set[int] = set()
    for f in frames:
        if f < 0:
            resolved.add(f + n_timepoints)
        else:
            resolved.add(f)
    return resolved


# ---------------------------------------------------------------------------
# RTMSequence — MDASequence subclass with stimulation + ref support
# ---------------------------------------------------------------------------


class RTMSequence(MDASequence):
    """MDASequence with stimulation, reference acquisition, and pipeline metadata.

    Iterating yields RTMEvent objects (not plain MDAEvents).
    Concatenate multiple sequences with ``+`` for multi-phase experiments.

    ``stim_frames`` and ``ref_frames`` accept sets, frozensets, or ``range``
    objects.  Negative indices are resolved relative to the total number of
    timepoints (e.g. ``-1`` → last frame).
    """

    stim_channels: tuple[Channel, ...] = ()
    stim_frames: set[int] | frozenset[int] = Field(default_factory=frozenset)
    stim_exposure: None | float | int | tuple[float | int, ...] | list[float | int] = None
    ref_channels: tuple[Channel, ...] = ()
    ref_frames: set[int] | frozenset[int] = Field(default_factory=frozenset)
    rtm_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {**MDASequence.model_config, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _validate_stim_exposure(self) -> RTMSequence:
        exp = self.stim_exposure
        if exp is None or isinstance(exp, (int, float)):
            return self
        # sequence type — length must match stim_frames
        if len(exp) != len(self.stim_frames):
            raise ValueError(
                f"stim_exposure length ({len(exp)}) must match "
                f"stim_frames length ({len(self.stim_frames)})"
            )
        return self

    def __iter__(self) -> Iterator[RTMEvent]:
        """Yield RTMEvents (overrides MDASequence.__iter__)."""
        return self.iter_events()

    def iter_events(self) -> Iterator[RTMEvent]:
        """Yield RTMEvents by grouping parent MDAEvents by (t, p).

        The (t, p) iteration order respects ``axis_order`` (inherited from
        MDASequence).  Dict insertion order preserves the first-encounter
        order produced by ``iter_sequence``, so the resulting RTMEvents
        follow the same nesting that ``axis_order`` dictates.

        Negative indices in ``stim_frames`` and ``ref_frames`` are resolved
        relative to the total number of timepoints.
        """
        groups: dict[tuple, dict] = {}
        for mda_ev in iter_sequence(self):
            t = mda_ev.index.get("t", 0)
            p = mda_ev.index.get("p", 0)
            key = (t, p)
            if key not in groups:
                groups[key] = {
                    "channels": [],
                    "x_pos": mda_ev.x_pos,
                    "y_pos": mda_ev.y_pos,
                    "z_pos": mda_ev.z_pos,
                    "min_start_time": mda_ev.min_start_time,
                }
            if mda_ev.channel:
                ch_name = mda_ev.channel.config
                groups[key]["channels"].append(
                    Channel(config=ch_name, exposure=mda_ev.exposure or 0)
                )

        merged_meta = {**self.metadata, **self.rtm_metadata}
        stim_tuple = tuple(self.stim_channels)
        ref_tuple = tuple(self.ref_channels)

        # Resolve negative indices for stim_frames and ref_frames
        max_t = max((t for (t, _p) in groups), default=0)
        n_timepoints = max_t + 1
        stim_set = _resolve_frame_set(self.stim_frames, n_timepoints)
        ref_set = _resolve_frame_set(self.ref_frames, n_timepoints)

        # Build per-frame stim exposure mapping
        stim_exposure_map: dict[int, float | int] | None = None
        if self.stim_exposure is not None and not isinstance(self.stim_exposure, (int, float)):
            # Map each stim frame to its corresponding exposure
            sorted_stim_frames = sorted(stim_set)
            stim_exposure_map = dict(zip(sorted_stim_frames, self.stim_exposure))

        for (t, p), grp in groups.items():
            if stim_tuple and t in stim_set:
                if self.stim_exposure is None:
                    stim = stim_tuple
                elif isinstance(self.stim_exposure, (int, float)):
                    stim = tuple(
                        Channel(config=ch.config, exposure=self.stim_exposure, group=ch.group)
                        for ch in stim_tuple
                    )
                else:
                    exp = stim_exposure_map[t]
                    stim = tuple(
                        Channel(config=ch.config, exposure=exp, group=ch.group)
                        for ch in stim_tuple
                    )
            else:
                stim = ()
            ref = ref_tuple if ref_tuple and t in ref_set else ()
            yield RTMEvent(
                index={"t": t, "p": p},
                channels=tuple(grp["channels"]),
                stim_channels=stim,
                ref_channels=ref,
                x_pos=grp["x_pos"],
                y_pos=grp["y_pos"],
                z_pos=grp["z_pos"],
                min_start_time=grp["min_start_time"],
                metadata=merged_meta,
            )

    @staticmethod
    def _offset_events(
        events_a: list[RTMEvent], events_b: list[RTMEvent],
    ) -> list[RTMEvent]:
        """Append *events_b* to *events_a* with offset timepoints and times."""
        if not events_a:
            return list(events_b)
        if not events_b:
            return list(events_a)

        max_t = max(e.index.get("t", 0) for e in events_a) + 1
        max_time = max(e.min_start_time or 0 for e in events_a)
        if len(events_b) >= 2:
            dt = (events_b[1].min_start_time or 0) - (
                events_b[0].min_start_time or 0
            )
        else:
            dt = 1.0
        time_offset = max_time + dt

        result = list(events_a)
        for ev in events_b:
            new_t = ev.index.get("t", 0) + max_t
            new_time = (ev.min_start_time or 0) + time_offset
            result.append(
                ev.model_copy(
                    update={
                        "index": {**dict(ev.index), "t": new_t},
                        "min_start_time": new_time,
                    }
                )
            )
        return result

    def __add__(self, other: RTMSequence | list[RTMEvent]) -> list[RTMEvent]:
        """Concatenate two RTMSequences (or an RTMSequence and an event list).

        Timepoints in ``other`` are offset so they continue after ``self``.
        ``min_start_time`` is also offset by the last event's time + interval.
        """
        events_a = list(self)
        events_b = list(other)
        return self._offset_events(events_a, events_b)

    def __radd__(self, other: list[RTMEvent]) -> list[RTMEvent]:
        """Support ``list[RTMEvent] + RTMSequence``."""
        if isinstance(other, list):
            return self._offset_events(other, list(self))
        return NotImplemented
