from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass
import enum
from typing import Any, Generic, Iterator, Literal, Optional, TypeVar

_T = TypeVar("_T")
import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from faro.segmentation.base import Segmentator
from dataclasses import dataclass, InitVar


class FrameDispenser(Generic[_T]):
    """Frame-ordered handoff for per-frame values between pipeline workers.

    Replaces ``queue.SimpleQueue`` in places where values must be associated
    with the frame that produced them. A SimpleQueue hands out entries FIFO
    by thread-wait-arrival-order; concurrent pipeline workers could beat
    earlier frames to the queue and mix up which value went with which
    frame (see the TracksDispenser bug rationale — same pattern applies to
    stim masks and any other per-frame handoff).

    Values are keyed by an integer frame index (``event.index["t"]``).
    Two consumption modes are supported:

    - :meth:`get_predecessor` — wait for the most recent put *before* this
      frame (walking past skipped frames). Used for tracking where frame N
      needs frame N-1's accumulated df.
    - :meth:`get_at_frame` — wait for *this* frame's put specifically.
      Used for stim masks where frame N's stim event needs exactly frame
      N's mask.

    Entries are pruned after each successful :meth:`get_predecessor` so
    memory stays bounded at ~1 entry per FOV in steady state.

    Thread-safe. All public methods acquire an internal condition variable.
    """

    def __init__(self) -> None:
        self._entries: dict[int, _T] = {}
        self._skipped: set[int] = set()
        self._cond = threading.Condition()

    def put_for_frame(self, idx: int, value: _T) -> None:
        """Record *value* as this frame's output.

        Wakes any worker blocked in a ``get_*`` call.
        """
        with self._cond:
            self._entries[idx] = value
            self._cond.notify_all()

    def skip_frame(self, idx: int) -> None:
        """Mark a frame as skipped — no value will be produced for it.

        :meth:`get_predecessor` walks past skipped frames to find the
        most recent actual put; :meth:`get_at_frame` returns ``None``.
        Use this when a frame is dropped (e.g. pipeline queue saturated,
        exception) so downstream waiters don't block indefinitely.
        """
        with self._cond:
            self._skipped.add(idx)
            self._cond.notify_all()

    def get_predecessor(
        self, idx: int, timeout: Optional[float] = None
    ) -> Optional[_T]:
        """Return the value from the most recent resolved predecessor.

        Blocks until the immediate predecessor is resolved (either put or
        marked skipped). Walks past skipped frames. Returns ``None`` if
        the entire predecessor chain back to 0 is skipped or empty
        (typically meaning this is the first frame of the experiment).

        Args:
            idx: The frame index whose predecessor is wanted.
            timeout: Maximum seconds to block. ``None`` means wait
                forever.

        Raises:
            queue.Empty: if *timeout* elapses without the predecessor
                becoming available.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                status, info = self._resolve_predecessor_locked(idx)
                if status == "found":
                    value = self._entries[info]
                    self._prune_through_locked(info)
                    return value
                if status == "empty":
                    # Prune now — the tail of _skipped has no future consumer.
                    self._prune_through_locked(idx - 1)
                    return None
                # status == "waiting"; info = the unresolved index
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty(
                        f"FrameDispenser: timeout waiting for frame {info}"
                    )
                self._cond.wait(remaining)

    def get_at_frame(
        self, idx: int, timeout: Optional[float] = None
    ) -> Optional[_T]:
        """Return the value put *at* frame ``idx``.

        Blocks until ``idx`` is resolved (either put or marked skipped).
        Returns the value if put, or ``None`` if skipped.

        Args:
            idx: The frame index to wait for.
            timeout: Maximum seconds to block. ``None`` means wait
                forever.

        Raises:
            queue.Empty: if *timeout* elapses without ``idx`` being
                resolved.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if idx in self._entries:
                    value = self._entries.pop(idx)
                    self._skipped.discard(idx)
                    return value
                if idx in self._skipped:
                    self._skipped.discard(idx)
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty(
                        f"FrameDispenser: timeout waiting for frame {idx}"
                    )
                self._cond.wait(remaining)

    def reset(self) -> None:
        """Clear all state. Call only when no workers are in-flight —
        waiters with ``timeout=None`` would otherwise block on a chain
        whose predecessors you just erased.
        """
        with self._cond:
            self._entries.clear()
            self._skipped.clear()
            self._cond.notify_all()

    def _resolve_predecessor_locked(
        self, idx: int
    ) -> tuple[Literal["found", "empty", "waiting"], Optional[int]]:
        """Walk backward from ``idx - 1`` past skipped frames.

        Returns ``("found", k)`` if entry k is the nearest resolved put,
        ``("empty", None)`` if we walked past 0 without hitting an entry,
        or ``("waiting", k)`` if index k is unresolved (caller must wait).

        Caller must hold ``_cond``.
        """
        prev = idx - 1
        while prev >= 0:
            if prev in self._entries:
                return "found", prev
            if prev in self._skipped:
                prev -= 1
                continue
            return "waiting", prev
        return "empty", None

    def _prune_through_locked(self, idx: int) -> None:
        """Drop entries and skip markers with index ``<= idx``.

        Called after a successful ``get_predecessor`` — once frame N has
        consumed entry N-1, no future frame will ask for anything
        ``<= N-1`` (each frame asks only for its immediate predecessor).
        """
        for k in list(self._entries):
            if k <= idx:
                del self._entries[k]
        for k in list(self._skipped):
            if k <= idx:
                self._skipped.remove(k)


class FovState:
    """Per-FOV mutable state for tracking and stimulation."""

    def __init__(self):
        self.stim_mask_queue: FrameDispenser[Any] = FrameDispenser()
        self.tracks_queue: FrameDispenser[pd.DataFrame] = FrameDispenser()
        self.parquet_lock = threading.Lock()
        self.linker = None
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
            "ref_channels": [ch.config for ch in self.ref_channels],
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

    @field_validator("channels", mode="before")
    @classmethod
    def _coerce_channels(cls, value: Any) -> Any:
        """Convert custom Channel/PowerChannel dataclasses to dicts so useq can parse them."""
        from collections.abc import Sequence as Seq

        if not isinstance(value, Seq) or isinstance(value, str):
            return value
        coerced = []
        for v in value:
            if isinstance(v, Channel):
                coerced.append(asdict(v))
            else:
                coerced.append(v)
        return coerced

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

    def __len__(self) -> int:
        """Number of RTMEvents (unique (t, p) groups) in this sequence."""
        return sum(1 for _ in self.iter_events())

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
                    "pos_name": mda_ev.pos_name,
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
                pos_name=grp["pos_name"],
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
        result.sort(key=lambda e: (e.min_start_time or 0, e.index.get("p", 0)))
        return result

    def __add__(self, other: RTMSequence | list[RTMEvent]) -> list[RTMEvent]:
        """Concatenate two RTMSequences (or an RTMSequence and an event list).

        Timepoints in ``other`` are offset so they continue after ``self``.
        ``min_start_time`` is also offset by the last event's time + interval.
        """
        events_a = list(self)
        events_b = list(other)
        return self._offset_events(events_a, events_b)

    def check_fov_batching(
        self, time_per_fov: float, n_parallel: int | None = None,
    ) -> bool:
        """Check whether all FOVs in this sequence can be imaged in parallel.

        Args:
            time_per_fov: Time (in seconds) to image one FOV.
            n_parallel: Max FOVs per batch.  If *None*, computed from
                ``time_per_fov`` and the sequence's timepoint interval.
        """
        from faro.core.utils import check_fov_batching
        return check_fov_batching(list(self), time_per_fov, n_parallel)

    def __radd__(self, other: list[RTMEvent]) -> list[RTMEvent]:
        """Support ``list[RTMEvent] + RTMSequence``."""
        if isinstance(other, list):
            return self._offset_events(other, list(self))
        return NotImplemented
