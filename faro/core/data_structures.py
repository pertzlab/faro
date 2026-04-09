from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass
import enum
from typing import Any, Iterable, Iterator
import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from faro.segmentation.base import Segmentator
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
from faro.core._useq_compat import SLMImage


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

    def plan_events(
        self,
        *,
        stim_mode: str = "current",
        build_slm=None,
        resolve_group=None,
        resolve_power=None,
    ) -> list[MDAEvent]:
        """Return MDAEvents in dispatch order for this (t, p) group.

        Splits the event into imaging/ref and stim sub-events, resolves the
        SLM image via ``build_slm`` if the group has stim channels, and
        orders them according to ``stim_mode``:

        - ``"current"`` (default): image → stim. The stim event fires after
          the imaging frame at the same (t, p), so the stim mask can be
          derived from analysis of the freshly-acquired image.
        - ``"previous"``: stim → image. The stim mask is reused from the
          previous timepoint's analysis; imaging happens afterwards.

        This is the per-RTMEvent planning step. Stim events are inserted
        *within* a single (t, p) group, NOT appended to the whole sequence:
        over the full run, stim events still appear interleaved at their
        correct (t, p) positions.

        Parameters
        ----------
        stim_mode:
            ``"current"`` or ``"previous"``.
        build_slm:
            Callable ``(rtm_event) -> SLMImage | None``. Called only when the
            event has stim channels. Return ``None`` to leave ``slm_image``
            unset (e.g. for microscopes without a DMD).
        resolve_group, resolve_power:
            Forwarded to :meth:`to_mda_events`.
        """
        mda_events = self.to_mda_events(
            resolve_group=resolve_group,
            resolve_power=resolve_power,
            stim_slm_image=None,
        )
        img_events = [
            e for e in mda_events
            if e.metadata.get("img_type") != ImgType.IMG_STIM
        ]
        stim_events = [
            e for e in mda_events
            if e.metadata.get("img_type") == ImgType.IMG_STIM
        ]

        if stim_events and build_slm is not None:
            slm = build_slm(self)
            if slm is not None:
                stim_events = [
                    e.model_copy(update={"slm_image": slm}) for e in stim_events
                ]

        if stim_mode == "previous":
            return stim_events + img_events
        return img_events + stim_events

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
        order produced by the parent iterator, so the resulting RTMEvents
        follow the same nesting that ``axis_order`` dictates.

        Negative indices in ``stim_frames`` and ``ref_frames`` are resolved
        relative to the total number of timepoints.
        """
        groups: dict[tuple, dict] = {}
        # Call the parent's iter_events directly (not via self) so we
        # bypass this very override and get the plain-MDAEvent stream.
        # MDASequence.__iter__ would dispatch back into RTMSequence.iter_events
        # and recurse forever. Equivalent to the old private ``iter_sequence``
        # call but via the parent class's public method.
        for mda_ev in MDASequence.iter_events(self):
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


# ---------------------------------------------------------------------------
# combine — axis-keyed composition of experiments
# ---------------------------------------------------------------------------


def _infer_interval(
    src: Any, events: list[RTMEvent], fallback: float = 1.0,
) -> float:
    """Best-effort 'what's the spacing between consecutive timepoints'.

    Prefers the source sequence's own ``time_plan.interval`` — this is the
    authoritative answer and is available whenever ``src`` is still an
    ``RTMSequence``. Falls back to the smallest non-zero gap between
    ``min_start_time`` values across events (which is robust to multi-FOV
    experiments where adjacent events in iteration order share a timepoint),
    and finally to ``fallback`` when nothing else is available.
    """
    if isinstance(src, RTMSequence) and src.time_plan is not None:
        interval = getattr(src.time_plan, "interval", None)
        if interval is not None:
            try:
                return float(interval)
            except (TypeError, ValueError):
                pass
    times = sorted({e.min_start_time or 0 for e in events})
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return min(gaps) if gaps else fallback


def _combine_pair(
    a: RTMSequence | Iterable[RTMEvent],
    b: RTMSequence | Iterable[RTMEvent],
    *,
    axis: str,
    offset_time: bool,
) -> list[RTMEvent]:
    """Pairwise merge used by :func:`combine` (not a public API).

    Offsets ``b``'s ``axis`` index past ``a``'s max, optionally shifts
    ``b``'s ``min_start_time``, and either appends (axis="t") or
    sorts-by-time to interleave (axis="p"). All the tricky semantics
    of experiment composition live here; :func:`combine` just folds
    this over a variadic input list.

    Note: the channel-match precondition for ``axis="p"`` is enforced
    in :func:`combine` itself (which still sees the original sources),
    not here — by the time ``_combine_pair`` runs, ``a`` has usually
    already been flattened to a list.
    """
    events_a = list(a)
    events_b = list(b)

    if not events_a:
        return events_b
    if not events_b:
        return events_a

    # Axis offset: shift b's axis key past a's max.
    max_key = max(e.index.get(axis, 0) for e in events_a) + 1

    # Time offset: shift b's min_start_time past a's end.
    time_offset = 0.0
    if offset_time:
        max_time_a = max(e.min_start_time or 0 for e in events_a)
        interval = _infer_interval(b, events_b)
        time_offset = max_time_a + interval

    offset_b: list[RTMEvent] = []
    for ev in events_b:
        updates: dict = {
            "index": {**dict(ev.index), axis: ev.index.get(axis, 0) + max_key},
        }
        if offset_time:
            updates["min_start_time"] = (ev.min_start_time or 0) + time_offset
        offset_b.append(ev.model_copy(update=updates))

    # Merge strategy depends on axis:
    # - axis="t": append. Each source's own axis_order ordering is
    #   preserved across the boundary, because the time offset guarantees
    #   events_b all come after events_a chronologically. A re-sort here
    #   would scramble ptcz phase boundaries.
    # - axis="p": sort by (min_start_time, p) to interleave the two
    #   position groups at each timepoint.
    if axis == "t":
        return events_a + offset_b

    merged = events_a + offset_b
    merged.sort(key=lambda e: (e.min_start_time or 0, e.index.get("p", 0)))
    return merged


def combine(
    *sources: RTMSequence | Iterable[RTMEvent],
    axis: str = "t",
    offset_time: bool | None = None,
) -> list[RTMEvent]:
    """Combine N experiments by offsetting one axis of each past the previous.

    The single, explicit composition primitive for multi-step experiments.
    Two common shapes:

    - ``axis="t"`` (default, sequential):
        Each source runs AFTER the previous one in wall-clock time. Every
        source's ``t`` indices and ``min_start_time`` are shifted past the
        last event of the accumulated result. This is the phase-chain used
        for multi-step experiments like ``baseline + treatment + washout``::

            events = combine(baseline, treatment, washout, axis="t")

    - ``axis="p"`` (parallel / interleave):
        Sources run ALONGSIDE each other, at additional position indices.
        Each source's ``p`` indices are shifted past the accumulated max;
        the combined event list is sorted by time so FOVs from every
        sub-experiment are visited at each timepoint. Useful for
        multi-setup experiments where stim schedules, treatments, or
        metadata differ per FOV group but everything shares the clock::

            events = combine(setup_a, setup_b, setup_c, axis="p")

    Single-source and empty calls are degenerate cases::

        combine()                   # -> []
        combine(seq)                # -> list(seq)
        combine(seq, axis="p")      # -> list(seq)

    Parameters
    ----------
    *sources:
        Two or more ``RTMSequence`` instances (or already-iterated
        ``list[RTMEvent]``) to combine left-to-right.
    axis:
        Which index key to offset for each subsequent source. Defaults to
        ``"t"``. Any key present on the events is valid; uncommon axes
        (``"c"``, ``"z"``, ...) are accepted but rarely what you want.
    offset_time:
        Whether subsequent sources' ``min_start_time`` is also shifted
        past the accumulated end. Defaults to ``True`` when ``axis="t"``
        (so phases actually run sequentially), and ``False`` otherwise
        (so parallel sub-experiments share the clock). Set explicitly to
        override — e.g. ``offset_time=True`` with ``axis="p"`` gives a
        parallel block that starts with a wall-clock delay.

    Returns
    -------
    list[RTMEvent]
        Flat event list ready to pass to ``ctrl.run_experiment``.

    Raises
    ------
    ValueError
        When ``axis="p"`` and two sources declare different imaging
        channels. The v1 writer allocates a single channel set across all
        positions, so heterogeneous channels per FOV are unsupported
        until the useq-schema v2 migration lands.
    """
    if not sources:
        return []
    if offset_time is None:
        offset_time = (axis == "t")

    # Precondition for axis="p": every RTMSequence source must declare
    # the same imaging channels. The v1 writer allocates a single channel
    # set across all positions, so heterogeneous channels per FOV are
    # unsupported until v2 sub-sequences land. This check has to run on
    # the raw sources (before the first _combine_pair flattens them),
    # because once ``result`` is a plain ``list[RTMEvent]`` the original
    # ``channels`` tuple is no longer recoverable.
    if axis == "p":
        seq_sources = [s for s in sources if isinstance(s, RTMSequence)]
        if len(seq_sources) >= 2:
            ref_channels = [getattr(ch, "config", None) for ch in seq_sources[0].channels]
            for s in seq_sources[1:]:
                other = [getattr(ch, "config", None) for ch in s.channels]
                if other != ref_channels:
                    raise ValueError(
                        f"combine(axis='p') requires matching imaging channels; "
                        f"got {ref_channels} vs {other}. Heterogeneous channels "
                        f"per position is a v2 feature — track the useq-schema "
                        f"v2 migration."
                    )

    result: list[RTMEvent] = list(sources[0])
    for src in sources[1:]:
        result = _combine_pair(
            result, src, axis=axis, offset_time=offset_time,
        )
    return result
