import queue
from dataclasses import dataclass
import enum
import numpy as np
import pandas as pd
from rtm_pymmcore.segmentation.base import Segmentator
from dataclasses import dataclass, InitVar


class FovState:
    """Per-FOV mutable state for tracking and stimulation."""

    def __init__(self):
        self.stim_mask_queue = queue.SimpleQueue()
        self.tracks_queue = queue.SimpleQueue()
        self.linker = None
        self.tracks_queue.put(pd.DataFrame())  # initial empty dataframe
        self.fov_timestep_counter = 0


@dataclass
class SegmentationMethod:
    name: str
    segmentation_class: Segmentator
    use_channel: int = 0
    save_tracked: bool = False


@dataclass
class Channel:
    name: str
    exposure: int
    group: str = None
    power: int = None
    device_name: str = None
    property_name: str = None


@dataclass
class StimChannel:
    name: str
    group: str
    power: int = None
    device_name: str = None
    power_property_name: str = None


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
    IMG_OPTOCHECK = enum.auto()


# ---------------------------------------------------------------------------
# RTMEvent — extended MDAEvent with multi-channel + stimulation support
# ---------------------------------------------------------------------------
from useq import MDAEvent
from useq._mda_event import SLMImage


class RTMEvent(MDAEvent):
    """Extended acquisition event: imaging channels + optional stim channels.

    Inherits all MDAEvent fields (index, x_pos, y_pos, z_pos, min_start_time,
    metadata, etc.). Adds multi-channel and stimulation support.

    The parent ``channel``/``exposure`` fields are not used directly — use
    ``channels`` and ``stim_channels`` instead. Call ``to_mda_events()`` to
    convert to standard MDAEvents for the microscope.
    """

    channels: tuple[Channel, ...] = ()
    stim_channels: tuple[Channel, ...] = ()

    class Config:
        arbitrary_types_allowed = True

    def to_mda_events(self, *, resolve_group=None, dmd=None,
                      stim_slm_image=None) -> list[MDAEvent]:
        """Convert to standard useq MDAEvents.

        Args:
            resolve_group: callable(config_name) -> group name
            dmd: DMD object for SLM image creation (optional)
            stim_slm_image: pre-computed SLMImage for stim (optional)

        Returns:
            List of standard MDAEvents (NOT RTMEvents).
        """
        events = []
        fov = self.index.get("p", 0)
        timestep = self.index.get("t", 0)
        has_stim = len(self.stim_channels) > 0
        fname = f"{fov:03d}_{timestep:05d}"

        base_meta = {
            **self.metadata,
            "fov": fov,
            "timestep": timestep,
            "fname": fname,
            "time": self.min_start_time or 0,
            "stim": has_stim,
        }
        if has_stim:
            sch = self.stim_channels[0]
            base_meta["stim_power"] = sch.power
            base_meta["stim_exposure"] = sch.exposure

        # Imaging channels
        for i, ch in enumerate(self.channels):
            ch_dict = {"config": ch.name}
            if ch.group:
                ch_dict["group"] = ch.group
            elif resolve_group:
                ch_dict["group"] = resolve_group(ch.name)

            props = None
            if ch.device_name and ch.property_name and ch.power is not None:
                props = [(ch.device_name, ch.property_name, ch.power)]

            events.append(MDAEvent(
                index={**dict(self.index), "c": i},
                channel=ch_dict,
                exposure=ch.exposure,
                x_pos=self.x_pos if i == 0 else None,
                y_pos=self.y_pos if i == 0 else None,
                z_pos=self.z_pos,
                min_start_time=self.min_start_time,
                metadata={**base_meta, "img_type": ImgType.IMG_RAW},
                properties=props,
            ))

        # Stim channels → separate events with slm_image
        if has_stim:
            for ch in self.stim_channels:
                ch_dict = {"config": ch.name}
                if ch.group:
                    ch_dict["group"] = ch.group
                elif resolve_group:
                    ch_dict["group"] = resolve_group(ch.name)

                props = None
                if ch.device_name and ch.property_name and ch.power is not None:
                    props = [(ch.device_name, ch.property_name, ch.power)]

                events.append(MDAEvent(
                    index=dict(self.index),  # no "c" for stim
                    channel=ch_dict,
                    exposure=ch.exposure,
                    min_start_time=self.min_start_time,
                    metadata={**base_meta, "img_type": ImgType.IMG_STIM},
                    properties=props,
                    slm_image=stim_slm_image,  # filled by Controller
                ))

        return events
