"""Pure-Python fake microscope hardware for tests.

Builds a real :class:`~pymmcore_plus.experimental.unicore.UniMMCore`
instance wired to pure-Python camera and SLM devices. Tests get the
real pymmcore-plus MDA engine, signal chain, config-group machinery,
and ``validate_hardware`` surface without any C++ device adapters or
on-disk ``.cfg`` file.

The :class:`Scene` protocol is the test's plug-in: declare the
microscope shape (image size, channels, optional SLM) and provide a
``render(event)`` that maps an :class:`~useq.MDAEvent` to a frame. The
Controller runs through the normal pymmcore-plus path and
``frameReady`` fires with your rendered images.

Usage::

    class MyScene:
        image_height = 256
        image_width = 256
        channels = ["phase-contrast", "stim-405"]
        slm_name = "SLM"
        slm_shape = (256, 256)

        def render(self, event):
            return numpy_image

    core = build_core(MyScene())
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from pymmcore_plus.experimental.unicore import (
    GenericDevice,
    SimpleCameraDevice,
    UniMMCore,
)
from pymmcore_plus.experimental.unicore.devices._slm import SLMDevice
from useq import MDAEvent


DEFAULT_CHANNEL_GROUP = "Channel"
CAMERA_LABEL = "Camera"


@runtime_checkable
class Scene(Protocol):
    """Plug-in for :func:`build_core`: image source + hardware shape.

    Required: ``image_height``, ``image_width``, and
    ``render(event) -> np.ndarray``.

    Optional:

    - ``channels``: list of channel-config names to define on the core.
      Default: ``("phase-contrast",)``. Each config sets
      ``Camera.Exposure`` as a no-op so pymmcore accepts the group.
    - ``slm_name`` / ``slm_shape``: declare an SLM; when set,
      :class:`FakeMicroscope` attaches it and stim dispatches flow
      through ``setSLMImage`` / ``displaySLMImage``.
    - ``on_slm_displayed(mask, event) -> None``: fires after each
      ``displaySLMImage`` with the dispatched mask and triggering event.
    """

    image_height: int
    image_width: int

    def render(self, event: MDAEvent) -> np.ndarray: ...


class FakeCamera(SimpleCameraDevice):
    """Camera device that pulls each frame from a :class:`Scene`.

    Uses :class:`SimpleCameraDevice` so ROI handling comes for free.
    The :class:`~useq.MDAEvent` currently being executed is captured
    via the core's ``eventStarted`` signal (see :func:`build_core`) so
    :meth:`snap` can render the right frame.
    """

    def __init__(self, scene: Scene) -> None:
        super().__init__()
        self._scene = scene
        self.current_event: Optional[MDAEvent] = None
        self._exposure = 10.0

    def name(self) -> str:
        return CAMERA_LABEL

    def description(self) -> str:
        return "Pure-Python fake camera for tests"

    def sensor_shape(self) -> Tuple[int, int]:
        return (self._scene.image_height, self._scene.image_width)

    def dtype(self):
        return np.uint16

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = float(exposure)

    def snap(self, buffer: np.ndarray) -> Mapping:
        buffer[:] = self._scene.render(self.current_event)
        return {}


class FakeSLM(SLMDevice):
    """SLM device that records each dispatched mask via the scene hook."""

    def __init__(self, scene: Scene, camera: FakeCamera) -> None:
        super().__init__()
        self._scene = scene
        self._camera = camera
        self._buf: Optional[np.ndarray] = None
        self._exposure = 0.0

    def name(self) -> str:
        return self._scene.slm_name

    def description(self) -> str:
        return "Pure-Python fake SLM for tests"

    def shape(self) -> Tuple[int, int]:
        return self._scene.slm_shape

    def dtype(self):
        return np.uint8

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = float(exposure)

    def set_image(self, image: np.ndarray) -> None:
        self._buf = np.array(image)

    def display_image(self) -> None:
        on_slm = getattr(self._scene, "on_slm_displayed", None)
        if on_slm is not None and self._buf is not None:
            on_slm(self._buf, self._camera.current_event)


class _PropertyHolder(GenericDevice):
    """Stub device that registers arbitrary named properties.

    Used for light sources (``Spectra``, ``LedDMD`` etc.) that
    ``validate_hardware`` needs to see.
    Each property is a bare string slot or a float slot with ``(lo, hi)``
    limits.
    """

    def __init__(
        self, name: str, properties: Mapping[str, Optional[Tuple[float, float]]]
    ) -> None:
        super().__init__()
        self._name = name
        for prop, limits in properties.items():
            default = 0.0 if limits is not None else ""
            self.register_property(prop, default_value=default, limits=limits)

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return f"Fake device: {self._name}"


def build_core(
    scene: Scene,
    *,
    camera_exposure_limits: Optional[Tuple[float, float]] = None,
    extra_devices: Optional[
        Mapping[str, Mapping[str, Optional[Tuple[float, float]]]]
    ] = None,
    config_data: Optional[
        Mapping[Tuple[str, str], Iterable[Tuple[str, str, str]]]
    ] = None,
    extra_configs: Optional[Mapping[str, Iterable[str]]] = None,
    channel_group: Optional[str] = None,
) -> UniMMCore:
    """Build a ``UniMMCore`` wired with :class:`FakeCamera` (and SLM).

    Acquisition tests pass a ``Scene`` and nothing else. Validation tests
    (``validate_hardware``) pass the kwargs to register property limits,
    extra light-source devices, and per-config property settings so
    range checks find what they expect.

    ``channel_group=""`` leaves the core's channel group unset (some
    validation tests rely on ``getChannelGroup() == ""`` triggering a
    full config-group scan). Devices referenced in ``config_data`` but
    not in ``extra_devices`` are auto-registered so callers don't have
    to spell them twice.
    """
    core = UniMMCore()

    camera = FakeCamera(scene)
    if camera_exposure_limits is not None:
        camera.set_property_limits("Exposure", camera_exposure_limits)
    core.loadPyDevice(CAMERA_LABEL, camera)
    core.initializeDevice(CAMERA_LABEL)
    core.setCameraDevice(CAMERA_LABEL)
    core.setAutoShutter(False)

    slm_name = getattr(scene, "slm_name", None)
    if slm_name:
        slm = FakeSLM(scene, camera)
        core.loadPyDevice(slm_name, slm)
        core.initializeDevice(slm_name)
        core.setSLMDevice(slm_name)

    merged_devices: dict[str, dict[str, Optional[Tuple[float, float]]]] = {
        d: dict(p) for d, p in (extra_devices or {}).items()
    }
    if config_data:
        for settings in config_data.values():
            for dev, prop, _value in settings:
                if dev == CAMERA_LABEL:
                    continue
                merged_devices.setdefault(dev, {}).setdefault(prop, None)

    for dev_name, props in merged_devices.items():
        core.loadPyDevice(dev_name, _PropertyHolder(dev_name, props))
        core.initializeDevice(dev_name)

    configured: set[tuple[str, str]] = set()
    if config_data:
        for (group, config_name), settings in config_data.items():
            for device, prop, value in settings:
                core.defineConfig(group, config_name, device, prop, value)
            configured.add((group, config_name))

    group = channel_group if channel_group is not None else DEFAULT_CHANNEL_GROUP
    default_group = group or DEFAULT_CHANNEL_GROUP
    channels = getattr(scene, "channels", ("phase-contrast",))
    for name in channels:
        if (default_group, name) not in configured:
            core.defineConfig(default_group, name, CAMERA_LABEL, "Exposure", "10")

    if extra_configs:
        for g, names in extra_configs.items():
            for n in names:
                if (g, n) not in configured:
                    core.defineConfig(g, n, CAMERA_LABEL, "Exposure", "10")

    if group:
        core.setChannelGroup(group)

    core.mda.events.eventStarted.connect(
        lambda event: setattr(camera, "current_event", event)
    )
    return core


# ---------------------------------------------------------------------------
# Validation-test convenience builder
# ---------------------------------------------------------------------------


class _ValidationScene:
    """Minimal scene for validation tests; ``render`` never fires."""

    image_height = 64
    image_width = 64

    def __init__(self, channels: Iterable[str]):
        self.channels = tuple(channels)

    def render(self, event):
        return np.zeros((self.image_height, self.image_width), dtype=np.uint16)


def build_validation_core(
    *,
    config_groups: Optional[Mapping[str, Iterable[str]]] = None,
    property_limits: Optional[Mapping[Tuple[str, str], Tuple[float, float]]] = None,
    devices: Optional[Mapping[str, Iterable[str]]] = None,
    config_data: Optional[
        Mapping[Tuple[str, str], Iterable[Tuple[str, str, str]]]
    ] = None,
    channel_group: str = "",
) -> UniMMCore:
    """Build a ``UniMMCore`` for ``validate_hardware`` tests.

    Accepts the flatter kwargs that the validation tests natively use
    (``property_limits``, ``devices``, ``config_data``) and translates
    them to :func:`build_core`'s shape. ``property_limits`` entries on
    ``Camera.Exposure`` become ``camera_exposure_limits``; the rest feed
    into ``extra_devices``. ``config_groups`` is split into the first
    group's channels (handled by the scene) plus the rest (via
    ``extra_configs``).
    """
    groups = (
        dict(config_groups)
        if config_groups
        else {"Channel": ["phase-contrast", "DAPI", "membrane"]}
    )
    main_group = next(iter(groups), "Channel")
    main_channels = list(groups.get(main_group, []))
    extras = {g: list(cfgs) for g, cfgs in groups.items() if g != main_group}

    camera_limits: Optional[Tuple[float, float]] = None
    extra_devices: dict[str, dict[str, Optional[Tuple[float, float]]]] = {}
    if devices:
        for dev, props in devices.items():
            if dev == CAMERA_LABEL:
                continue
            extra_devices.setdefault(dev, {}).update({p: None for p in props})
    if property_limits:
        for (dev, prop), lims in property_limits.items():
            if dev == CAMERA_LABEL and prop == "Exposure":
                camera_limits = lims
            else:
                extra_devices.setdefault(dev, {})[prop] = lims

    return build_core(
        _ValidationScene(main_channels),
        camera_exposure_limits=camera_limits,
        extra_devices=extra_devices or None,
        config_data=config_data,
        extra_configs=extras or None,
        channel_group=channel_group,
    )
