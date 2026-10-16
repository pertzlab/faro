"""Reusable test fixtures for faro.

Downstream labs writing custom stimulators / segmentators / trackers /
feature extractors can import these to stand up a minimal pipeline
against :class:`tests.fake_microscope.FakeMicroscope` without pulling
in the full integration-test file.

What's here
-----------

:class:`CircleScene` — two bright circles, optional blank frames,
    optional SLM. Good default for stim/tracking regression tests.

:func:`make_circle_image` — single synthetic frame; use it when you
    want to sanity-check a segmentator or feature extractor without
    running the whole pipeline.

:func:`make_events` — small batch of :class:`~faro.core.data_structures.RTMEvent`
    with one imaging channel plus optional ``stim_frames``.

:func:`make_pipeline` — ImageProcessingPipeline with sensible defaults
    (``OtsuSegmentator`` + your tracker + ``SimpleFE``); accepts a
    stimulator kwarg.

:func:`run_and_wait` / :func:`run_and_wait_long` — run an experiment
    and block until pipeline queues drain. Shuts the analyzer down.

:func:`assert_no_background_errors` — readable failure message listing
    any errors the analyzer recorded on its worker threads.

:func:`tracker` — pytest fixture parametrised over
    ``TrackerTrackpy`` and ``TrackerMotile``. Use in your integration
    tests to exercise both linking backends.
"""

from __future__ import annotations

import numpy as np
import pytest
from useq import MDAEvent

from faro.core.controller import Controller
from faro.core.data_structures import Channel, RTMEvent, SegmentationMethod
from faro.core.pipeline import ImageProcessingPipeline
from faro.feature_extraction.simple import SimpleFE
from faro.segmentation.base import OtsuSegmentator
from faro.stimulation.center_circle import CenterCircle
from faro.tracking.motile_tracker import TrackerMotile
from faro.tracking.trackpy import TrackerTrackpy


def render_disks(
    positions: np.ndarray,
    *,
    img_size: int,
    radius: int,
    value: int,
    offset: tuple[int, int] = (0, 0),
    dtype=np.uint16,
) -> np.ndarray:
    """Render filled disks at each (row, col) position into a blank frame.

    Small helper used by the two synthetic-cell scenes
    (:class:`~tests.test_tracking_accuracy.SyntheticCellScene` and the
    one in :mod:`tests.test_tracking_divisions`). ``offset`` lets a
    caller shift all disks, e.g. for a second imaging channel with a
    registration offset.
    """
    img = np.zeros((img_size, img_size), dtype=dtype)
    yy, xx = np.ogrid[:img_size, :img_size]
    dr, dc = offset
    for r, c in positions:
        img[(yy - (r + dr)) ** 2 + (xx - (c + dc)) ** 2 <= radius**2] = value
    return img


class CrashingStimulator(CenterCircle):
    """Stimulator whose ``get_stim_mask`` always raises.

    Used by failure-path tests to verify the pipeline's crash handling
    unblocks downstream stim-mask consumers instead of deadlocking on
    the per-frame queue timeout.
    """

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        raise RuntimeError("Stimulation crashed!")


IMG_SIZE = 256
CIRCLE1_CENTER = (64, 64)  # (row, col)
CIRCLE1_RADIUS = 20
CIRCLE2_CENTER = (192, 192)
CIRCLE2_RADIUS = 15


@pytest.fixture(
    params=[
        pytest.param(TrackerTrackpy, id="Trackpy"),
        pytest.param(TrackerMotile, id="Motile"),
    ],
)
def tracker(request):
    """Per-test tracker instance; parametrises tests across both backends."""
    return request.param(search_range=50, memory=3)


def make_circle_image() -> np.ndarray:
    """256x256 uint16 frame with two bright circles at fixed positions."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)
    y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    m1 = (y - CIRCLE1_CENTER[0]) ** 2 + (
        x - CIRCLE1_CENTER[1]
    ) ** 2 <= CIRCLE1_RADIUS**2
    m2 = (y - CIRCLE2_CENTER[0]) ** 2 + (
        x - CIRCLE2_CENTER[1]
    ) ** 2 <= CIRCLE2_RADIUS**2
    img[m1] = 50_000
    img[m2] = 50_000
    return img


def make_events(
    n_timepoints: int, *, stim_frames=(), n_channels: int = 1
) -> list[RTMEvent]:
    """Events with optional stim on selected frames."""
    stim_set = set(stim_frames)
    stim_ch = (Channel(config="stim-405", exposure=100),)
    channels = tuple(
        Channel(config="phase-contrast", exposure=50) for _ in range(n_channels)
    )
    return [
        RTMEvent(
            index={"t": t, "p": 0},
            channels=channels,
            stim_channels=stim_ch if t in stim_set else (),
            metadata={},
        )
        for t in range(n_timepoints)
    ]


class CircleScene:
    """Scene plugin for :class:`~tests.fake_microscope.FakeMicroscope`.

    Renders :func:`make_circle_image` every frame, except timepoints in
    ``blank_frames`` (which come back all-zero for empty-FOV tests).
    When ``with_slm=True`` the scene declares an SLM so stim dispatches
    flow through ``setSLMImage`` / ``displaySLMImage``; each dispatched
    mask is appended to ``slm_events`` as ``(frame_idx, ndarray)``.
    """

    image_height = IMG_SIZE
    image_width = IMG_SIZE
    channels = ("phase-contrast", "stim-405")

    def __init__(self, *, blank_frames: set[int] = frozenset(), with_slm: bool = False):
        self.blank_frames = set(blank_frames)
        self.slm_events: list[tuple[int, np.ndarray]] = []
        self.slm_name = "SLM" if with_slm else None
        self.slm_shape = (IMG_SIZE, IMG_SIZE) if with_slm else None

    def render(self, event: MDAEvent) -> np.ndarray:
        if event.index.get("t", 0) in self.blank_frames:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)
        return make_circle_image()

    def on_slm_displayed(self, image: np.ndarray, event: MDAEvent) -> None:
        self.slm_events.append((event.index.get("t", 0), image))


def make_pipeline(
    storage_path: str,
    *,
    tracker,
    stimulator=None,
    with_stim: bool = False,
    save_tracked: bool = False,
) -> ImageProcessingPipeline:
    """Pipeline with OtsuSegmentator + given tracker + SimpleFE.

    ``with_stim=True`` attaches a default :class:`CenterCircle`
    stimulator; pass ``stimulator=`` directly for anything custom.
    """
    stim = (
        stimulator
        if stimulator is not None
        else (CenterCircle() if with_stim else None)
    )
    return ImageProcessingPipeline(
        storage_path=storage_path,
        segmentators=[SegmentationMethod("labels", OtsuSegmentator(), 0, save_tracked)],
        tracker=tracker,
        feature_extractor=SimpleFE("labels"),
        stimulator=stim,
    )


def assert_no_background_errors(ctrl: Controller) -> None:
    """Fail with a readable summary if the analyzer recorded any errors."""
    if ctrl.background_errors:
        summary = "\n".join(
            f"  [{e.source}] {e.exc_type}: {e.message}" for e in ctrl.background_errors
        )
        raise AssertionError(f"Background errors during acquisition:\n{summary}")


def run_and_wait(
    ctrl: Controller, events: list[RTMEvent], stim_mode: str = "current"
) -> None:
    """Run an experiment and block until the analyzer idles. Then shut down.

    ``run_experiment`` is non-blocking and creates the Analyzer on its
    worker thread, so wait for the run to finish before touching
    ``ctrl._analyzer``. Without the wait this races the worker and loses
    on a loaded machine.
    """
    ctrl.run_experiment(events, stim_mode=stim_mode, validate=False).wait()
    ctrl._analyzer.wait_idle()
    ctrl._analyzer.shutdown(wait=True)


def run_and_wait_long(
    ctrl: Controller,
    events: list[RTMEvent],
    stim_mode: str = "current",
    timeout: float = 120,
) -> None:
    """Like :func:`run_and_wait` with a longer drain timeout for slow pipelines."""
    ctrl.run_experiment(events, stim_mode=stim_mode, validate=False).wait()
    ctrl._analyzer.wait_idle(timeout=timeout)
    ctrl._analyzer.shutdown(wait=True)
