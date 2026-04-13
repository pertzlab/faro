"""Regression tests for the FrameDispenser interaction with stim modes.

The existing TestEndToEndStim* classes in test_pipeline_integration.py do not
exercise the consumer path because their fake CircleMicroscope has
``self.dmd is None`` — the controller's stim-event branch (``if self._mic.dmd``)
is never taken, so ``_build_stim_slm`` never runs and no
``stim_mask_queue.get_at_frame`` is called. With the FrameDispenser fix
(``get_at_frame(t)`` waits for *exactly* frame t's mask), ``previous`` mode
would deadlock on a real microscope unless ``_build_stim_slm`` requests
frame ``t-1`` instead.

These tests exercise the dispenser semantics directly, so the bug would have
been caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from useq import MDAEvent

from faro.core.controller import Analyzer, Controller
from faro.core.data_structures import (
    Channel,
    FovState,
    PowerChannel,
    RTMEvent,
)
from faro.core.pipeline import ImageProcessingPipeline
from faro.feature_extraction.simple import SimpleFE
from faro.microscope.base import AbstractMicroscope
from faro.segmentation.base import OtsuSegmentator
from faro.stimulation.base import StimWithPipeline
from faro.stimulation.center_circle import CenterCircle
from faro.tracking.trackpy import TrackerTrackpy


class _FakeDMD:
    """Minimal DMD stand-in: enough for ``Controller._build_stim_slm`` to run."""

    name = "FakeDMD"

    def affine_transform(self, img):
        return img


class _FakeMicroscope(AbstractMicroscope):
    """Microscope with a DMD attribute set, so the stim branch fires."""

    def __init__(self):
        super().__init__()
        self.dmd = _FakeDMD()


def _stim_event(t: int, fov: int = 0) -> RTMEvent:
    """RTMEvent with one imaging channel and one stim channel."""
    return RTMEvent(
        index={"t": t, "p": fov},
        channels=(Channel(config="img", exposure=10),),
        stim_channels=(PowerChannel(config="stim", exposure=10, power=10),),
        metadata={},
    )


@pytest.fixture
def ctrl_and_state():
    """A minimal Controller + populated FovState with masks at frames 2, 3."""
    pipeline = ImageProcessingPipeline(
        storage_path="/tmp/_unused_dispenser_test",
        segmentators=None,
        feature_extractor=None,
        tracker=None,
        stimulator=CenterCircle(),  # StimWithPipeline subclass
    )
    analyzer = Analyzer(pipeline=pipeline)
    ctrl = Controller(_FakeMicroscope(), pipeline)
    ctrl._analyzer = analyzer

    fov = analyzer.get_fov_state(0)
    # Pre-populate the dispenser with masks for frames 2 and 3,
    # mark frames 0 and 1 as skipped (non-stim frames in the experiment).
    fov.stim_mask_queue.skip_frame(0)
    fov.stim_mask_queue.skip_frame(1)
    fov.stim_mask_queue.put_for_frame(2, np.full((4, 4), 2, dtype=np.uint8))
    fov.stim_mask_queue.put_for_frame(3, np.full((4, 4), 3, dtype=np.uint8))

    yield ctrl, fov

    analyzer.shutdown(wait=False)


def test_current_mode_uses_frame_t_mask(ctrl_and_state):
    """Default frame_offset=0 → consumer asks for frame t's mask."""
    ctrl, _ = ctrl_and_state
    rtm_event = _stim_event(t=3)
    slm = ctrl._build_stim_slm(rtm_event)
    # mask_3 is filled with 3
    assert isinstance(slm.data, np.ndarray)
    assert slm.data[0, 0] == 3


def test_previous_mode_uses_frame_t_minus_1_mask(ctrl_and_state):
    """frame_offset=-1 → consumer asks for frame t-1's mask."""
    ctrl, _ = ctrl_and_state
    rtm_event = _stim_event(t=3)
    slm = ctrl._build_stim_slm(rtm_event, frame_offset=-1)
    # mask_2 is filled with 2
    assert slm.data[0, 0] == 2


def test_previous_mode_falls_back_to_no_stim_when_predecessor_skipped(ctrl_and_state):
    """If frame t-1 was skipped (non-stim frame), ``False`` is sent to the SLM."""
    ctrl, _ = ctrl_and_state
    rtm_event = _stim_event(t=1)  # t-1 = 0 was skipped
    slm = ctrl._build_stim_slm(rtm_event, frame_offset=-1)
    assert slm.data is False
