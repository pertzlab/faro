"""Stim-related pipeline integration tests.

Covers the stim branch of the Controller end-to-end under both
``stim_mode="current"`` (mask from the fresh frame) and
``stim_mode="previous"`` (mask from the previous frame). Also covers
the three stimulator shortcut paths (:class:`Stim`,
:class:`StimWithImage`, :class:`StimWithPipeline`) and the Analyzer's
stim-mask-timeout background-error recording.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import tifffile

from faro.core.controller import Analyzer, Controller
from faro.core.data_structures import Channel, RTMEvent
from faro.stimulation.base import Stim, StimWithImage, StimWithPipeline

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import (
    CircleScene,
    CrashingStimulator,
    make_events,
    make_pipeline as _make_pipeline,
    run_and_wait,
    tracker,  # noqa: F401 — parametrized fixture, auto-discovered by pytest
)


N_TIMEPOINTS = 5


# ===================================================================
# Test stimulators and helpers (shared across stim-mode tests below)
# ===================================================================


class TestEndToEndStimCurrent:
    """5 timepoints, stim on frames 2-4, mode='current'."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, tracker=tracker, with_stim=True)
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_stim_masks_saved(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_TIMEPOINTS
        ), f"Expected {N_TIMEPOINTS} stim mask TIFFs, got {len(files)}"

    def test_stim_masks_nonzero_for_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert mask.max() > 0, f"Frame {i} should have nonzero stim mask"
            else:
                assert mask.max() == 0, f"Frame {i} should have zero stim mask"

    def test_segmentation_still_correct(self):
        labels_dir = os.path.join(self.path, "labels")
        files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".tiff")])
        assert len(files) == N_TIMEPOINTS
        for f in files:
            labels = tifffile.imread(os.path.join(labels_dir, f))
            unique = set(np.unique(labels)) - {0}
            assert len(unique) == 2, f"{f}: expected 2 labels, got {len(unique)}"


# ===================================================================
# Test Class 4: End-to-end, stim mode="previous"
# ===================================================================


class TestEndToEndStimPrevious:
    """5 timepoints, stim on frames 2-4, mode='previous'."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(self.path, tracker=tracker, with_stim=True)
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="previous")

    def test_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        assert os.path.isdir(stim_dir)
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert (
            len(files) == N_TIMEPOINTS
        ), f"Expected {N_TIMEPOINTS} stim mask files, got {len(files)}"

    def test_segmentation_works(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_consistent(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2, f"Expected 2 particles, got {len(particles)}"
        for pid in particles:
            rows = df[df["particle"] == pid]
            assert len(rows) == N_TIMEPOINTS


class MetadataOnlyStim(Stim):
    """Base Stim — returns all-ones mask using only metadata["img_shape"]."""

    def get_stim_mask(self, metadata: dict):
        h, w = metadata["img_shape"]
        return np.ones((h, w), dtype=np.uint8), None


class ImageBasedStim(StimWithImage):
    """StimWithImage — thresholds the raw image to build a stim mask."""

    def get_stim_mask(self, metadata: dict, img: np.ndarray):
        # Use first channel, threshold at half-max
        frame = img[0] if img.ndim == 3 else img
        thresh = frame.max() / 2
        mask = (frame > thresh).astype(np.uint8)
        return mask, None


class _FrameTaggingStim(StimWithPipeline):
    """StimWithPipeline that returns a mask whose pixel value = current timestep.

    Used by :class:`TestStimModeMaskSelectionCurrent` /
    :class:`TestStimModeMaskSelectionPrevious` so the SLM image delivered
    to each stim event can be traced back to the frame that produced it.
    """

    required_metadata: set[str] = {"img_shape", "timestep"}

    def get_stim_mask(
        self,
        *,
        label_images,
        metadata: dict,
        img=None,
        tracks=None,
    ):
        h, w = metadata["img_shape"]
        t = metadata["timestep"]
        return np.full((h, w), t, dtype=np.uint8), None


def _sole_mask_value(mask: np.ndarray) -> int:
    """Return the single tag value of a _FrameTaggingStim mask.

    ``_FrameTaggingStim`` fills the whole mask with ``metadata["timestep"]``,
    so every pixel is equal. Assert that invariant and return the value.
    """
    unique = np.unique(mask)
    assert unique.size == 1, f"expected uniform mask, got values {unique}"
    return int(unique.item())


# ===================================================================
# End-to-end stim-mode mask-selection tests (CircleScene with_slm=True)
#
# These classes actually exercise Controller._build_stim_slm and the
# stim_mask_queue consumer by enabling the scene's SLM. The
# _FrameTaggingStim returns a mask whose pixel value = the frame that
# produced it, so asserting on scene.slm_events gives a
# direct "event at frame t received mask from frame X" check.
# ===================================================================


class TestStimModeMaskSelectionCurrent:
    """``current`` mode: stim at frame t must receive frame t's mask."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=_FrameTaggingStim()
        )
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, events, stim_mode="current")

    def test_stim_events_received_correct_masks(self):
        # One SLM image per stim event; each carries its own frame's mask.
        assert len(self.mic.scene.slm_events) == len(self.STIM_FRAMES)
        for (event_t, slm_image), expected in zip(
            self.mic.scene.slm_events, self.STIM_FRAMES
        ):
            assert event_t == expected
            assert _sole_mask_value(slm_image) == expected, (
                f"Frame {event_t} received mask tagged "
                f"{_sole_mask_value(slm_image)}, expected {expected}"
            )


class TestStimModeMaskSelectionPrevious:
    """``previous`` mode: stim at frame t must receive frame t-1's mask.

    The first stim frame (frame 2) does not actually fire — the controller's
    ``_stim_pending`` guard skips it because no previous stim frame exists
    for this FOV yet — so SLM events land on frames 3, 4 with masks from
    frames 2, 3 respectively.
    """

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=_FrameTaggingStim()
        )
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, events, stim_mode="previous")

    def test_all_stim_events_fire_with_predecessor_mask(self):
        """Every stim event fires, including the first — the pipeline always
        computes a mask in previous mode, even for non-stim frames, so the
        first stim frame's t-1 peek finds the prior computed mask.
        """
        # Event frame → mask source frame under "previous" semantics.
        expected = [(2, 1), (3, 2), (4, 3)]
        assert len(self.mic.scene.slm_events) == len(expected)
        for (event_t, slm_image), (exp_event, exp_mask) in zip(
            self.mic.scene.slm_events, expected
        ):
            assert event_t == exp_event
            assert _sole_mask_value(slm_image) == exp_mask, (
                f"Stim event at frame {event_t} received mask from frame "
                f"{_sole_mask_value(slm_image)}, expected {exp_mask} "
                f"(previous-frame semantics)"
            )


class TestStimModeCurrentAtFrameZero:
    """Symmetric edge case to ``previous`` at frame 0: ``current`` at t=0 must
    work (stim fires after t=0's own pipeline finishes and puts mask_0).
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=_FrameTaggingStim()
        )
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        events = make_events(3, stim_frames=(0,))
        run_and_wait(self.ctrl, events, stim_mode="current")

    def test_frame_zero_stim_fires_with_own_mask(self):
        assert len(self.mic.scene.slm_events) == 1
        event_t, slm_image = self.mic.scene.slm_events[0]
        assert event_t == 0
        assert _sole_mask_value(slm_image) == 0


def _make_multi_fov_events(
    n_timepoints: int, *, stim_frames_per_fov: dict[int, tuple[int, ...]]
) -> list[RTMEvent]:
    """Build events interleaved across multiple FOVs.

    ``stim_frames_per_fov`` maps ``fov_index`` → set of timepoints that carry
    a stim channel for that FOV. Events are ordered by ``(t, p)``.
    """
    stim_ch = (Channel(config="stim-405", exposure=100),)
    events = []
    for t in range(n_timepoints):
        for fov in sorted(stim_frames_per_fov):
            has_stim = t in stim_frames_per_fov[fov]
            events.append(
                RTMEvent(
                    index={"t": t, "p": fov},
                    channels=(Channel(config="phase-contrast", exposure=50),),
                    stim_channels=stim_ch if has_stim else (),
                    metadata={},
                )
            )
    return events


class TestStimModePreviousMultiFov:
    """Each FOV maintains independent dispenser state, so stim events in FOV 1
    must never consume a mask produced by FOV 0 (or vice versa). With
    always-compute, every stim event fires — including the first one on each
    FOV.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=_FrameTaggingStim()
        )
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        # FOV 0: stim on frames (2, 3, 4); FOV 1: stim on frames (3, 4).
        events = _make_multi_fov_events(
            5, stim_frames_per_fov={0: (2, 3, 4), 1: (3, 4)}
        )
        run_and_wait(self.ctrl, events, stim_mode="previous")

    def test_each_fov_uses_its_own_previous_mask(self):
        # _FrameTaggingStim tags by timestep only, and each FOV has its own
        # dispenser. FOV 0 fires on t=(2,3,4) with masks from (1,2,3); FOV 1
        # fires on t=(3,4) with masks from (2,3).
        events_by_tag = sorted(
            (t, _sole_mask_value(slm)) for t, slm in self.mic.scene.slm_events
        )
        assert events_by_tag == [(2, 1), (3, 2), (3, 2), (4, 3), (4, 3)]


class TestStimModePreviousAtFrameZero:
    """Edge case: ``previous`` mode at frame 0 has no t-1.

    The controller passes ``suppress_stim=True`` to ``plan_events`` when
    ``stim_mode == "previous"`` and ``t == 0``, so the stim event is
    never queued. Firing a blank mask would still activate the DMD
    (mirror bleed-through leaks ~1% of nominal intensity), so outright
    suppression is the only way to guarantee zero stim at t=0.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=MetadataOnlyStim()
        )
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        events = make_events(2, stim_frames=(0,))
        run_and_wait(self.ctrl, events, stim_mode="previous")

    def test_no_stim_at_frame_0(self):
        assert self.mic.scene.slm_events == []


class TestStimModePreviousPipelineCrashDoesNotDeadlock:
    """If frame t's pipeline crashes inside the stim branch, the try/finally
    must call ``skip_frame`` on stim_mask_queue so frame t+1's "previous"-mode
    consumer sees a skipped predecessor (None) rather than blocking until
    the stim-mask timeout. Reuses the existing ``CrashingStimulator``.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=CrashingStimulator()
        )
        # Short queue timeout: without this, each pipeline stim-mask
        # wait would block for the full 20 s default, doubling test
        # runtime to ~40 s per parametrize instance.
        self.pipeline._queue_timeout = 1
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        events = make_events(4, stim_frames=(2, 3))
        run_and_wait(self.ctrl, events, stim_mode="previous")

    def test_all_stim_events_gracefully_fall_back(self):
        # CrashingStimulator crashes on every call — including the always-
        # compute path on non-stim frames. So every frame's pipeline skips
        # the dispenser, and every stim event's t-1 peek returns None →
        # SLMImage(data=False). Neither frame's stim actually fires, but
        # neither deadlocks either.
        assert len(self.mic.scene.slm_events) == 2
        for event_t, slm in self.mic.scene.slm_events:
            assert event_t in (2, 3)
            assert not slm.any()

    def test_fallbacks_are_counted(self):
        # Every all-off fallback bumps the counter that _build_stim_slm
        # stamps into RunStatus as it happens.
        assert self.ctrl._n_stim_fallbacks == 2

    def test_warnings_land_in_experiment_log(self):
        # The run mirrors faro log records into log.txt in the experiment
        # folder, so the fallback warnings survive the console.
        content = open(os.path.join(self.path, "log.txt"), encoding="utf-8").read()
        assert content.count("ALL-OFF") == 2

    def test_log_handler_detached_after_run(self):
        # Once the MDA finished, nothing may keep writing into the
        # experiment folder: records on the "faro" and "pymmcore-plus"
        # loggers go only to the host's own handlers (stderr / pymmcore
        # logfile), not log.txt.
        log_path = os.path.join(self.path, "log.txt")
        size_before = os.path.getsize(log_path)
        # Any logger under the "faro" namespace stands in for the modules
        # (controller, microscopes) that log during a run.
        logging.getLogger("faro.microscope").warning(
            "post-run record, must not reach log.txt"
        )
        logging.getLogger("pymmcore-plus").warning(
            "post-run record, must not reach log.txt"
        )
        assert os.path.getsize(log_path) == size_before


class _CountingStim(StimWithPipeline):
    """StimWithPipeline that records every ``get_stim_mask`` call so tests
    can assert pipelines don't compute on frames that won't consume the result.

    ``call_timesteps`` is appended from the pipeline worker thread and read by
    the test thread after ``run_and_wait`` joins — that join establishes the
    happens-before needed for safe reads; ``list.append`` is atomic in CPython.
    """

    required_metadata: set[str] = {"img_shape"}

    def __init__(self):
        self.call_timesteps: list[int] = []

    def get_stim_mask(self, *, label_images, metadata: dict, img=None, tracks=None):
        self.call_timesteps.append(metadata.get("timestep", -1))
        h, w = metadata["img_shape"]
        return np.zeros((h, w), dtype=np.uint8), None


class TestCurrentModeSkipsComputeOnNonStim:
    """In ``current`` mode, no one peeks a non-stim frame's mask, so the
    pipeline must not invoke the stimulator on non-stim frames — that's
    wasted work.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.stim = _CountingStim()
        self.pipeline = _make_pipeline(self.path, tracker=tracker, stimulator=self.stim)
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        events = make_events(5, stim_frames=(2, 3))
        run_and_wait(self.ctrl, events, stim_mode="current")

    def test_stim_computed_only_on_stim_frames(self):
        assert sorted(self.stim.call_timesteps) == [2, 3]


class TestPreviousModeComputesOnEveryFrame:
    """In ``previous`` mode, every frame must compute so the next frame's
    t-1 peek has something to read.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.stim = _CountingStim()
        self.pipeline = _make_pipeline(self.path, tracker=tracker, stimulator=self.stim)
        self.mic = FakeMicroscope(CircleScene(with_slm=True))
        self.ctrl = Controller(self.mic, self.pipeline)
        events = make_events(5, stim_frames=(2, 3))
        run_and_wait(self.ctrl, events, stim_mode="previous")

    def test_stim_computed_on_every_frame(self):
        assert sorted(self.stim.call_timesteps) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "stim_mode, tag_offset",
    [("current", 0), ("previous", -1)],
)
class TestStimMaskFileReflectsFired:
    """The stim_mask .tiff under frame t stores what actually fired at t.

    ``current`` mode → offset 0 (stored = mask computed at t).
    ``previous`` mode → offset -1 (stored = mask computed at t-1).
    Non-stim frames always store zeros.
    """

    STIM_FRAMES = (2, 3)
    N_FRAMES = 5

    def test_stored_mask_is_fired_mask(self, tmp_dir, tracker, stim_mode, tag_offset):
        pipeline = _make_pipeline(
            tmp_dir, tracker=tracker, stimulator=_FrameTaggingStim()
        )
        ctrl = Controller(FakeMicroscope(CircleScene(with_slm=True)), pipeline)
        events = make_events(self.N_FRAMES, stim_frames=self.STIM_FRAMES)
        run_and_wait(ctrl, events, stim_mode=stim_mode)

        stim_dir = os.path.join(tmp_dir, "stim_mask")
        for t in range(self.N_FRAMES):
            mask = tifffile.imread(os.path.join(stim_dir, f"000_{t:05d}.tiff"))
            if t in self.STIM_FRAMES:
                expected = t + tag_offset
                unique = np.unique(mask)
                assert unique.size == 1 and int(unique.item()) == expected, (
                    f"[{stim_mode}] frame {t} should store mask from frame "
                    f"{expected}, got {unique.tolist()}"
                )
            else:
                assert mask.max() == 0, (
                    f"[{stim_mode}] frame {t} (non-stim) should store zeros, "
                    f"got {mask.max()}"
                )


# ===================================================================
# Test Class 5: Stim (metadata-only) shortcut — mode="current"
# ===================================================================


class TestStimMetadataOnlyCurrent:
    """Base Stim bypasses pipeline — mask computed synchronously in controller."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=MetadataOnlyStim()
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_stim_masks_saved(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_stim_masks_nonzero_for_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert (
                    mask.max() > 0
                ), f"Frame {i}: metadata-only stim should produce nonzero mask"
            else:
                assert (
                    mask.max() == 0
                ), f"Frame {i}: non-stim frame should have zero mask"

    def test_segmentation_still_runs(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_still_works(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2


# ===================================================================
# Test Class 6: Stim (metadata-only) shortcut — mode="previous"
# ===================================================================


class TestStimMetadataOnlyPrevious:
    """Base Stim in 'previous' mode — still computes synchronously."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=MetadataOnlyStim()
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="previous")

    def test_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_consistent(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2
        for pid in particles:
            assert len(df[df["particle"] == pid]) == N_TIMEPOINTS


# ===================================================================
# Test Class 7: StimWithImage shortcut — mode="current"
# ===================================================================


class TestStimWithImageCurrent:
    """StimWithImage — mask computed in storage worker before pipeline."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=ImageBasedStim()
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="current")

    def test_stim_masks_saved(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_stim_masks_nonzero_for_stim_frames(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = sorted(os.listdir(stim_dir))
        for i, f in enumerate(files):
            mask = tifffile.imread(os.path.join(stim_dir, f))
            if i in self.STIM_FRAMES:
                assert (
                    mask.max() > 0
                ), f"Frame {i}: image-based stim should produce nonzero mask"
            else:
                assert (
                    mask.max() == 0
                ), f"Frame {i}: non-stim frame should have zero mask"

    def test_segmentation_still_runs(self):
        labels_dir = os.path.join(self.path, "labels")
        files = [f for f in os.listdir(labels_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_still_works(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2


# ===================================================================
# Test Class 8: StimWithImage shortcut — mode="previous"
# ===================================================================


class TestStimWithImagePrevious:
    """StimWithImage in 'previous' mode — mask from storage worker, used next frame."""

    STIM_FRAMES = (2, 3, 4)

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir, tracker):
        self.path = tmp_dir
        self.pipeline = _make_pipeline(
            self.path, tracker=tracker, stimulator=ImageBasedStim()
        )
        self.mic = FakeMicroscope(CircleScene())
        self.ctrl = Controller(self.mic, self.pipeline)
        self.events = make_events(N_TIMEPOINTS, stim_frames=self.STIM_FRAMES)
        run_and_wait(self.ctrl, self.events, stim_mode="previous")

    def test_stim_masks_produced(self):
        stim_dir = os.path.join(self.path, "stim_mask")
        files = [f for f in os.listdir(stim_dir) if f.endswith(".tiff")]
        assert len(files) == N_TIMEPOINTS

    def test_tracking_consistent(self):
        tracks_dir = os.path.join(self.path, "tracks")
        parquet_files = [f for f in os.listdir(tracks_dir) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(tracks_dir, parquet_files[0]))
        particles = df["particle"].unique()
        assert len(particles) == 2
        for pid in particles:
            assert len(df[df["particle"] == pid]) == N_TIMEPOINTS


# ===================================================================
# Stress test helpers
# ===================================================================


class TestStimMaskTimeout:
    """Analyzer.get_stim_mask records a background error when the queue times out.

    Regression guard for the silent-stim-frame bug: on the first stim
    frame, the pipeline hadn't produced a mask yet, get_stim_mask timed
    out, and the fallback silently sent False to the SLM with nothing
    recorded. Hardware tests (which assert background_errors == [])
    should fail loudly in that case.
    """

    @pytest.fixture
    def analyzer(self, tmp_dir, tracker) -> Iterator[Analyzer]:
        pipeline = _make_pipeline(tmp_dir, tracker=tracker, stimulator=ImageBasedStim())
        instance = Analyzer(pipeline=pipeline)
        yield instance
        instance.shutdown(wait=True)

    def test_timeout_records_background_error(self, analyzer):
        result = analyzer.get_stim_mask(fov_index=7, metadata={}, timeout=0.05)
        assert result is None
        assert len(analyzer.background_errors) == 1
        err = analyzer.background_errors[0]
        assert err.source == "stim_mask"
        assert err.exc_type == "TimeoutError"
        assert "FOV 7" in err.message
        assert "0.05" in err.message
        # Traceback should reflect the TimeoutError raise site, not the
        # bare QueueEmpty — regression guard for the format_exc() bug.
        assert "TimeoutError" in err.traceback

    def test_ready_mask_returns_value_without_recording(self, analyzer):
        """Happy path: a mask already in the queue is returned, no error recorded."""
        mask = np.ones((32, 32), dtype=np.uint8)
        analyzer.get_fov_state(0).stim_mask_queue.put_for_frame(0, mask)
        result = analyzer.get_stim_mask(
            fov_index=0, metadata={"timestep": 0}, timeout=5.0
        )
        assert result is mask
        assert analyzer.background_errors == []


# ===================================================================
# Test Class: empty-frame edge cases (no cells detected)
# ===================================================================
