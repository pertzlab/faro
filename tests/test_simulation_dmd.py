"""``UniMMCoreSimulation`` wraps the virtual scope's SLM in the real ``DMD``.

Runs faro's calibration routine against the teaching simulator, whose
projector can be misaligned via ``sim.slm_affine``, and checks that the
recovered transform puts masks where they were drawn. Requires ``vmteach``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import skimage.draw

pytest.importorskip("vmteach")

from faro.core.data_structures import Channel, RTMSequence  # noqa: E402
from faro.core.dmd import DMD  # noqa: E402
from faro.microscope.simulation import UniMMCoreSimulation  # noqa: E402

STIM = {"config": "CyanStim", "exposure": 50}


def _scope(slm_affine=None):
    from vmteach import load_microscope

    core, sim = load_microscope("optogenetic", seed=0, mode="stepped")  # deterministic
    sim.slm_affine = slm_affine
    mic = UniMMCoreSimulation(mmc=core)
    mic.init_scope()
    return mic, sim


def _target():
    t = np.zeros((512, 512), np.uint8)
    for r, c in ((128, 128), (256, 300), (400, 180)):
        rr, cc = skimage.draw.disk((r, c), 14)
        t[rr, cc] = 255
    return t


def _max_reprojection_error(affine, truth_2x3):
    """Largest distance, in px, between where ``affine`` and the true
    projector put a grid of DMD points across the field."""
    grid = np.array([(x, y, 1.0) for x in (60, 256, 450) for y in (60, 256, 450)])
    got = (grid @ affine.T)[:, :2]
    want = grid @ np.asarray(truth_2x3, float).T
    return np.linalg.norm(got - want, axis=1).max()


def _iou_of_projected_light(mic, dmd_mask, target):
    core = mic.mmc
    core.setSLMImage(mic.dmd.name, dmd_mask)
    core.setConfig("Channel", STIM["config"])
    core.setExposure(STIM["exposure"])
    core.snapImage()
    lit = core.getImage() > 120
    want = target > 0
    return (lit & want).sum() / (lit | want).sum()


def test_init_scope_attaches_uncalibrated_dmd():
    mic, _ = _scope()
    assert isinstance(mic.dmd, DMD)
    assert mic.dmd.affine is None


def test_validate_events_warns_until_calibrated():
    mic, _ = _scope()
    events = list(
        RTMSequence(
            time_plan={"interval": 1.0, "loops": 2},
            stage_positions=[{"x": 0.0, "y": 0.0, "z": 0.0}],
            channels=[{"config": "miRFP", "exposure": 50}],
            stim_channels=[STIM],
            stim_frames=range(2),
        )
    )
    with pytest.warns(UserWarning, match="DMD not calibrated"):
        assert mic.validate_hardware(events) is False
    mic.calibrate_dmd(STIM)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mic.validate_hardware(events) is True


@pytest.mark.parametrize("channel", [STIM, STIM["config"], Channel(**STIM)])
def test_calibrate_dmd_accepts_dict_name_or_channel(channel):
    mic, _ = _scope()
    mic.calibrate_dmd(channel)
    # Spots are detected on integer pixels, so allow sub-pixel slack.
    assert _max_reprojection_error(mic.dmd.affine, np.eye(3)[:2]) < 1.0


def test_calibration_recovers_misaligned_projector():
    th, s = np.deg2rad(4), 0.92
    truth = np.array(
        [[s * np.cos(th), -s * np.sin(th), 30.0], [s * np.sin(th), s * np.cos(th), -18.0]]
    )
    mic, _ = _scope(slm_affine=truth)
    target = _target()
    mic.calibrate_dmd(STIM)

    # a few px over the field: spot centroids land on integer pixels, and
    # the simulated widefield glow and hot pixels shift them slightly
    assert _max_reprojection_error(mic.dmd.affine, truth) < 3.0

    # An uncalibrated (identity) mask lands off target; the calibrated one lands on it.
    assert _iou_of_projected_light(mic, target, target) < 0.2
    assert _iou_of_projected_light(mic, mic.dmd.affine_transform(target), target) > 0.85
