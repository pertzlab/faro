"""Drop-in ``PyMMCoreMicroscope`` backed by a real ``UniMMCore`` + Python devices.

The Controller drives the Python devices through the real pymmcore-plus
MDA engine, so tests exercise actual ``mmc.run_mda``, the ``frameReady``
signal chain, ``setSLMImage`` / ``displaySLMImage`` dispatch, and
``validate_hardware`` queries.

Usage::

    scene = MyScene(...)
    mic = FakeMicroscope(scene)
    ctrl = Controller(mic, pipeline)
    ctrl.run_experiment(events)

Scene-specific state lives on ``mic.scene``.
"""

from __future__ import annotations

from faro.microscope.pymmcore import PyMMCoreMicroscope

from faro.testing.fake_mmc import Scene, build_core


class SimDMD:
    """Identity DMD stand-in: camera and SLM share one coordinate space, so
    ``affine_transform`` is the identity and no calibration is needed."""

    def __init__(self, name: str):
        self.name = name
        self.affine = True  # always "calibrated"

    def affine_transform(self, mask):
        return mask


class FakeMicroscope(PyMMCoreMicroscope):
    """``PyMMCoreMicroscope`` backed by ``UniMMCore`` and a :class:`Scene`.

    If the scene declares an SLM (``slm_name`` set), a :class:`SimDMD`
    is attached so the Controller's stim branch runs; the scene's
    ``on_slm_displayed`` fires with each dispatched mask.
    """

    def __init__(self, scene: Scene) -> None:
        super().__init__()
        self.scene = scene
        self.mmc = build_core(scene)
        if getattr(scene, "slm_name", None):
            self.dmd = SimDMD(scene.slm_name)

    def init_scope(self) -> None:
        pass

    def post_experiment(self) -> None:
        pass
