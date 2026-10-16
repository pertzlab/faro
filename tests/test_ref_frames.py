"""Reference frames on the deferred pipeline path.

A reference frame is stored in two parts: its imaging channels as "raw" and
its reference channels as "ref". When the pipeline is overloaded, a frame is
reloaded from storage later; the reload must give the pipeline both parts.
"""

from __future__ import annotations

import logging

import numpy as np
from useq import MDAEvent

from faro.core.controller import Analyzer
from faro.core.data_structures import ImgType
from faro.core.writers import TiffWriter
from faro.feature_extraction.base_ref import FeatureExtractorRef
from faro.tracking.trackpy import TrackerTrackpy

from tests.fixtures import make_circle_image, make_pipeline

H, W = make_circle_image().shape


def _metadata(img_type):
    return {
        "fov": 0,
        "timestep": 0,
        "fname": "000_00000",
        "img_type": img_type,
        "channels": ["phase-contrast", "fitc"],
        "ref_channels": ["ref"],
        "stim": False,
    }


class TestReloadFrame:
    def test_reference_frame_gets_its_reference_channels_back(self, tmp_dir):
        writer = TiffWriter(tmp_dir)
        raw = np.full((2, H, W), 1, dtype=np.uint16)
        ref = np.full((1, H, W), 7, dtype=np.uint16)
        meta = _metadata(ImgType.IMG_REF)
        writer.write(raw, meta, "raw")
        writer.write(ref, meta, "ref")

        analyzer = Analyzer(writer=writer)
        try:
            img = analyzer._reload_frame(meta)
            assert img.shape == (3, H, W)  # imaging channels, then reference
            np.testing.assert_array_equal(img[:2], raw)
            np.testing.assert_array_equal(img[2], ref[0])
            # A normal frame is only its imaging channels.
            raw_meta = {**meta, "img_type": ImgType.IMG_RAW}
            assert analyzer._reload_frame(raw_meta).shape == (2, H, W)
        finally:
            analyzer.shutdown(wait=False)


class SpyRefFE(FeatureExtractorRef):
    def __init__(self):
        super().__init__("labels")
        self.calls = 0

    def extract_features(self, segmentation_results, image, df_tracked, metadata):
        self.calls += 1
        return df_tracked


class TestReferenceChannelsMissing:
    def test_pipeline_skips_the_reference_features(self, tmp_dir, caplog):
        """The imaging channels must not be measured as the reference."""
        pipeline = make_pipeline(tmp_dir, tracker=TrackerTrackpy(search_range=50))
        spy = SpyRefFE()
        pipeline.feature_extractor_ref = spy
        analyzer = Analyzer(pipeline, writer=TiffWriter(tmp_dir))
        try:
            meta = _metadata(ImgType.IMG_REF)
            meta["channels"] = ["phase-contrast"]
            event = MDAEvent(index={"t": 0, "p": 0}, metadata=meta)
            only_imaging = make_circle_image()[np.newaxis]  # no reference channel
            with caplog.at_level(logging.WARNING, logger="faro.core.pipeline"):
                pipeline.run(img=only_imaging, event=event)
            assert spy.calls == 0
            assert "has no reference channels" in caplog.text
        finally:
            analyzer.shutdown(wait=False)
