"""Tests for ``faro.core.writers``.

Exercises the writer API directly (no Controller, no microscope), with a
focus on schema invariants that downstream analysis scripts depend on:

* TiffWriter: filename convention (``<folder>/<fname>.tiff``), lazy
  folder creation, round-trip of written bytes.
* OmeZarrWriter single-position: direct store path, ``(t, c, y, x)``, label-group
  creation, stim-channel routing.
* OmeZarrWriter multi-position: direct store path, ``(t, p, c, y, x)``
  axis layout, multiscales metadata.
* OmeZarrWriterPlate: plate layout with rows/columns/wells.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from faro.core.data_structures import Channel, RTMEvent
from faro.core.writers import (
    OmeZarrRawReader,
    OmeZarrWriter,
    OmeZarrWriterPlate,
    TiffWriter,
    repair_ome_zarr_labels,
)


IMG_H, IMG_W = 64, 64
N_T = 4
IMG_CHANNELS = ["phase-contrast", "fitc"]
STIM_CHANNELS = 1


def _raw(t: int, p: int, *, n_channels: int = len(IMG_CHANNELS)) -> np.ndarray:
    """Multi-channel raw frame tagged with ``t, p`` for identifiability."""
    arr = np.zeros((n_channels, IMG_H, IMG_W), dtype=np.uint16)
    arr[..., t % IMG_H, p % IMG_W] = 50_000 + t * 100 + p
    return arr


def _mask(t: int, p: int) -> np.ndarray:
    """Simple 2D label image."""
    arr = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    arr[10:20, 10:20] = 1
    arr[30:40, 30:40] = 2
    return arr


def _meta(t: int, p: int, *, stim: bool = False) -> dict:
    return {
        "timestep": t,
        "fov": p,
        "stim": stim,
        "fname": f"{p:03d}_{t:05d}",
    }


# ===========================================================================
# TiffWriter
# ===========================================================================


class TestTiffWriter:
    """TIFF writer: one file per frame, lazy folder creation."""

    def test_write_creates_file_at_expected_path(self, tmp_dir):
        writer = TiffWriter(tmp_dir)
        img = _mask(0, 0)
        writer.write(img, _meta(0, 0), "labels")
        assert os.path.exists(os.path.join(tmp_dir, "labels", "000_00000.tiff"))

    def test_write_is_roundtrippable(self, tmp_dir):
        """Read-back must equal the original array."""
        writer = TiffWriter(tmp_dir)
        img = _mask(2, 1)
        writer.write(img, _meta(2, 1), "labels")
        roundtripped = tifffile.imread(
            os.path.join(tmp_dir, "labels", "001_00002.tiff")
        )
        np.testing.assert_array_equal(roundtripped, img)

    def test_auto_creates_folders(self, tmp_dir):
        """Folders are created lazily per-write with ``exist_ok``."""
        writer = TiffWriter(tmp_dir)
        for folder in ("raw", "labels", "stim_mask", "particles"):
            writer.write(_mask(0, 0), _meta(0, 0), folder)
            assert os.path.isdir(os.path.join(tmp_dir, folder))

    def test_save_events_produces_events_json(self, tmp_dir):
        writer = TiffWriter(tmp_dir)
        events = [
            RTMEvent(
                index={"t": t, "p": 0},
                channels=(Channel("phase-contrast", 50),),
            )
            for t in range(3)
        ]
        writer.save_events(events)
        assert os.path.exists(os.path.join(tmp_dir, "events.json"))

    def test_close_is_noop(self, tmp_dir):
        """TiffWriter has no buffers; close must not fail."""
        TiffWriter(tmp_dir).close()


# ===========================================================================
# OmeZarrWriter: single-position
# ===========================================================================


ZARR_DIRNAME = "acquisition.ome.zarr"


def _write_full_run(writer, *, n_pos: int, n_t: int = N_T) -> None:
    """Write a full t×p×{raw,stim,labels} run through ``writer``."""
    for t in range(n_t):
        for p in range(n_pos):
            writer.write(_raw(t, p), _meta(t, p), "raw")
            if t % 2 == 0:
                writer.write(
                    np.zeros((IMG_H, IMG_W), dtype=np.uint16),
                    _meta(t, p, stim=True),
                    "stim",
                )
            writer.write(_mask(t, p), _meta(t, p), "labels")


class TestOmeZarrWriterSinglePosition:
    """Single-FOV uses the direct-zarr path: ``(t, c, y, x)`` at root ``0``."""

    @pytest.fixture
    def zarr_path(self, tmp_dir):
        writer = OmeZarrWriter(tmp_dir, store_stim_images=True, n_timepoints=N_T)
        writer.init_stream(
            position_names=["Pos0"],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        _write_full_run(writer, n_pos=1)
        writer.close()
        return Path(tmp_dir) / ZARR_DIRNAME

    def test_store_exists_and_reopens(self, zarr_path):
        assert zarr_path.is_dir()
        zarr.open_group(str(zarr_path), mode="r")

    def test_raw_array_shape_and_dtype(self, zarr_path):
        """Single-pos layout exposes raw as ``0`` with (t, c, y, x)."""
        root = zarr.open_group(str(zarr_path), mode="r")
        raw = root["0"]
        # imaging channels + stim channels
        assert raw.shape == (N_T, len(IMG_CHANNELS) + STIM_CHANNELS, IMG_H, IMG_W)
        assert str(raw.dtype) == "uint16"

    def test_labels_group_populated(self, zarr_path):
        """Labels are written as an NGFF label group at the root."""
        root = zarr.open_group(str(zarr_path), mode="r")
        labels_arr = root["labels/labels/0"]
        assert labels_arr.shape[-2:] == (IMG_H, IMG_W)


# ===========================================================================
# OmeZarrWriter: multi-position (direct path)
# ===========================================================================


class TestOmeZarrWriterMultiPosition:
    """Multi-FOV uses the direct-zarr path: single 5D array with a p axis."""

    N_POS = 3

    @pytest.fixture
    def zarr_path(self, tmp_dir):
        writer = OmeZarrWriter(tmp_dir, store_stim_images=True, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(self.N_POS)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        _write_full_run(writer, n_pos=self.N_POS)
        writer.close()
        return Path(tmp_dir) / ZARR_DIRNAME

    def test_raw_has_position_axis(self, zarr_path):
        """Shape is ``(t, p, c, y, x)`` in direct mode."""
        root = zarr.open_group(str(zarr_path), mode="r")
        raw = root["0"]
        assert raw.shape == (
            N_T,
            self.N_POS,
            len(IMG_CHANNELS) + STIM_CHANNELS,
            IMG_H,
            IMG_W,
        )

    def test_multiscales_axis_names(self, zarr_path):
        """NGFF metadata advertises the correct axis order."""
        root = zarr.open_group(str(zarr_path), mode="r")
        ome = root.attrs["ome"]
        axes = [a["name"] for a in ome["multiscales"][0]["axes"]]
        assert axes == ["t", "p", "c", "y", "x"]

    def test_raw_content_identifiable(self, zarr_path):
        """Raw array preserves the per-(t, p) tag we wrote."""
        root = zarr.open_group(str(zarr_path), mode="r")
        raw = np.asarray(root["0"])
        for t in range(N_T):
            for p in range(self.N_POS):
                expected = 50_000 + t * 100 + p
                assert raw[t, p, 0, t % IMG_H, p % IMG_W] == expected


# ===========================================================================
# OmeZarrWriter.read_raw — deferred-pipeline reload path
# ===========================================================================


class TestOmeZarrReadRaw:
    """``read_raw`` must round-trip the imaging channels of a written frame.

    This is the backend hook the deferred-pipeline path uses to reload a frame
    whose in-RAM copy was dropped under overload. It must (a) work mid-run
    (before close), (b) return imaging channels only — stim readout stripped,
    and (c) be correct for direct (multi-pos), stream (single-pos), and plate
    layouts.
    """

    def _writer(self, cls, tmp_dir, n_pos):
        writer = cls(tmp_dir, store_stim_images=True, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(n_pos)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        return writer

    @pytest.mark.parametrize(
        "cls,n_pos",
        [
            (OmeZarrWriter, 1),  # single-position stream path
            (OmeZarrWriter, 3),  # multi-position direct path
            (OmeZarrWriterPlate, 3),  # plate / per-well path
        ],
    )
    def test_read_raw_roundtrips_imaging_channels(self, cls, n_pos, tmp_dir):
        writer = self._writer(cls, tmp_dir, n_pos)
        try:
            _write_full_run(writer, n_pos=n_pos)
            # Reload mid-run (writer still open), exactly as the deferred worker
            # does once pipeline capacity frees up.
            for t in range(N_T):
                for p in range(n_pos):
                    got = writer.read_raw(_meta(t, p))
                    # Imaging channels only — the stim channel must be stripped.
                    assert got.shape == (len(IMG_CHANNELS), IMG_H, IMG_W)
                    np.testing.assert_array_equal(got, _raw(t, p))
        finally:
            writer.close()


class TestReadRef:
    """``read_ref`` returns the reference channels of a reference frame.

    The deferred-pipeline path needs it to rebuild a reference frame from
    its two stored parts ("raw" and "ref").
    """

    @pytest.mark.parametrize("cls", [TiffWriter, OmeZarrWriter, OmeZarrWriterPlate])
    def test_read_ref_roundtrips(self, cls, tmp_dir):
        writer = cls(tmp_dir)
        ref = _raw(1, 0, n_channels=1)
        writer.write(ref, _meta(1, 0), "ref")
        np.testing.assert_array_equal(writer.read_ref(_meta(1, 0)), ref)
        writer.close()


class TestOmeZarrRawReader:
    """The shared reader must read a *closed* store from a path alone.

    This is the offline re-analysis usage (ControllerSimulated / pipeline_post):
    no live writer, just the store on disk. Same layout coverage as the live
    read, plus the stim-stripping option used by re-analysis.
    """

    def _write_closed_store(self, cls, tmp_dir, n_pos):
        writer = cls(tmp_dir, store_stim_images=True, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(n_pos)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        _write_full_run(writer, n_pos=n_pos)
        writer.close()
        return Path(tmp_dir) / ZARR_DIRNAME

    @pytest.mark.parametrize(
        "cls,n_pos",
        [
            (OmeZarrWriter, 1),  # single-position stream
            (OmeZarrWriter, 3),  # multi-position direct
            (OmeZarrWriterPlate, 3),  # plate / per-well
        ],
    )
    def test_reads_all_channels_by_default(self, cls, n_pos, tmp_dir):
        path = self._write_closed_store(cls, tmp_dir, n_pos)
        reader = OmeZarrRawReader(str(path))
        for t in range(N_T):
            for p in range(n_pos):
                got = reader.read(t, p)
                # No stripping requested → imaging + stim channels present.
                assert got.shape == (len(IMG_CHANNELS) + STIM_CHANNELS, IMG_H, IMG_W)
                np.testing.assert_array_equal(got[: len(IMG_CHANNELS)], _raw(t, p))

    @pytest.mark.parametrize(
        "cls,n_pos",
        [
            (OmeZarrWriter, 1),
            (OmeZarrWriter, 3),
            (OmeZarrWriterPlate, 3),
        ],
    )
    def test_strips_stim_channels_when_requested(self, cls, n_pos, tmp_dir):
        path = self._write_closed_store(cls, tmp_dir, n_pos)
        reader = OmeZarrRawReader(str(path))
        for t in range(N_T):
            for p in range(n_pos):
                got = reader.read(t, p, n_imaging_channels=len(IMG_CHANNELS))
                assert got.shape == (len(IMG_CHANNELS), IMG_H, IMG_W)
                np.testing.assert_array_equal(got, _raw(t, p))


# ===========================================================================
# OmeZarrWriter: stim routing
# ===========================================================================


class TestOmeZarrStimRouting:
    """``store_stim_images=False`` keeps stim readouts out of the raw array."""

    def test_stim_disabled_excludes_stim_channels(self, tmp_dir):
        writer = OmeZarrWriter(tmp_dir, store_stim_images=False, n_timepoints=N_T)
        writer.init_stream(
            position_names=["Pos0"],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=STIM_CHANNELS,
        )
        for t in range(N_T):
            writer.write(_raw(t, 0), _meta(t, 0), "raw")
        writer.close()

        root = zarr.open_group(str(Path(tmp_dir) / ZARR_DIRNAME), mode="r")
        raw = root["0"]
        # Only the imaging channels — stim channels excluded
        assert raw.shape[-3] == len(IMG_CHANNELS)


# ===========================================================================
# OmeZarrWriterPlate
# ===========================================================================


class TestOmeZarrWriterPlate:
    """Plate writer lays positions out as a single-row HCS plate."""

    N_POS = 3

    @pytest.fixture
    def zarr_path(self, tmp_dir):
        writer = OmeZarrWriterPlate(tmp_dir, store_stim_images=False, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(self.N_POS)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=0,
        )
        for t in range(N_T):
            for p in range(self.N_POS):
                writer.write(_raw(t, p), _meta(t, p), "raw")
                writer.write(_mask(t, p), _meta(t, p), "labels")
        writer.close()
        return Path(tmp_dir) / ZARR_DIRNAME

    def test_plate_metadata_defines_one_well_per_position(self, zarr_path):
        """Each position becomes a well in row ``A``."""
        root = zarr.open_group(str(zarr_path), mode="r")
        plate = root.attrs["ome"]["plate"]
        assert [r["name"] for r in plate["rows"]] == ["A"]
        cols = [c["name"] for c in plate["columns"]]
        assert cols == [str(i + 1) for i in range(self.N_POS)]
        well_paths = sorted(w["path"] for w in plate["wells"])
        assert well_paths == [f"A/{i + 1}" for i in range(self.N_POS)]

    def test_well_image_array_shape(self, zarr_path):
        """Each well hosts a single image group ``<row>/<col>/0`` whose ``0``
        array carries the (t, c, y, x) raw stack."""
        for i in range(self.N_POS):
            arr = zarr.open_array(
                str(zarr_path / "A" / str(i + 1) / "0" / "0"), mode="r"
            )
            assert arr.shape == (N_T, len(IMG_CHANNELS), IMG_H, IMG_W)


# ===========================================================================
# OmeZarrWriter: label pre-sizing, dynamic extension, crash recovery
# ===========================================================================


class TestOmeZarrContinuePastDeclared:
    """A run continued past ``n_timepoints`` lands in the store, and growing
    the axis is one metadata write per batch, not one per frame.

    Covers both layouts: single position ``(t, c, y, x)`` and multi-position
    ``(t, p, c, y, x)``. The single-position case used to be an ome-writers
    stream that could not grow at all (bounded) or grew one timepoint per
    frame (unbounded), rewriting ``0/zarr.json`` on every write.
    """

    @pytest.mark.parametrize("n_pos", [1, 2])
    def test_continue_grows_once_and_keeps_frames(self, tmp_dir, n_pos):
        writer = OmeZarrWriter(tmp_dir, store_stim_images=False, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(n_pos)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=0,
        )
        for t in range(N_T):
            for p in range(n_pos):
                writer.write(_raw(t, p), _meta(t, p), "raw")

        # A second phase is appended: the Controller pre-sizes once ...
        writer.set_n_timepoints(2 * N_T)
        array_meta = Path(tmp_dir) / ZARR_DIRNAME / "0" / "zarr.json"
        snapshot = array_meta.read_bytes()
        # ... and the frames of that phase must not touch the metadata again.
        for t in range(N_T, 2 * N_T):
            for p in range(n_pos):
                writer.write(_raw(t, p), _meta(t, p), "raw")
        assert array_meta.read_bytes() == snapshot, "zarr.json rewritten per frame"
        writer.close()

        store = str(Path(tmp_dir) / ZARR_DIRNAME)
        raw = zarr.open_group(store, mode="r")["0"]
        assert raw.shape[0] == 2 * N_T
        reader = OmeZarrRawReader(store)
        for p in range(n_pos):
            last = reader.read(2 * N_T - 1, p)
            np.testing.assert_array_equal(last, _raw(2 * N_T - 1, p))


class TestOmeZarrLabelPresizeAndRepair:
    """Labels are pre-sized to the full run, extend in one shot, and survive
    a crashed close()."""

    N_POS = 2

    def _crashed_store(self, tmp_dir, *, n_written: int) -> Path:
        """Write ``n_written`` of ``N_T`` declared timepoints, then *don't*
        close — simulating a hard crash mid-acquisition. ``events.json`` is
        persisted (as in a real run) so repair can find the true length.
        """
        writer = OmeZarrWriter(tmp_dir, store_stim_images=False, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(self.N_POS)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=0,
        )
        events = []
        for t in range(n_written):
            for p in range(self.N_POS):
                writer.write(_raw(t, p), _meta(t, p), "raw")
                writer.write(_mask(t, p), _meta(t, p), "labels")
                events.append(
                    RTMEvent(
                        index={"t": t, "p": p},
                        channels=(Channel("phase-contrast", 50),),
                    )
                )
        writer.save_events(events)
        # NOTE: no writer.close() — the crash is the whole point.
        return Path(tmp_dir) / ZARR_DIRNAME

    def test_labels_presized_to_full_length(self, tmp_dir):
        """Before any trim, the label array spans the declared timepoints —
        so a live napari reader sees the full time axis."""
        store = self._crashed_store(tmp_dir, n_written=2)
        arr = zarr.open_array(str(store / "labels" / "labels" / "0"), mode="r")
        assert arr.shape == (N_T, self.N_POS, IMG_H, IMG_W)

    def test_close_trims_raw_and_labels_consistently(self, tmp_dir):
        """A clean close shrinks raw and labels to the same written length."""
        writer = OmeZarrWriter(tmp_dir, store_stim_images=False, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(self.N_POS)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=0,
        )
        for t in range(2):  # only 2 of N_T
            for p in range(self.N_POS):
                writer.write(_raw(t, p), _meta(t, p), "raw")
                writer.write(_mask(t, p), _meta(t, p), "labels")
        writer.close()

        store = Path(tmp_dir) / ZARR_DIRNAME
        labels = zarr.open_array(str(store / "labels" / "labels" / "0"), mode="r")
        raw = zarr.open_array(str(store / "0"), mode="r")
        assert labels.shape[0] == 2
        assert raw.shape[0] == 2

    def test_set_n_timepoints_extends_in_one_shot(self, tmp_dir):
        """``set_n_timepoints`` grows raw + labels to the new length at once,
        without per-frame resize churn."""
        writer = OmeZarrWriter(tmp_dir, store_stim_images=False, n_timepoints=N_T)
        writer.init_stream(
            position_names=[f"Pos{i}" for i in range(self.N_POS)],
            channel_names=IMG_CHANNELS,
            image_height=IMG_H,
            image_width=IMG_W,
            n_timepoints=N_T,
            n_stim_channels=0,
        )
        # First write creates the label array at the declared N_T.
        writer.write(_raw(0, 0), _meta(0, 0), "raw")
        writer.write(_mask(0, 0), _meta(0, 0), "labels")

        writer.set_n_timepoints(2 * N_T)  # a new phase was appended at runtime
        raw = writer._raw_array
        labels = writer._label_arrays["labels"]
        assert raw.shape[0] == 2 * N_T
        assert labels.shape[0] == 2 * N_T
        # No-op when not actually extending.
        writer.set_n_timepoints(N_T)
        assert writer._raw_array.shape[0] == 2 * N_T

    def test_repair_rebuilds_missing_group_metadata(self, tmp_dir):
        """After a crash with the group ``zarr.json`` files deleted, repair
        rebuilds them, trims phantom timepoints, and leaves a readable store."""
        store = self._crashed_store(tmp_dir, n_written=2)
        labels_dir = store / "labels"

        # Simulate the reported failure: group metadata lost, arrays intact.
        (labels_dir / "zarr.json").unlink()
        (labels_dir / "labels" / "zarr.json").unlink()
        assert (labels_dir / "labels" / "0" / "zarr.json").exists()  # array kept

        repaired = repair_ome_zarr_labels(str(store))
        assert repaired == ["labels"]

        # Collection group: child list restored.
        labels_grp = zarr.open_group(str(labels_dir), mode="r")
        assert labels_grp.attrs["ome"]["labels"] == ["labels"]

        # Label image group: multiscales with the 4D (t, p, y, x) axes.
        label_grp = zarr.open_group(str(labels_dir / "labels"), mode="r")
        axes = [a["name"] for a in label_grp.attrs["ome"]["multiscales"][0]["axes"]]
        assert axes == ["t", "p", "y", "x"]
        assert "image-label" in label_grp.attrs["ome"]

        # Phantom timepoints trimmed to the 2 acquired (from events.json).
        arr = zarr.open_array(str(labels_dir / "labels" / "0"), mode="r")
        assert arr.shape[0] == 2

    def test_repair_is_idempotent(self, tmp_dir):
        """Running repair on an already-valid store is a harmless no-op."""
        store = self._crashed_store(tmp_dir, n_written=2)
        first = repair_ome_zarr_labels(str(store))
        second = repair_ome_zarr_labels(str(store))
        assert first == second == ["labels"]
