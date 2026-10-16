"""Storage backends for acquisition data.

Three implementations:
- TiffWriter: current behavior (individual TIFF files per frame)
- OmeZarrWriter: single 5D array with position slider (t, p, c, y, x)
- OmeZarrWriterPlate: plate/well layout with spatial tiling in napari
"""

from __future__ import annotations

import os
import threading
import time
from typing import Protocol, runtime_checkable

import numpy as np
import tifffile

from faro.core.utils import create_folders


# Network shares + Windows AV occasionally hold zarr's `.partial.{uuid}`
# chunk file open for tens of ms while we try to rename it to the final
# chunk path, surfacing as PermissionError [WinError 5]. Retry with
# exponential backoff before giving up; the writer's _write_lock keeps
# our own threads from racing on the same chunk in the meantime.
_WRITE_RETRY_ATTEMPTS = 6
_WRITE_RETRY_BASE_DELAY = 0.1  # s; doubles each attempt → ~6.4s total worst case

# Default channel colors for omero metadata (hex RGB)
_DEFAULT_CHANNEL_COLORS = [
    "0000FF",  # blue (DAPI)
    "00FF00",  # green (GFP/FITC)
    "FF0000",  # red (RFP/Rhodamine)
    "FF00FF",  # magenta (Cy5)
    "FFFF00",  # yellow
    "00FFFF",  # cyan
]


# ---------------------------------------------------------------------------
# Writer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Writer(Protocol):
    """Abstraction for image storage during acquisition."""

    storage_path: str

    def write(self, img: np.ndarray, metadata: dict, folder: str) -> None:
        """Write an image to storage.

        Args:
            img: Image array (2D or 3D with leading channel dim).
            metadata: Event metadata (must contain 'fname', 'fov', 'timestep').
            folder: Logical folder name ('raw', 'labels', 'stim_mask', ...).
        """
        ...

    def read_raw(self, metadata: dict) -> np.ndarray:
        """Read back a previously written raw frame as ``(c, y, x)``.

        Used by the deferred-pipeline path: under pipeline overload the in-RAM
        frame is dropped and reloaded later from storage, so the reload must go
        through whichever backend wrote it. Returns only the imaging channels
        (stim readout channels stripped), matching what was passed to
        ``write(img, metadata, "raw")``.

        Args:
            metadata: Event metadata (must contain 'fname', 'fov', 'timestep').
        """
        ...

    def read_ref(self, metadata: dict) -> np.ndarray:
        """Read back the reference channels of a reference frame.

        A reference frame is stored in two parts: its imaging channels as
        ``"raw"`` and its reference channels as ``"ref"``. The deferred
        pipeline path needs both: ``read_raw`` for the imaging channels and
        this method for the reference channels.

        Args:
            metadata: Event metadata (must contain 'fname', 'fov', 'timestep').
        """
        ...

    def save_events(self, events) -> None:
        """Save acquisition events as ``events.json`` in the storage path."""
        ...

    def close(self) -> None:
        """Flush buffers and release resources."""
        ...


# ---------------------------------------------------------------------------
# TIFF writer (current default)
# ---------------------------------------------------------------------------


class TiffWriter:
    """Writes each frame as an individual compressed TIFF file.

    This preserves the existing storage layout::

        storage_path/
        ├── raw/          001_00042.tiff
        ├── labels/       001_00042.tiff
        ├── stim_mask/    ...
        └── ...
    """

    def __init__(self, storage_path: str, folders: list[str] | None = None):
        self.storage_path = storage_path
        if folders:
            create_folders(storage_path, folders)

    def write(self, img: np.ndarray, metadata: dict, folder: str) -> None:
        os.makedirs(os.path.join(self.storage_path, folder), exist_ok=True)
        fname = metadata["fname"]
        tifffile.imwrite(
            os.path.join(self.storage_path, folder, fname + ".tiff"),
            img,
            compression="zlib",
            compressionargs={"level": 5},
        )

    def read_raw(self, metadata: dict) -> np.ndarray:
        fname = metadata["fname"]
        return tifffile.imread(os.path.join(self.storage_path, "raw", fname + ".tiff"))

    def read_ref(self, metadata: dict) -> np.ndarray:
        fname = metadata["fname"]
        return tifffile.imread(os.path.join(self.storage_path, "ref", fname + ".tiff"))

    def save_events(self, events) -> None:
        from faro.core.conversion import save_events_json

        save_events_json(self.storage_path, events)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# OME-Zarr writer
# ---------------------------------------------------------------------------


def _extract_positions_from_events(events) -> list[str]:
    """Extract unique position names from RTMEvents, ordered by index.

    Returns a list of position names like ``['Pos0', 'Pos1', ...]``.
    Uses ``event.pos_name`` when available, otherwise ``'Pos{i}'``.
    """
    positions: dict[int, str] = {}
    for ev in events:
        p = ev.index.get("p", 0)
        if p not in positions:
            name = getattr(ev, "pos_name", None) or f"Pos{p}"
            positions[p] = name
    return [positions[k] for k in sorted(positions)]


def _extract_channel_names_from_events(events) -> list[str]:
    """Return imaging channel names from the first RTMEvent.

    Falls back to ``['ch0', 'ch1', ...]`` if no channel configs are found.
    """
    for ev in events:
        if hasattr(ev, "channels") and ev.channels:
            return [ch.config for ch in ev.channels]
    return ["ch0"]


def _extract_n_timepoints_from_events(events) -> int:
    """Return the number of unique timepoints from events."""
    return max((ev.index.get("t", 0) for ev in events), default=0) + 1


def _extract_n_stim_channels_from_events(events) -> int:
    """Return the number of stim channels from events (0 if none)."""
    for ev in events:
        if hasattr(ev, "stim_channels") and ev.stim_channels:
            return len(ev.stim_channels)
    return 0


from useq import Axis

_TRAILING_AXES: tuple[str, ...] = ("c", "y", "x")

_NGFF_AXIS_TYPE: dict[str, str] = {
    Axis.TIME: "time",
    Axis.POSITION: "other",
    Axis.CHANNEL: "channel",
    Axis.Z: "space",
    "y": "space",
    "x": "space",
}

# Metadata key used by RTMEvent.to_mda_events() for each leading axis.
_META_KEY: dict[str, str] = {Axis.TIME: "timestep", Axis.POSITION: "fov"}


def _derive_direct_axes(n_pos: int) -> list[str]:
    """Return leading axis keys for the multi-position direct-mode store."""
    if n_pos > 1:
        return [Axis.TIME, Axis.POSITION]
    return [Axis.TIME]


def _growable_dim_indices(axis_keys: list[str]) -> tuple[int, ...]:
    """Indices (within ``axis_keys``) of axes that can grow on write.

    Only ``t`` is unbounded in practice; other FARO dimensions are fixed
    by the event list at open time.
    """
    return tuple(i for i, k in enumerate(axis_keys) if k == Axis.TIME)


class OmeZarrRawReader:
    """Read-only access to raw imaging frames in an OME-Zarr acquisition store.

    Single home for the store-layout knowledge so that every "reload a raw
    frame from disk" caller shares one implementation instead of re-deriving
    indexing from axes metadata. Two consumers:

    - the live deferred-pipeline reload (:meth:`OmeZarrWriter.read_raw`), and
    - offline re-analysis (``ControllerSimulated``).

    Handles the three layouts faro writes:

    - direct multi-position — root ``"0"`` array, axes ``(t, p, c, y, x)``
    - single-position stream — root ``"0"`` array, axes ``(t, c, y, x)``
    - plate — one array per well at ``"<well>/0/0"``, axes ``(t, c, y, x)``,
      ordered by ``(rowIndex, columnIndex)`` to match FOV index.

    Opens the store read-only. Concurrent with a live writer this is safe for
    already-written frames; the deferred path only ever reloads frames older
    than the most recent write, which are flushed by the time they are read.
    """

    def __init__(self, zarr_path: str):
        import zarr

        self._root = zarr.open_group(zarr_path, mode="r")
        ome = dict(self._root.attrs).get("ome", {})
        plate = ome.get("plate")
        if plate is not None:
            self._is_plate = True
            wells = sorted(
                plate.get("wells", []),
                key=lambda w: (w.get("rowIndex", 0), w.get("columnIndex", 0)),
            )
            # Each well holds a single field "0"; its raw array is "<well>/0/0".
            self._well_arrays = [f"{w['path']}/0/0" for w in wells]
        else:
            self._is_plate = False
            self._raw = self._root["0"]
            axes = ome.get("multiscales", [{}])[0].get("axes", [])
            self._axes = [a["name"] for a in axes]

    def read(
        self, timestep: int, fov: int, *, n_imaging_channels: int | None = None
    ) -> np.ndarray:
        """Return raw frame ``(c, y, x)`` for ``(timestep, fov)``.

        If ``n_imaging_channels`` is given, trailing stim-readout channels are
        stripped so callers get imaging channels only.
        """
        if self._is_plate:
            frame = np.asarray(self._root[self._well_arrays[fov]][timestep])
        else:
            has_p = "p" in self._axes
            has_c = "c" in self._axes
            arr = self._raw
            if has_p and has_c:
                frame = np.asarray(arr[timestep, fov])
            elif has_p:
                frame = np.asarray(arr[timestep, fov])[np.newaxis]
            elif has_c:
                frame = np.asarray(arr[timestep])
            else:
                frame = np.asarray(arr[timestep])[np.newaxis]
        if (
            n_imaging_channels
            and frame.ndim == 3
            and frame.shape[0] > n_imaging_channels
        ):
            frame = frame[:n_imaging_channels]
        return frame


class OmeZarrWriter:
    """Streams acquisition data into a single OME-Zarr v0.5 store.

    The raw array is built directly with zarr-python (see
    :meth:`_init_stream_direct`): ``(t, c, y, x)`` for a single position,
    ``(t, p, c, y, x)`` for several, always at the root ``"0"`` array. The
    time axis is pre-sized to ``n_timepoints``, grown once per appended batch
    by :meth:`set_n_timepoints`, and trimmed to the written length on
    :meth:`close`, so a run can be continued or extended without rewriting
    the array metadata on every frame.

    Positions are configured automatically by the Controller from the
    event list (via :meth:`init_stream`), or can be set manually.

    Routing:
    - ``"raw"`` — written into the primary OME-Zarr image array.
    - ``"stim"`` — written as additional channel(s) of the raw array.
      Non-stim timepoints keep the zero fill value.
    - ``"ref"`` — TIFF fallback (different channel count).
    - Everything else — stored as NGFF label groups under each position's
      image group, created lazily on first write.

    Layout (bf2raw)::

        storage_path/
        ├── acquisition.ome.zarr/
        │   ├── Pos0/
        │   │   ├── 0/                 raw + stim readout (t, c, y, x)
        │   │   └── labels/
        │   │       ├── labels/        segmentation masks
        │   │       ├── stim_mask/     stimulation masks
        │   │       └── ...
        │   ├── Pos1/ ...
        │   ├── OME/                   series index
        │   └── zarr.json              bf2raw metadata
        ├── ref/                       TIFF fallback
        └── tracks/                    parquet files (unchanged)
    """

    def __init__(
        self,
        storage_path: str,
        dtype: str = "uint16",
        *,
        store_stim_images: bool = True,
        n_timepoints: int | None = None,
        label_dtype: str = "uint16",
        raw_chunk_t: int = 1,
        raw_shard_t: int | None = None,
        label_chunk_t: int = 1,
        label_shard_t: int = 50,
        overwrite: bool = True,
    ):
        """
        Args:
            storage_path: Root directory for all outputs.
            dtype: Pixel dtype for raw data.
            store_stim_images: If True and the experiment has stim channels,
                stim readout images are stored as additional channel(s) in the
                raw zarr array (zeros for non-stim timepoints).  If False
                (default), stim readouts fall back to TIFF.
            n_timepoints: Expected number of timepoints; pre-sizes the raw
                and label arrays. Continued or extended runs grow it via
                :meth:`set_n_timepoints`; :meth:`close` trims to the written
                length.
            label_dtype: Dtype for label arrays.
            raw_chunk_t: Temporal chunk size for raw data.
            raw_shard_t: Temporal shard size for raw data (None = same as chunk).
            label_chunk_t: Temporal chunk size for labels (1 = random access).
            label_shard_t: Temporal shard size for labels (groups chunks into shards).
            overwrite: Whether to overwrite existing zarr store.
        """
        self.storage_path = storage_path
        self._dtype = dtype
        self._label_dtype = label_dtype
        self._label_chunk_t = label_chunk_t
        self._label_shard_t = label_shard_t
        self._store_stim_images = store_stim_images
        self._n_stim_channels: int = 0  # set by init_stream from events
        self._n_timepoints = n_timepoints
        self._raw_chunk_t = raw_chunk_t
        self._raw_shard_t = raw_shard_t
        self._overwrite = overwrite

        # Set by init_stream() — derived from events + microscope
        self._stream = None  # ome-writers stream (OmeZarrWriterPlate only)
        self._raw_array = None  # direct zarr array (multi-position)
        self._image_height: int = 0
        self._image_width: int = 0
        self._n_imaging_channels: int = 0
        self._position_names: list[str] = []
        self._zarr_path = os.path.join(storage_path, "acquisition.ome.zarr")

        # Direct-mode axis layout. Populated by _init_stream_direct.
        self._axis_keys: list[str] = []
        self._growable_dims: tuple[int, ...] = ()

        # Stim channel ordering buffers
        self._stim_pending: bool = False
        self._stim_buffer: dict[tuple[int, int], np.ndarray] = {}
        self._zero_stim: np.ndarray | None = None

        # TIFF fallback for ref images
        self._tiff = TiffWriter(storage_path)

        # Label arrays created lazily on first write (one array per label name)
        self._label_arrays: dict[str, object] = {}
        self._label_lock = threading.Lock()
        # Highest timepoint actually written to the raw / any label array;
        # used to trim pre-sized arrays back down on close (-1 = none written).
        self._raw_max_t: int = -1
        self._label_max_t: int = -1
        # The pipeline runs in 4 worker threads + a storage thread; all of
        # them call writer.write(). The OME-Zarr store is sharded across
        # channels (one shard per (t, p), all channels packed in), so a
        # write to one channel does a read-modify-write of the whole shard
        # file. Concurrent writes to different channels of the same (t, p)
        # — e.g. raw imaging from the storage thread and the stim channel
        # from a pipeline worker — race on zarr's partial-shard atomic
        # rename and surface as PermissionError [WinError 5] on Windows
        # / Z: SMB shares. Serialize all .write() calls to keep the
        # rename atomic per shard.
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Event persistence
    # ------------------------------------------------------------------

    def save_events(self, events) -> None:
        from faro.core.conversion import save_events_json

        save_events_json(self.storage_path, events)

    # ------------------------------------------------------------------
    # Stream initialization (called by Controller with event-derived positions)
    # ------------------------------------------------------------------

    def init_stream(
        self,
        position_names: list[str],
        channel_names: list[str],
        image_height: int,
        image_width: int,
        n_timepoints: int | None = None,
        n_stim_channels: int = 0,
    ) -> None:
        """Create the OME-Zarr store.

        Called by :class:`Controller` during :meth:`run_experiment` with
        values derived from the event list and microscope hardware.
        Must be called before the first :meth:`write`.

        Args:
            position_names: One name per FOV (extracted from events).
            channel_names: Imaging channel names (extracted from events).
            image_height: Camera height in pixels (from microscope).
            image_width: Camera width in pixels (from microscope).
            n_timepoints: Number of timepoints (from events).
            n_stim_channels: Number of stim channels (from events).
        """
        if n_timepoints is not None and self._n_timepoints is None:
            self._n_timepoints = n_timepoints
        # Only include stim channels if store_stim_images is True
        if self._store_stim_images:
            self._n_stim_channels = n_stim_channels

        self._position_names = position_names
        self._n_imaging_channels = len(channel_names)
        self._image_height = image_height
        self._image_width = image_width

        # Build full channel name list (imaging + stim readout)
        all_channel_names = list(channel_names)
        for i in range(self._n_stim_channels):
            all_channel_names.append(f"stim_{i}")
        total_channels = len(all_channel_names)

        # Build the store directly with zarr-python for any number of
        # positions. The single-position case used to go through an
        # ome-writers stream, but that stream cannot be continued: a bounded
        # time axis refuses appends past the declared count, and an unbounded
        # one (ome-writers <= 0.3.2) grows by exactly one timepoint per frame,
        # rewriting the array's zarr.json on every write. The direct path
        # pre-sizes, grows once per batch, and trims on close.
        self._init_stream_direct(
            position_names,
            all_channel_names,
            total_channels,
        )

    def _init_stream_direct(
        self,
        position_names: list[str],
        all_channel_names: list[str],
        total_channels: int,
    ) -> None:
        """Build the raw array directly with zarr-python.

        Used for every position count. With one position the leading axes
        are ``(t,)``; with several they are ``(t, p)``. Bypasses ome-writers /
        yaozarrs, whose axis order validator rejects a custom position axis
        between time and space, and whose stream cannot be pre-sized and
        grown in one step (see :meth:`set_n_timepoints`).
        """
        import shutil
        import zarr

        if self._overwrite and os.path.exists(self._zarr_path):
            shutil.rmtree(self._zarr_path)

        n_pos = len(position_names)
        leading_axes = _derive_direct_axes(n_pos)
        self._axis_keys = leading_axes
        self._growable_dims = _growable_dim_indices(leading_axes)

        def _leading_size(key: str) -> int:
            if key == Axis.TIME:
                return self._n_timepoints or 1
            if key == Axis.POSITION:
                return n_pos
            return 1

        def _chunk_for(key: str) -> int:
            return self._raw_chunk_t if key == Axis.TIME else 1

        def _shard_for(key: str) -> int:
            if key == Axis.TIME:
                return self._raw_shard_t or self._raw_chunk_t
            return 1

        trailing_sizes = {
            "c": total_channels,
            "y": self._image_height,
            "x": self._image_width,
        }
        trailing_chunks = (1, self._image_height, self._image_width)
        trailing_shards = (total_channels, self._image_height, self._image_width)

        shape = tuple(_leading_size(k) for k in leading_axes) + tuple(
            trailing_sizes[k] for k in _TRAILING_AXES
        )
        chunks = tuple(_chunk_for(k) for k in leading_axes) + trailing_chunks
        shards = tuple(_shard_for(k) for k in leading_axes) + trailing_shards

        axes = [
            {"name": k, "type": _NGFF_AXIS_TYPE.get(k, "other")}
            for k in leading_axes + list(_TRAILING_AXES)
        ]
        # Bake the OME metadata into the group's initial zarr.json write.
        # Assigning root.attrs afterwards triggers a *second* write -- an
        # atomic temp-file + os.replace over the just-created zarr.json --
        # which intermittently fails with PermissionError (WinError 5) on
        # SMB/network drives, where the just-written file is still held by
        # an oplock or AV scan. Writing zarr.json exactly once avoids it.
        ome_metadata = {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0] * len(axes)}
                            ],
                        }
                    ],
                }
            ],
            "omero": {
                "channels": [
                    {
                        "label": name,
                        "active": True,
                        "color": _DEFAULT_CHANNEL_COLORS[
                            i % len(_DEFAULT_CHANNEL_COLORS)
                        ],
                        "window": {"start": 0, "end": 65535},
                    }
                    for i, name in enumerate(all_channel_names)
                ],
            },
        }
        root = zarr.open_group(
            self._zarr_path, mode="w", attributes={"ome": ome_metadata}
        )

        # Create the raw data array
        self._raw_array = root.create_array(
            "0",
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=self._dtype,
            fill_value=0,
            overwrite=True,
            dimension_names=[str(a["name"]) for a in axes],
            chunk_key_encoding={"name": "v2", "separator": "."},
        )

        # Track frame count per position for sequential appending
        self._direct_frame_count: dict[int, int] = {p: 0 for p in range(n_pos)}

        # No ome-writers stream — we write directly
        self._stream = None

    # ------------------------------------------------------------------
    # Direct-mode axis helpers
    # ------------------------------------------------------------------

    def _leading_index(self, metadata: dict) -> tuple[int, ...]:
        """Build the leading index tuple for a direct-mode write."""
        return tuple(metadata.get(_META_KEY.get(k, k), 0) for k in self._axis_keys)

    def _maybe_resize_leading(self, arr, leading_idx: tuple[int, ...]) -> None:
        """Grow any growable leading axis that the incoming write exceeds."""
        growable = self._growable_dims
        if not growable:
            return
        shape = arr.shape
        if not any(leading_idx[d] >= shape[d] for d in growable):
            return
        with self._label_lock:  # reused lock — thread-safe resize
            cur = list(arr.shape)
            changed = False
            for d in growable:
                idx = leading_idx[d]
                if idx >= cur[d]:
                    cur[d] = idx + 1
                    changed = True
            if changed:
                arr.resize(tuple(cur))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, img: np.ndarray, metadata: dict, folder: str) -> None:
        with self._write_lock:
            last_err: Exception | None = None
            for attempt in range(_WRITE_RETRY_ATTEMPTS):
                try:
                    if folder == "raw":
                        self._write_raw(img, metadata)
                    elif folder == "stim":
                        self._write_stim(img, metadata)
                    elif folder == "ref":
                        self._tiff.write(img, metadata, folder)
                    else:
                        self._write_label(img, metadata, folder)
                    return
                except (PermissionError, OSError) as e:
                    # WinError 5 / EACCES typically means antivirus or the
                    # SMB share momentarily holds the `.partial.{uuid}` chunk
                    # open while zarr tries to rename it. Backing off briefly
                    # almost always clears it.
                    last_err = e
                    if attempt + 1 < _WRITE_RETRY_ATTEMPTS:
                        time.sleep(_WRITE_RETRY_BASE_DELAY * (2**attempt))
            assert last_err is not None
            raise last_err

    def read_raw(self, metadata: dict) -> np.ndarray:
        """Read back a raw frame's imaging channels as ``(c, y, x)``.

        Two storage modes:
        - Multi-position *direct* mode keeps a live, random-access zarr array
          (``self._raw_array``); we index it directly, so there is no reopen
          and no flush-timing concern.
        - Single-position *stream* mode appends through ome-writers; the raw
          data lands at the store's root ``"0"`` array on disk, which we open
          read-only. Deferred frames are always older than the most recent
          write, so they are flushed by the time they are reloaded.

        Held under ``_write_lock``: this runs on the deferred-pipeline thread
        concurrently with acquisition writes, and zarr-python's shared sync
        loop is not safe to drive from a reader thread while writer threads are
        mid-flush (it surfaces as "cannot schedule new futures after shutdown").
        The writer already serializes every write through this lock, so taking
        it here makes raw read-back mutually exclusive with writes. Reads are
        fast and deferral is the catch-up path, so the added contention is
        negligible.
        """
        with self._write_lock:
            if self._raw_array is not None:
                leading_idx = self._leading_index(metadata)
                frame = np.asarray(self._raw_array[leading_idx])
            else:
                frame = np.asarray(self._read_raw_from_disk(metadata))
        return self._strip_stim_channels(frame)

    def read_ref(self, metadata: dict) -> np.ndarray:
        """Read back the reference channels of a reference frame.

        This writer stores reference frames as TIFF files (see ``write``), so
        the TIFF writer reads them back.
        """
        return self._tiff.read_ref(metadata)

    def _read_raw_from_disk(self, metadata: dict) -> np.ndarray:
        """Read a raw frame from the on-disk store. ``(c, y, x)``, stim kept.

        Delegates layout handling to :class:`OmeZarrRawReader`, which covers the
        stream and plate layouts alike. A fresh reader is opened per call so it
        always sees the current (growing) store shape. Stim stripping is done by
        the caller (:meth:`read_raw`), so we read all channels here.
        """
        t = int(metadata.get("timestep", 0))
        p = int(metadata.get("fov", 0))
        return OmeZarrRawReader(self._zarr_path).read(t, p)

    def _strip_stim_channels(self, frame: np.ndarray) -> np.ndarray:
        """Drop trailing stim-readout channels so callers get imaging only.

        The raw array packs imaging channels followed by stim channels; the
        ``write(..., "raw")`` input was imaging-only, so reads must match.
        """
        n_img = self._n_imaging_channels
        if frame.ndim == 3 and n_img and frame.shape[0] > n_img:
            return frame[:n_img]
        return frame

    def close(self) -> None:
        if self._stream is not None:
            if self._stim_pending:
                self._append_stim_zeros()
                self._stim_pending = False
            self._stream.close()
            self._stream = None
        # Trim pre-sized raw + label arrays down to the timepoints actually
        # written, so a run that stops early doesn't persist phantom (all-zero)
        # timepoints — and raw/labels keep a matching length. Time is always
        # leading axis 0 in direct mode.
        self._trim_arrays_to_written()
        # For direct mode, raw_array is managed by zarr (no explicit close)
        self._raw_array = None
        self._tiff.close()

    @staticmethod
    def _trim_time_axis(arr, max_t: int) -> None:
        """Shrink ``arr``'s leading time axis to ``max_t + 1`` if oversized."""
        if max_t < 0:
            return
        target_t = max_t + 1
        if arr.shape[0] > target_t:
            arr.resize((target_t,) + tuple(arr.shape[1:]))

    def _trim_arrays_to_written(self) -> None:
        if self._raw_array is not None:
            self._trim_time_axis(self._raw_array, self._raw_max_t)
        for arr in self._label_arrays.values():
            self._trim_time_axis(arr, self._label_max_t)

    def set_n_timepoints(self, n: int) -> None:
        """Extend the declared run length, pre-sizing arrays up to ``n`` once.

        Call this from a dynamic-queue / multi-phase driver when events are
        appended past the originally declared ``n_timepoints``. It grows the
        raw and label arrays' time axis to ``n`` in a single resize each,
        instead of letting per-frame writes crawl the shape up one timepoint at
        a time (which rewrites each array's ``zarr.json`` on every frame — a
        replace-over-existing that is slow and crash-fragile on SMB shares).

        No-op when ``n`` does not exceed the current declared length. The
        Controller calls this from ``continue_experiment`` and
        ``extend_experiment``. :class:`OmeZarrWriterPlate` still streams raw
        frames through a bounded ome-writers stream, so for it this only grows
        the label arrays and plate runs cannot be continued yet.
        """
        if n <= (self._n_timepoints or 0):
            return
        self._n_timepoints = n
        # Same lock as _maybe_resize_leading: this runs on the caller's thread
        # while the writer thread may be growing the same arrays.
        with self._label_lock:
            if self._raw_array is not None and self._raw_array.shape[0] < n:
                self._raw_array.resize((n,) + tuple(self._raw_array.shape[1:]))
            for arr in self._label_arrays.values():
                if arr.shape[0] < n:
                    arr.resize((n,) + tuple(arr.shape[1:]))

    # ------------------------------------------------------------------
    # Raw frames → zarr array (direct) or ome-writers stream (plate subclass)
    # ------------------------------------------------------------------

    def _write_raw(self, img: np.ndarray, metadata: dict) -> None:
        """Write imaging frame(s) into the OME-Zarr store."""
        if self._stream is not None:
            # ome-writers stream (only OmeZarrWriterPlate sets one up)
            self._write_raw_stream(img, metadata)
        else:
            self._write_raw_direct(img, metadata)

    def _write_raw_stream(self, img: np.ndarray, metadata: dict) -> None:
        """Single-position: append via ome-writers stream."""
        if self._stim_pending:
            self._append_stim_zeros()
            self._stim_pending = False

        if img.ndim == 2:
            self._stream.append(img)
        elif img.ndim == 3:
            for c in range(img.shape[0]):
                self._stream.append(img[c])
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {img.shape}")

        if self._n_stim_channels == 0:
            return

        t = metadata.get("timestep", 0)
        p = metadata.get("fov", 0)
        buffered = self._stim_buffer.pop((t, p), None)

        if buffered is not None:
            self._append_stim_frame(buffered)
        elif not metadata.get("stim", False):
            self._append_stim_zeros()
        else:
            self._stim_pending = True

    def _write_raw_direct(self, img: np.ndarray, metadata: dict) -> None:
        """Multi-position: write directly to zarr array."""
        arr = self._raw_array
        leading_idx = self._leading_index(metadata)
        self._maybe_resize_leading(arr, leading_idx)

        if img.ndim == 3:
            arr[leading_idx + (slice(None, img.shape[0]), slice(None), slice(None))] = (
                img
            )
        elif img.ndim == 2:
            arr[leading_idx + (0, slice(None), slice(None))] = img
        self._raw_max_t = max(self._raw_max_t, metadata.get("timestep", 0))

    def _write_stim(self, img: np.ndarray, metadata: dict) -> None:
        """Handle a stim readout frame."""
        if self._n_stim_channels == 0:
            self._tiff.write(img, metadata, "stim")
            return

        frame = img.squeeze() if img.ndim > 2 and img.shape[0] == 1 else img

        # Direct mode: write stim channel directly to zarr (random access)
        if self._raw_array is not None:
            leading_idx = self._leading_index(metadata)
            self._maybe_resize_leading(self._raw_array, leading_idx)
            stim_start = self._n_imaging_channels
            if frame.ndim == 2:
                self._raw_array[
                    leading_idx + (stim_start, slice(None), slice(None))
                ] = frame
            else:
                self._raw_array[
                    leading_idx
                    + (
                        slice(stim_start, stim_start + frame.shape[0]),
                        slice(None),
                        slice(None),
                    )
                ] = frame
            return

        if self._stim_pending:
            self._append_stim_frame(frame)
            self._stim_pending = False
        else:
            t = metadata.get("timestep", 0)
            p = metadata.get("fov", 0)
            self._stim_buffer[(t, p)] = frame

    def _append_stim_frame(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            self._stream.append(frame)
        else:
            for c in range(frame.shape[0]):
                self._stream.append(frame[c])

    def _append_stim_zeros(self) -> None:
        if self._zero_stim is None:
            self._zero_stim = np.zeros(
                (self._image_height, self._image_width),
                dtype=np.dtype(self._dtype),
            )
        for _ in range(self._n_stim_channels):
            self._stream.append(self._zero_stim)

    # ------------------------------------------------------------------
    # Label / mask frames → NGFF label groups (lazy creation)
    # ------------------------------------------------------------------

    def _write_label(self, img: np.ndarray, metadata: dict, name: str) -> None:
        """Write a label frame to the NGFF label array."""
        frame = img.squeeze() if img.ndim > 2 and img.shape[0] == 1 else img

        # Lazy init on first encounter for this label name
        if name not in self._label_arrays:
            with self._label_lock:
                if name not in self._label_arrays:
                    self._create_label_array(name, frame)

        arr = self._label_arrays[name]
        leading_idx = self._leading_index(metadata)
        self._maybe_resize_leading(arr, leading_idx)
        arr[leading_idx] = frame
        self._label_max_t = max(self._label_max_t, metadata.get("timestep", 0))

    def _create_label_array(self, name: str, sample_frame: np.ndarray) -> None:
        """Create an NGFF-compliant label array under the root image group.

        Layout mirrors the raw data array's leading axes plus ``(y, x)``;
        labels do not have a channel axis.
        """
        import zarr

        # Fallback when init_stream wasn't called (labels-only writer path).
        if not self._axis_keys:
            self._axis_keys = _derive_direct_axes(len(self._position_names))
            self._growable_dims = _growable_dim_indices(self._axis_keys)

        leading_axes = self._axis_keys

        def _label_size(key: str) -> int:
            if key == Axis.TIME:
                # Pre-size to the full experiment length when known, so the
                # array (and its zarr.json) is written once and never resized
                # mid-run. Resize churn is what can leave metadata truncated on
                # a crash, and a fixed shape lets a live napari reader open the
                # store once and see the whole time axis. Falls back to 0
                # (grow-on-write) when unbounded.
                return self._n_timepoints or 0
            if key == Axis.POSITION:
                return len(self._position_names)
            return 1

        def _label_chunk(key: str) -> int:
            return self._label_chunk_t if key == Axis.TIME else 1

        def _label_shard(key: str) -> int:
            return self._label_shard_t if key == Axis.TIME else 1

        leading_shape = tuple(_label_size(k) for k in leading_axes)
        leading_chunks = tuple(_label_chunk(k) for k in leading_axes)
        leading_shards = tuple(_label_shard(k) for k in leading_axes)

        array_shape = leading_shape + (self._image_height, self._image_width)
        chunks = leading_chunks + (self._image_height, self._image_width)
        shards = leading_shards + (self._image_height, self._image_width)
        axes = [
            {"name": k, "type": _NGFF_AXIS_TYPE.get(k, "other")}
            for k in leading_axes + ["y", "x"]
        ]
        scale = [1.0] * len(axes)

        # Root IS the image group
        img_grp = zarr.open_group(self._zarr_path, mode="a")

        # labels/ container — write at BOTH levels for compatibility:
        # - under "ome" namespace (NGFF v0.5 correct)
        # - at top level (ome-zarr-py / napari-ome-zarr compat)
        labels_grp = img_grp.require_group("labels")
        ome_attrs = dict(labels_grp.attrs.get("ome", {}))
        existing = list(ome_attrs.get("labels", []))
        if name not in existing:
            existing.append(name)
            ome_attrs["labels"] = existing
            ome_attrs["version"] = "0.5"
            labels_grp.attrs["ome"] = ome_attrs
            labels_grp.attrs["labels"] = existing  # ome-zarr-py compat

        label_grp = labels_grp.require_group(name)

        # Resizable array at resolution level 0
        arr = label_grp.create_array(
            "0",
            shape=array_shape,
            chunks=chunks,
            shards=shards,
            dtype=sample_frame.dtype,
            fill_value=0,
            overwrite=True,
            chunk_key_encoding={"name": "v2", "separator": "."},
        )

        # Write metadata at both levels (v0.5 ome namespace + ome-zarr-py compat)
        multiscales = [
            {
                "name": name,
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": scale}
                        ],
                    }
                ],
            }
        ]
        image_label = {"source": {"image": "../../"}}

        label_grp.attrs["ome"] = {
            "version": "0.5",
            "multiscales": multiscales,
            "image-label": image_label,
        }
        # ome-zarr-py compat (reads from top-level attrs)
        label_grp.attrs["multiscales"] = multiscales
        label_grp.attrs["image-label"] = image_label

        self._label_arrays[name] = arr


# ---------------------------------------------------------------------------
# OME-Zarr writer — plate/well layout
# ---------------------------------------------------------------------------


class OmeZarrWriterPlate(OmeZarrWriter):
    """Streams acquisition data into an OME-Zarr v0.5 plate store.

    Each FOV position becomes a well in a single-row plate.
    napari-ome-zarr tiles the positions spatially as a mosaic.

    Inherits all stim buffering, ref fallback, and close logic from
    :class:`OmeZarrWriter`.  Overrides stream initialization (plate layout)
    and label writing (per-well labels instead of shared array).

    Layout::

        storage_path/
        ├── acquisition.ome.zarr/
        │   ├── zarr.json              plate metadata
        │   ├── A/
        │   │   ├── 1/                 well for Pos0
        │   │   │   ├── zarr.json      well metadata
        │   │   │   └── 0/             image group (field of view)
        │   │   │       ├── 0/         raw array (t, c, y, x)
        │   │   │       └── labels/
        │   │   │           ├── labels/
        │   │   │           ├── stim_mask/
        │   │   │           └── particles/
        │   │   ├── 2/                 well for Pos1
        │   │   └── ...
        ├── ref/                       TIFF fallback
        └── tracks/                    parquet files
    """

    def init_stream(
        self,
        position_names: list[str],
        channel_names: list[str],
        image_height: int,
        image_width: int,
        n_timepoints: int | None = None,
        n_stim_channels: int = 0,
    ) -> None:
        """Create the ome-writers stream with plate/well layout."""
        from ome_writers import (
            AcquisitionSettings,
            Dimension,
            Plate,
            Position,
            create_stream,
        )

        if n_timepoints is not None and self._n_timepoints is None:
            self._n_timepoints = n_timepoints
        if self._store_stim_images:
            self._n_stim_channels = n_stim_channels

        self._position_names = position_names
        self._n_imaging_channels = len(channel_names)
        self._image_height = image_height
        self._image_width = image_width

        all_channel_names = list(channel_names)
        for i in range(self._n_stim_channels):
            all_channel_names.append(f"stim_{i}")
        total_channels = len(all_channel_names)

        dimensions = []

        # Time (unbounded OK — positions are handled via plate, not as first dim)
        t_kwargs: dict = dict(
            name="t",
            count=self._n_timepoints,
            chunk_size=self._raw_chunk_t,
            type="time",
        )
        if self._raw_shard_t is not None and self._raw_shard_t != self._raw_chunk_t:
            t_kwargs["shard_size_chunks"] = self._raw_shard_t // self._raw_chunk_t
        dimensions.append(Dimension(**t_kwargs))

        # Positions as plate wells (single row "A", columns "1", "2", ...)
        col_names = [str(i + 1) for i in range(len(position_names))]
        pos_coords = [
            Position(
                name="0",  # single field per well
                plate_row="A",
                plate_column=col,
            )
            for col in col_names
        ]
        dimensions.append(Dimension(name="p", type="position", coords=pos_coords))

        # Channels
        if total_channels > 1:
            dimensions.append(
                Dimension(
                    name="c",
                    count=total_channels,
                    chunk_size=total_channels,
                    type="channel",
                    coords=all_channel_names,
                )
            )

        # Spatial
        dimensions.append(
            Dimension(
                name="y", count=image_height, chunk_size=image_height, type="space"
            )
        )
        dimensions.append(
            Dimension(name="x", count=image_width, chunk_size=image_width, type="space")
        )

        # Plate metadata — single row with one column per position
        plate = Plate(
            row_names=["A"],
            column_names=col_names,
        )

        settings = AcquisitionSettings(
            root_path=self._zarr_path,
            dimensions=dimensions,
            dtype=self._dtype,
            plate=plate,
            overwrite=self._overwrite,
        )

        self._stream = create_stream(settings)

        # Store well paths for label writing: "A/1/0", "A/2/0", ...
        self._well_image_paths = [os.path.join("A", col, "0") for col in col_names]

    # ------------------------------------------------------------------
    # Raw writing — uses ome-writers stream (always, plate handles positions)
    # ------------------------------------------------------------------

    def _write_raw(self, img: np.ndarray, metadata: dict) -> None:
        """Plate always uses the ome-writers stream (no direct mode)."""
        self._write_raw_stream(img, metadata)

    # Raw read-back (deferred reload) is inherited from OmeZarrWriter: its
    # _read_raw_from_disk uses OmeZarrRawReader, which auto-detects the plate
    # layout from the store's ome metadata — no plate-specific override needed.

    # ------------------------------------------------------------------
    # Label writing — per-well (each well has its own label groups)
    # ------------------------------------------------------------------

    def _write_label(self, img: np.ndarray, metadata: dict, name: str) -> None:
        """Write a label frame into the well's image group."""
        t = metadata.get("timestep", 0)
        pos = metadata.get("fov", 0)
        frame = img.squeeze() if img.ndim > 2 and img.shape[0] == 1 else img

        key = (name, pos)
        if key not in self._label_arrays:
            with self._label_lock:
                if key not in self._label_arrays:
                    self._create_label_array_plate(name, pos, frame)

        arr = self._label_arrays[key]

        # Resize time dimension (axis 0) if needed
        if t >= arr.shape[0]:
            with self._label_lock:
                if t >= arr.shape[0]:
                    new_shape = list(arr.shape)
                    new_shape[0] = t + 1
                    arr.resize(tuple(new_shape))

        arr[t] = frame

    def _create_label_array_plate(
        self,
        name: str,
        pos: int,
        sample_frame: np.ndarray,
    ) -> None:
        """Create a label array under the well's image group."""
        import zarr

        array_shape = (0, self._image_height, self._image_width)
        chunks = (self._label_chunk_t, self._image_height, self._image_width)
        shards = (self._label_shard_t, self._image_height, self._image_width)
        axes = [
            {"name": k, "type": _NGFF_AXIS_TYPE.get(k, "other")}
            for k in (Axis.TIME, "y", "x")
        ]
        scale = [1.0, 1.0, 1.0]

        # Open the well's image group: <zarr_path>/A/<col>/0
        well_img_path = os.path.join(self._zarr_path, self._well_image_paths[pos])
        img_grp = zarr.open_group(well_img_path, mode="a")

        # labels/ container
        labels_grp = img_grp.require_group("labels")
        ome_attrs = dict(labels_grp.attrs.get("ome", {}))
        existing = list(ome_attrs.get("labels", []))
        if name not in existing:
            existing.append(name)
            ome_attrs["labels"] = existing
            ome_attrs["version"] = "0.5"
            labels_grp.attrs["ome"] = ome_attrs
            labels_grp.attrs["labels"] = existing  # ome-zarr-py compat

        label_grp = labels_grp.require_group(name)

        arr = label_grp.create_array(
            "0",
            shape=array_shape,
            chunks=chunks,
            shards=shards,
            dtype=sample_frame.dtype,
            fill_value=0,
            overwrite=True,
            chunk_key_encoding={"name": "v2", "separator": "."},
        )

        multiscales = [
            {
                "name": name,
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": scale}
                        ],
                    }
                ],
            }
        ]
        image_label = {"source": {"image": "../../"}}

        label_grp.attrs["ome"] = {
            "version": "0.5",
            "multiscales": multiscales,
            "image-label": image_label,
        }
        # ome-zarr-py compat
        label_grp.attrs["multiscales"] = multiscales
        label_grp.attrs["image-label"] = image_label

        self._label_arrays[(name, pos)] = arr


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def _true_n_timepoints(events_dir: str) -> int | None:
    """Return ``max(t) + 1`` from ``<events_dir>/events.json``, or None."""
    import json

    path = os.path.join(events_dir, "events.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return max(d["index"].get("t", 0) for d in data) + 1
    except (ValueError, KeyError, OSError):
        return None


def repair_ome_zarr_labels(
    zarr_path: str, *, events_path: str | None = None
) -> list[str]:
    """Rebuild missing group metadata for an OME-Zarr ``labels`` collection.

    Crash recovery for stores written by :class:`OmeZarrWriter`. A hard crash
    can leave the label *arrays* (and their per-array ``zarr.json``) on disk
    while the group-level ``labels/zarr.json`` and ``labels/<name>/zarr.json``
    are missing or stale — in zarr v3 a group without a ``zarr.json`` is not a
    valid node, so a reader finds no labels. This reconstructs them from
    whatever arrays exist on disk; it is idempotent and safe to re-run.

    When an ``events.json`` is found (next to the store, or via ``events_path``)
    each label array's time axis is trimmed to the true timepoint count, so a
    crashed run doesn't advertise phantom (all-zero) timepoints.

    Args:
        zarr_path: Path to the ``*.ome.zarr`` store (the image group; labels
            live under ``<zarr_path>/labels``).
        events_path: Directory holding ``events.json``. Defaults to the store's
            parent directory.

    Returns:
        The label names that were repaired (sorted).
    """
    import zarr

    labels_dir = os.path.join(zarr_path, "labels")
    if not os.path.isdir(labels_dir):
        return []

    # Discover label images: subdirectories that hold a level-0 array.
    names = sorted(
        d
        for d in os.listdir(labels_dir)
        if os.path.exists(os.path.join(labels_dir, d, "0", "zarr.json"))
    )
    if not names:
        return []

    n_t = _true_n_timepoints(events_path or os.path.dirname(zarr_path))

    for name in names:
        arr = zarr.open_array(os.path.join(labels_dir, name, "0"), mode="r+")
        if n_t is not None and arr.shape[0] > n_t:
            arr.resize((n_t,) + tuple(arr.shape[1:]))

        # Leading axes: 4D -> (t, p, y, x), 3D -> (t, y, x).
        leading = [Axis.TIME, Axis.POSITION] if arr.ndim == 4 else [Axis.TIME]
        axes = [
            {"name": k, "type": _NGFF_AXIS_TYPE.get(k, "other")}
            for k in leading + ["y", "x"]
        ]
        multiscales = [
            {
                "name": name,
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0] * len(axes)}
                        ],
                    }
                ],
            }
        ]
        image_label = {"source": {"image": "../../"}}
        label_grp = zarr.open_group(os.path.join(labels_dir, name), mode="a")
        label_grp.attrs["ome"] = {
            "version": "0.5",
            "multiscales": multiscales,
            "image-label": image_label,
        }
        label_grp.attrs["multiscales"] = multiscales  # ome-zarr-py compat
        label_grp.attrs["image-label"] = image_label

    labels_grp = zarr.open_group(labels_dir, mode="a")
    labels_grp.attrs["ome"] = {"version": "0.5", "labels": names}
    labels_grp.attrs["labels"] = names  # ome-zarr-py compat
    return names
