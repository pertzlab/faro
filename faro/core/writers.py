"""Storage backends for acquisition data.

Three implementations:
- TiffWriter: current behavior (individual TIFF files per frame)
- OmeZarrWriter: single 5D array with position slider (t, p, c, y, x)
- OmeZarrWriterPlate: plate/well layout with spatial tiling in napari
"""

from __future__ import annotations

import os
import threading
from typing import Protocol, runtime_checkable

import numpy as np
import tifffile

from faro.core.utils import create_folders

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


class OmeZarrWriter:
    """Streams acquisition data into a single OME-Zarr v0.5 store.

    Uses one ``ome-writers`` stream with a position dimension so that all
    positions live inside a single bf2raw zarr container.

    Positions are configured automatically by the Controller from the
    event list (via :meth:`init_stream`), or can be set manually.

    Routing:
    - ``"raw"`` — appended to the primary OME-Zarr image stream.
    - ``"stim"`` — appended as an additional channel in the raw stream.
      Zeros are written for non-stim timepoints.
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
            n_timepoints: Expected number of timepoints. None = unbounded.
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
        self._stream = None  # ome-writers stream (single-position)
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
        """Create the ome-writers stream.

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
        n_pos = len(position_names)

        if n_pos > 1:
            # --- Multi-position: build zarr store directly ---
            # ome-writers/yaozarrs can't handle custom axis types between
            # time and space (axis order bug). Build the store ourselves
            # so we get a single 5D array (t, p, c, y, x) with full control.
            self._init_stream_direct(
                position_names,
                all_channel_names,
                total_channels,
            )
        else:
            # --- Single position: use ome-writers ---
            self._init_stream_ome_writers(all_channel_names, total_channels)

    def _init_stream_ome_writers(
        self,
        all_channel_names: list[str],
        total_channels: int,
    ) -> None:
        """Single-position stream via ome-writers."""
        from ome_writers import AcquisitionSettings, Dimension, create_stream

        dimensions = []

        # Time (unbounded OK — it's the first dimension)
        t_kwargs: dict = dict(
            name="t",
            count=self._n_timepoints,
            chunk_size=self._raw_chunk_t,
            type="time",
        )
        if self._raw_shard_t is not None and self._raw_shard_t != self._raw_chunk_t:
            t_kwargs["shard_size_chunks"] = self._raw_shard_t // self._raw_chunk_t
        dimensions.append(Dimension(**t_kwargs))

        if total_channels > 1:
            dimensions.append(
                Dimension(
                    name="c",
                    count=total_channels,
                    chunk_size=1,  # one chunk per channel
                    shard_size_chunks=total_channels,  # all channels in one shard
                    type="channel",
                    coords=all_channel_names,
                )
            )

        dimensions.append(
            Dimension(
                name="y",
                count=self._image_height,
                chunk_size=self._image_height,
                type="space",
            )
        )
        dimensions.append(
            Dimension(
                name="x",
                count=self._image_width,
                chunk_size=self._image_width,
                type="space",
            )
        )

        settings = AcquisitionSettings(
            root_path=self._zarr_path,
            dimensions=dimensions,
            dtype=self._dtype,
            overwrite=self._overwrite,
        )
        self._stream = create_stream(settings)

    def _init_stream_direct(
        self,
        position_names: list[str],
        all_channel_names: list[str],
        total_channels: int,
    ) -> None:
        """Multi-position stream built directly with zarr-python.

        Bypasses yaozarrs' axis order validator which incorrectly rejects
        custom types between time and space.
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

        root = zarr.open_group(self._zarr_path, mode="w")

        axes = [
            {"name": k, "type": _NGFF_AXIS_TYPE.get(k, "other")}
            for k in leading_axes + list(_TRAILING_AXES)
        ]
        root.attrs["ome"] = {
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

        # Create the raw data array
        self._raw_array = root.create_array(
            "0",
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=self._dtype,
            fill_value=0,
            overwrite=True,
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
        return tuple(
            metadata.get(_META_KEY.get(k, k), 0) for k in self._axis_keys
        )

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
        if folder == "raw":
            self._write_raw(img, metadata)
        elif folder == "stim":
            self._write_stim(img, metadata)
        elif folder == "ref":
            self._tiff.write(img, metadata, folder)
        else:
            self._write_label(img, metadata, folder)

    def close(self) -> None:
        if self._stream is not None:
            if self._stim_pending:
                self._append_stim_zeros()
                self._stim_pending = False
            self._stream.close()
            self._stream = None
        # For direct mode, raw_array is managed by zarr (no explicit close)
        self._raw_array = None
        self._tiff.close()

    # ------------------------------------------------------------------
    # Raw frames → ome-writers stream
    # ------------------------------------------------------------------

    def _write_raw(self, img: np.ndarray, metadata: dict) -> None:
        """Append imaging frame(s) to the OME-Zarr store."""
        if self._stream is not None:
            # Single-position: use ome-writers stream
            self._write_raw_stream(img, metadata)
        else:
            # Multi-position: direct zarr write
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
            arr[leading_idx + (slice(None, img.shape[0]), slice(None), slice(None))] = img
        elif img.ndim == 2:
            arr[leading_idx + (0, slice(None), slice(None))] = img

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
                self._raw_array[leading_idx + (stim_start, slice(None), slice(None))] = frame
            else:
                self._raw_array[
                    leading_idx
                    + (slice(stim_start, stim_start + frame.shape[0]), slice(None), slice(None))
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
                return 0  # grows on write
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
