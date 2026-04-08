"""Conversion utilities for legacy formats.

Includes:
- ``df_to_events``: convert legacy df_acquire DataFrames to RTMEvent lists.
- ``events_to_df``: convert RTMEvent lists back to df_acquire DataFrames.
- ``save_events_json`` / ``load_events_json``: persist events as JSON.
- ``convert_tiff_to_omezarr``: migrate a TIFF-per-frame experiment directory
  to a single OME-Zarr v0.5 store using :class:`OmeZarrWriter`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from faro.core.data_structures import Channel, PowerChannel, RTMEvent


def _dict_to_channel(d: dict) -> Channel | PowerChannel:
    """Convert a channel dict to a Channel or PowerChannel dataclass.

    Returns PowerChannel if the dict contains a ``power`` key with a non-None
    value, otherwise returns a plain Channel.
    """
    if d.get("power") is not None:
        return PowerChannel(
            config=d["config"],
            exposure=d.get("exposure"),
            group=d.get("group"),
            power=d["power"],
        )
    return Channel(
        config=d["config"],
        exposure=d.get("exposure"),
        group=d.get("group"),
    )


def df_to_events(df_acquire: pd.DataFrame) -> list[RTMEvent]:
    """Convert a legacy *df_acquire* DataFrame to a list of :class:`RTMEvent`.

    One DataFrame row produces one RTMEvent.  Column mapping:

    =========== ====================================
    Column      RTMEvent field
    =========== ====================================
    fov         index["p"]
    timestep    index["t"]
    time        min_start_time
    fov_x       x_pos
    fov_y       y_pos
    fov_z       z_pos
    channels    channels (tuple of Channel dicts)
    stim=True   stim_channels built from stim_* cols
    =========== ====================================

    Remaining columns are forwarded as ``metadata``.
    """
    events: list[RTMEvent] = []

    # Columns that map directly to RTMEvent fields (not metadata)
    _SKIP_META = {
        "fov",
        "timestep",
        "time",
        "fov_x",
        "fov_y",
        "fov_z",
        "channels",
        "stim_channels",
        "ref_channels",
        "stim",
        "stim_channel_name",
        "stim_channel_group",
        "stim_channel_device_name",
        "stim_channel_power_property_name",
        "stim_power",
        "stim_exposure",
        "stim_timestep",
        "stim_exposure_list",
        "device_name",
        "property_name",
    }

    for _, row in df_acquire.iterrows():
        fov = int(row["fov"])
        timestep = int(row["timestep"])

        # --- imaging channels ---
        channels_raw = row.get("channels", ())
        channels = tuple(_dict_to_channel(d) for d in channels_raw)

        # --- stim channels ---
        # New format: stim_channels column contains tuple of channel dicts
        stim_channels: tuple[Channel, ...] = ()
        stim_channels_raw = row.get("stim_channels")
        if (
            stim_channels_raw
            and isinstance(stim_channels_raw, (list, tuple))
            and len(stim_channels_raw) > 0
        ):
            if row.get("stim", False):
                stim_exposure = row.get("stim_exposure")
                stim_chs = []
                for d in stim_channels_raw:
                    ch = _dict_to_channel(d)
                    # Override exposure with per-frame stim_exposure if available
                    if stim_exposure and pd.notna(stim_exposure):
                        ch = type(ch)(
                            config=ch.config,
                            exposure=float(stim_exposure),
                            group=ch.group,
                            **(
                                {"power": ch.power}
                                if isinstance(ch, PowerChannel)
                                else {}
                            ),
                        )
                    stim_chs.append(ch)
                stim_channels = tuple(stim_chs)
        # Fallback: old flat-column format
        elif (
            row.get("stim", False)
            and row.get("stim_exposure")
            and row.get("stim_power")
        ):
            stim_ch_name = row.get("stim_channel_name", "")
            stim_ch_group = row.get("stim_channel_group")
            stim_power = row.get("stim_power")
            stim_exposure = row.get("stim_exposure")
            stim_channels = (
                PowerChannel(
                    config=stim_ch_name,
                    exposure=stim_exposure,
                    group=stim_ch_group,
                    power=int(stim_power) if pd.notna(stim_power) else None,
                ),
            )

        # --- ref channels ---
        ref_channels: tuple[Channel, ...] = ()
        ref_channels_raw = row.get("ref_channels")
        if (
            ref_channels_raw
            and isinstance(ref_channels_raw, (list, tuple))
            and len(ref_channels_raw) > 0
        ):
            ref_channels = tuple(_dict_to_channel(d) for d in ref_channels_raw)

        # --- metadata: everything not consumed above ---
        metadata = {k: v for k, v in row.items() if k not in _SKIP_META}

        events.append(
            RTMEvent(
                index={"t": timestep, "p": fov},
                channels=channels,
                stim_channels=stim_channels,
                ref_channels=ref_channels,
                x_pos=row.get("fov_x"),
                y_pos=row.get("fov_y"),
                z_pos=row.get("fov_z"),
                min_start_time=float(row.get("time", 0)),
                metadata=metadata,
            )
        )

    return events


# ---------------------------------------------------------------------------
# RTMEvent ↔ JSON serialization
# ---------------------------------------------------------------------------


def _event_to_dict(event: RTMEvent) -> dict:
    """Serialize an RTMEvent to a JSON-compatible dict.

    Uses ``dataclasses.asdict`` for Channel/PowerChannel so that subclass
    fields (e.g. ``power``) are preserved.
    """
    return {
        "index": dict(event.index),
        "channels": [asdict(ch) for ch in event.channels],
        "stim_channels": [asdict(ch) for ch in event.stim_channels],
        "ref_channels": [asdict(ch) for ch in event.ref_channels],
        "x_pos": event.x_pos,
        "y_pos": event.y_pos,
        "z_pos": event.z_pos,
        "pos_name": event.pos_name,
        "min_start_time": event.min_start_time,
        "metadata": event.metadata,
    }


def _dict_to_event(d: dict) -> RTMEvent:
    """Deserialize a dict (from JSON) back to an RTMEvent."""
    return RTMEvent(
        index=d["index"],
        channels=tuple(_dict_to_channel(ch) for ch in d.get("channels", [])),
        stim_channels=tuple(_dict_to_channel(ch) for ch in d.get("stim_channels", [])),
        ref_channels=tuple(_dict_to_channel(ch) for ch in d.get("ref_channels", [])),
        x_pos=d.get("x_pos"),
        y_pos=d.get("y_pos"),
        z_pos=d.get("z_pos"),
        pos_name=d.get("pos_name"),
        min_start_time=d.get("min_start_time"),
        metadata=d.get("metadata", {}),
    )


def save_events_json(path: str, events) -> None:
    """Save a list of RTMEvents to ``<path>/events.json``."""
    data = [_event_to_dict(ev) for ev in events]
    filepath = os.path.join(path, "events.json")
    os.makedirs(path, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_events_json(path: str) -> list[RTMEvent]:
    """Load a list of RTMEvents from ``<path>/events.json``."""
    filepath = os.path.join(path, "events.json")
    with open(filepath) as f:
        data = json.load(f)
    return [_dict_to_event(d) for d in data]


# ---------------------------------------------------------------------------
# RTMEvent list → df_acquire DataFrame
# ---------------------------------------------------------------------------


def events_to_df(events: list[RTMEvent]) -> pd.DataFrame:
    """Convert a list of RTMEvents to a legacy *df_acquire* DataFrame.

    This is the inverse of :func:`df_to_events`.  The resulting DataFrame
    has the columns expected by :class:`ImageProcessingPipeline_postExperiment`.
    """
    rows: list[dict] = []
    for ev in events:
        fov = ev.index.get("p", 0)
        timestep = ev.index.get("t", 0)
        has_stim = len(ev.stim_channels) > 0

        row: dict = {
            "fov": fov,
            "timestep": timestep,
            "time": ev.min_start_time or 0.0,
            "fov_x": ev.x_pos,
            "fov_y": ev.y_pos,
            "fov_z": ev.z_pos,
            "fname": f"{fov:03d}_{timestep:05d}",
            "channels": [asdict(ch) for ch in ev.channels],
            "stim": has_stim,
        }

        if ev.stim_channels:
            row["stim_channels"] = [asdict(ch) for ch in ev.stim_channels]
        if ev.ref_channels:
            row["ref_channels"] = [asdict(ch) for ch in ev.ref_channels]

        # Forward event metadata as extra columns
        for k, v in ev.metadata.items():
            if k not in row:
                row[k] = v

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TIFF → OME-Zarr conversion
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"^(\d+)_(\d+)\.tiff?$", re.IGNORECASE)

# Folders that map to the raw image stream (not labels).
_RAW_FOLDERS = {"raw"}
# Folders stored as TIFF fallback by OmeZarrWriter (ref images have
# different channel counts, so they stay as TIFF).
_TIFF_FOLDERS = {"ref", "optocheck"}
# Folders whose images are stim readouts (appended as extra raw channels).
_STIM_IMG_FOLDERS = {"stim"}


def _parse_fname(fname: str) -> tuple[int, int] | None:
    """Return ``(fov, timestep)`` parsed from *fname*, or ``None``."""
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _scan_tiff_folder(folder_path: str) -> dict[tuple[int, int], str]:
    """Return ``{(fov, timestep): filepath, ...}`` for TIFFs in *folder_path*."""
    result: dict[tuple[int, int], str] = {}
    if not os.path.isdir(folder_path):
        return result
    for fname in os.listdir(folder_path):
        parsed = _parse_fname(fname)
        if parsed is not None:
            result[parsed] = os.path.join(folder_path, fname)
    return result


def convert_tiff_to_omezarr(
    src_path: str,
    dst_path: str,
    *,
    channel_names: list[str] | None = None,
    store_stim_images: bool = False,
    label_folders: list[str] | None = None,
    raw_chunk_t: int = 1,
    raw_shard_t: int | None = None,
    label_chunk_t: int = 1,
    label_shard_t: int = 50,
    overwrite: bool = True,
    copy_tracks: bool = True,
    verbose: bool = True,
) -> str:
    """Convert a TIFF-per-frame experiment to OME-Zarr v0.5.

    Reads the old directory layout::

        src_path/
        ├── raw/            {fov}_{timestep}.tiff  (C, H, W) or (H, W)
        ├── labels/         segmentation masks
        ├── particles/      tracked labels
        ├── stim_mask/      stimulation masks
        ├── stim/           stim readout images
        ├── ref/            reference images
        └── tracks/         parquet files

    and writes into ``dst_path`` using :class:`OmeZarrWriter`::

        dst_path/
        ├── acquisition.ome.zarr/
        │   ├── 0/                   raw data (t, p, c, y, x)
        │   └── labels/
        │       ├── labels/0         segmentation masks
        │       ├── particles/0      tracked labels
        │       └── stim_mask/0      stimulation masks
        ├── ref/                     TIFF fallback
        └── tracks/                  parquet (copied)

    Args:
        src_path: Root of the old TIFF experiment directory.
        dst_path: Destination directory for the OME-Zarr output.
        channel_names: Names for imaging channels. If ``None``, auto-named
            ``['ch0', 'ch1', ...]`` based on the channel dimension of the
            first raw TIFF.
        store_stim_images: If ``True`` and a ``stim/`` folder exists, stim
            readout images are stored as extra channels in the raw array.
            If ``False`` (default), stim readouts are stored as TIFFs.
        label_folders: Explicit list of subfolder names to treat as label
            arrays (e.g. ``['labels', 'particles', 'stim_mask']``).
            If ``None``, auto-detected: every subfolder that is not
            ``raw``, ``stim``, ``ref``, ``optocheck``, or ``tracks``.
        raw_chunk_t: Temporal chunk size for raw data.
        raw_shard_t: Temporal shard size for raw data.
        label_chunk_t: Temporal chunk size for label arrays.
        label_shard_t: Temporal shard size for label arrays.
        overwrite: Whether to overwrite an existing zarr store.
        copy_tracks: Whether to copy the ``tracks/`` folder to *dst_path*.
        verbose: Print progress to stdout.

    Returns:
        Path to the created ``acquisition.ome.zarr`` directory.
    """
    from faro.core.writers import OmeZarrWriter

    src = Path(src_path)
    dst = Path(dst_path)

    # --- Discover subfolders ---
    subfolders = [
        d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    if "raw" not in subfolders:
        raise FileNotFoundError(
            f"No 'raw/' subfolder found in {src_path}. "
            "Expected old TIFF format with raw/ directory."
        )

    # --- Scan raw files to determine dimensions ---
    raw_files = _scan_tiff_folder(str(src / "raw"))
    if not raw_files:
        raise FileNotFoundError(f"No TIFF files found in {src / 'raw'}")

    fovs = sorted({fov for fov, _ in raw_files})
    timesteps = sorted({ts for _, ts in raw_files})
    n_fov = len(fovs)
    n_timepoints = max(timesteps) + 1

    # Read first file to get shape info
    first_key = min(raw_files.keys())
    first_img = tifffile.imread(raw_files[first_key])
    if first_img.ndim == 3:
        n_channels, img_h, img_w = first_img.shape
    elif first_img.ndim == 2:
        n_channels = 1
        img_h, img_w = first_img.shape
    else:
        raise ValueError(f"Unexpected raw image shape: {first_img.shape}")

    dtype = str(first_img.dtype)

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but raw images "
            f"have {n_channels} channels"
        )

    # --- Detect stim folder ---
    has_stim_folder = "stim" in subfolders
    stim_files: dict[tuple[int, int], str] = {}
    n_stim_channels = 0
    if has_stim_folder and store_stim_images:
        stim_files = _scan_tiff_folder(str(src / "stim"))
        if stim_files:
            sample_stim = tifffile.imread(next(iter(stim_files.values())))
            n_stim_channels = sample_stim.shape[0] if sample_stim.ndim == 3 else 1

    # --- Auto-detect label folders ---
    _skip = _RAW_FOLDERS | _TIFF_FOLDERS | _STIM_IMG_FOLDERS | {"tracks"}
    if label_folders is None:
        label_folders = [d for d in subfolders if d not in _skip]
    label_files: dict[str, dict[tuple[int, int], str]] = {}
    for lf in label_folders:
        scanned = _scan_tiff_folder(str(src / lf))
        if scanned:
            label_files[lf] = scanned

    # --- Detect ref/optocheck folders ---
    ref_folders = [d for d in subfolders if d in _TIFF_FOLDERS]
    ref_files: dict[str, dict[tuple[int, int], str]] = {}
    for rf in ref_folders:
        scanned = _scan_tiff_folder(str(src / rf))
        if scanned:
            ref_files[rf] = scanned

    if verbose:
        print(f"Source: {src_path}")
        print(
            f"  FOVs: {n_fov}, Timepoints: {n_timepoints}, "
            f"Channels: {n_channels} {channel_names}"
        )
        print(f"  Image size: {img_h}x{img_w}, dtype: {dtype}")
        print(f"  Raw frames: {len(raw_files)}")
        if stim_files:
            print(f"  Stim frames: {len(stim_files)} ({n_stim_channels} ch)")
        if label_files:
            print(f"  Label folders: {list(label_files.keys())}")
        if ref_files:
            print(f"  Ref folders: {list(ref_files.keys())}")

    # --- Create OmeZarrWriter ---
    position_names = [f"Pos{fov}" for fov in fovs]

    writer = OmeZarrWriter(
        storage_path=str(dst),
        dtype=dtype,
        store_stim_images=store_stim_images,
        n_timepoints=n_timepoints,
        label_dtype="uint16",
        raw_chunk_t=raw_chunk_t,
        raw_shard_t=raw_shard_t,
        label_chunk_t=label_chunk_t,
        label_shard_t=label_shard_t,
        overwrite=overwrite,
    )

    writer.init_stream(
        position_names=position_names,
        channel_names=channel_names,
        image_height=img_h,
        image_width=img_w,
        n_timepoints=n_timepoints,
        n_stim_channels=n_stim_channels,
    )

    # --- Build FOV index mapping (fov_value → position index) ---
    fov_to_idx = {fov: i for i, fov in enumerate(fovs)}

    # --- Write raw frames (iterate in t, p order for stream compatibility) ---
    if verbose:
        print(f"\nWriting raw frames...")
    count = 0
    for t in timesteps:
        for fov in fovs:
            key = (fov, t)
            if key not in raw_files:
                continue
            img = tifffile.imread(raw_files[key])
            metadata = {
                "fname": f"{fov:03d}_{t:05d}",
                "fov": fov_to_idx[fov],
                "timestep": t,
                "stim": key in stim_files,
            }
            writer.write(img, metadata, "raw")

            # Write stim readout right after the raw frame (stream ordering)
            if key in stim_files:
                stim_img = tifffile.imread(stim_files[key])
                writer.write(stim_img, metadata, "stim")

            count += 1
    if verbose:
        print(f"  Wrote {count} raw frames")

    # --- Write stim images that aren't stored as channels ---
    if has_stim_folder and not store_stim_images:
        stim_tiff_files = _scan_tiff_folder(str(src / "stim"))
        if stim_tiff_files and verbose:
            print(f"Writing stim images as TIFF ({len(stim_tiff_files)} frames)...")
        for (fov, t), fpath in sorted(stim_tiff_files.items()):
            img = tifffile.imread(fpath)
            metadata = {
                "fname": f"{fov:03d}_{t:05d}",
                "fov": fov_to_idx[fov],
                "timestep": t,
            }
            writer.write(img, metadata, "stim")

    # --- Write label folders ---
    for lf_name, lf_files in label_files.items():
        if verbose:
            print(f"Writing label '{lf_name}' ({len(lf_files)} frames)...")
        for (fov, t), fpath in sorted(lf_files.items()):
            img = tifffile.imread(fpath)
            metadata = {
                "fname": f"{fov:03d}_{t:05d}",
                "fov": fov_to_idx[fov],
                "timestep": t,
            }
            writer.write(img, metadata, lf_name)

    # --- Write ref images (TIFF fallback) ---
    for rf_name, rf_files in ref_files.items():
        if verbose:
            print(f"Writing ref '{rf_name}' ({len(rf_files)} frames as TIFF)...")
        for (fov, t), fpath in sorted(rf_files.items()):
            img = tifffile.imread(fpath)
            metadata = {
                "fname": f"{fov:03d}_{t:05d}",
                "fov": fov_to_idx[fov],
                "timestep": t,
            }
            writer.write(img, metadata, "ref")

    writer.close()
    if verbose:
        print(f"\nOME-Zarr store created: {writer._zarr_path}")

    # --- Copy tracks ---
    src_tracks = src / "tracks"
    dst_tracks = dst / "tracks"
    if copy_tracks and src_tracks.is_dir():
        if dst_tracks.exists():
            shutil.rmtree(str(dst_tracks))
        shutil.copytree(str(src_tracks), str(dst_tracks))
        if verbose:
            n_parquet = len(list(dst_tracks.glob("*.parquet")))
            print(f"Copied tracks/ ({n_parquet} parquet files)")

    if verbose:
        print("Conversion complete.")

    return writer._zarr_path
