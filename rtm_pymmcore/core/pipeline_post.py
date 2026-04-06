# Offline reprocessing pipeline for post-experiment image processing.

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import tifffile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import rtm_pymmcore.segmentation.base as base_segmentation
import rtm_pymmcore.stimulation.base as base_stimulation
import rtm_pymmcore.tracking.base as abstract_tracker
import rtm_pymmcore.feature_extraction.base as abstract_fe
from rtm_pymmcore.core.data_structures import FovState, ImgType, SegmentationMethod
from rtm_pymmcore.core.utils import create_folders
from rtm_pymmcore.core.pipeline import (
    store_img,
    build_frame_dataframe,
    run_tracking,
    extract_and_merge_features,
    dispatch_stim_mask,
    convert_track_dtypes,
    save_segmentation_results,
)
from rtm_pymmcore.core.writers import Writer, TiffWriter, OmeZarrWriter


class ImageProcessingPipeline_postExperiment:
    def __init__(
        self,
        img_storage_path: str,
        out_path: str,
        df_acquire: pd.DataFrame | None = None,
        segmentators: List[SegmentationMethod] = None,
        feature_extractor: abstract_fe.FeatureExtractor = None,
        stimulator: base_stimulation.Stim = None,
        tracker: abstract_tracker.Tracker = None,
        feature_extractor_ref: abstract_fe.FeatureExtractor = None,
        use_old_segmentations: bool = False,
        use_old_stim_masks: bool = False,
        n_jobs: int = 2,
        correct_timestep_jumps: bool = False,
        events: list | None = None,
        writer: Writer | None = None,
    ):
        if events is not None:
            from rtm_pymmcore.core.conversion import events_to_df

            df_acquire = events_to_df(events)
        elif df_acquire is None:
            raise ValueError("Either 'events' or 'df_acquire' must be provided.")

        self.segmentators = segmentators
        self.feature_extractor = feature_extractor
        self.stimulator = stimulator
        self.tracker = tracker
        self.df_acquire = df_acquire
        self.feature_extractor_ref = feature_extractor_ref
        self.storage_path = out_path
        self.img_storage_path = img_storage_path
        self.use_old_segmentations = use_old_segmentations
        self.n_jobs = n_jobs
        self.correct_timestep_jumps = correct_timestep_jumps
        self.use_old_stim_masks = use_old_stim_masks
        self._events = events

        # Writer setup: user-provided or default TiffWriter
        self._writer = writer
        self._writer_is_omezarr = isinstance(writer, OmeZarrWriter)

        # Detect OME-Zarr input
        zarr_path = os.path.join(img_storage_path, "acquisition.ome.zarr")
        self._use_zarr = os.path.isdir(zarr_path)
        if self._use_zarr:
            import zarr

            self._zarr_store = zarr.open(zarr_path, mode="r")
            self._zarr_raw = self._zarr_store["0"]
            ome = self._zarr_store.attrs.get("ome", {})
            axes = ome.get("multiscales", [{}])[0].get("axes", [])
            self._zarr_axes = [a["name"] for a in axes]
        else:
            self._zarr_store = None

        # When using an OmeZarrWriter, folder creation is handled by the
        # zarr store itself — only tracks/ needs to exist on disk.
        if self._writer_is_omezarr:
            os.makedirs(os.path.join(self.storage_path, "tracks"), exist_ok=True)
        else:
            folders = ["raw", "tracks"]
            self.folders_to_move = folders.copy()
            stim_folders = ["stim_mask", "stim"]
            if self.stimulator is not None:
                folders.extend(stim_folders)
            if self.use_old_stim_masks:
                self.folders_to_move.extend(stim_folders)
                folders.extend(stim_folders)
            if self.tracker is not None:
                folders.append("particles")
            if self.feature_extractor is not None:
                if hasattr(self.feature_extractor, "extra_folders"):
                    folders.extend(self.feature_extractor.extra_folders)
            if self.segmentators is not None:
                for seg in self.segmentators:
                    folders.append(seg.name)
                    if self.use_old_segmentations:
                        self.folders_to_move.append(seg.name)
            if feature_extractor_ref is not None:
                folders.append("ref")
                if hasattr(feature_extractor_ref, "extra_folders"):
                    folders.extend(feature_extractor_ref.extra_folders)
            create_folders(self.storage_path, folders)

    # ------------------------------------------------------------------
    # OME-Zarr reading helpers
    # ------------------------------------------------------------------

    def _read_zarr_raw(self, timestep: int, fov: int) -> np.ndarray:
        """Read a raw frame from the zarr store, returning (c, y, x) array."""
        axes = self._zarr_axes
        has_p = "p" in axes
        has_c = "c" in axes
        arr = self._zarr_raw

        if has_p and has_c:
            img = np.asarray(arr[timestep, fov])  # (c, y, x)
        elif has_p:
            img = np.asarray(arr[timestep, fov])[np.newaxis]  # (1, y, x)
        elif has_c:
            img = np.asarray(arr[timestep])  # (c, y, x)
        else:
            img = np.asarray(arr[timestep])[np.newaxis]  # (1, y, x)
        return img

    def _read_zarr_label(
        self, label_name: str, timestep: int, fov: int
    ) -> np.ndarray | None:
        """Read a label frame from the zarr store, returning (y, x) array."""
        store = self._zarr_store
        lbl_path = f"labels/{label_name}/0"
        try:
            lbl_arr = store[lbl_path]
        except KeyError:
            return None
        has_p = "p" in self._zarr_axes
        if has_p:
            return np.asarray(lbl_arr[timestep, fov])
        return np.asarray(lbl_arr[timestep])

    def _init_omezarr_writer(self) -> None:
        """Initialize the OmeZarrWriter for reanalysis.

        Instead of creating a new raw array via ``init_stream``, this
        hardlinks the raw resolution-level directories (``0/``, ``1/``, …)
        from the source zarr into the output zarr and copies the root
        ``zarr.json``.  Only label arrays are written fresh by the writer.
        """
        import shutil
        from rtm_pymmcore.core.writers import _extract_positions_from_events

        events = self._events
        position_names = _extract_positions_from_events(events)

        # Read one frame to determine image dimensions
        row = self.df_acquire.iloc[0]
        t, fov = int(row["timestep"]), int(row["fov"])
        if self._use_zarr:
            sample = self._read_zarr_raw(t, fov)
        else:
            sample = tifffile.imread(
                os.path.join(self.img_storage_path, "raw", row["fname"] + ".tiff")
            )
        image_height, image_width = sample.shape[-2], sample.shape[-1]

        # Set writer attributes needed for label creation (skip init_stream
        # so no raw array is created in the output store).
        self._writer._position_names = position_names
        self._writer._image_height = image_height
        self._writer._image_width = image_width

        # Hardlink raw resolution levels from source zarr into output zarr
        src_zarr = os.path.join(self.img_storage_path, "acquisition.ome.zarr")
        dst_zarr = self._writer._zarr_path

        # Clean output zarr if it exists (init_stream normally does this,
        # but we skip it). Prevents stale labels from a previous run.
        if self._writer._overwrite and os.path.isdir(dst_zarr):
            shutil.rmtree(dst_zarr)
        os.makedirs(dst_zarr, exist_ok=True)

        # Copy root zarr.json (metadata for multiscales, omero, etc.)
        src_meta = os.path.join(src_zarr, "zarr.json")
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, os.path.join(dst_zarr, "zarr.json"))

        # Hardlink all numbered resolution directories (0/, 1/, 2/, …)
        # Labels are NOT hardlinked — they are written fresh by the writer.
        for entry in os.listdir(src_zarr):
            if entry.isdigit():
                self._hardlink_tree(
                    os.path.join(src_zarr, entry),
                    os.path.join(dst_zarr, entry),
                )

    @staticmethod
    def _hardlink_tree(src_dir: str, dst_dir: str) -> None:
        """Recursively hardlink all files from *src_dir* into *dst_dir*.

        Falls back to copying if hardlinking fails (e.g. on SMB shares).
        """
        import shutil

        for dirpath, _dirnames, filenames in os.walk(src_dir):
            rel = os.path.relpath(dirpath, src_dir)
            dst_sub = os.path.join(dst_dir, rel)
            os.makedirs(dst_sub, exist_ok=True)
            for fname in filenames:
                src_file = os.path.join(dirpath, fname)
                dst_file = os.path.join(dst_sub, fname)
                if not os.path.exists(dst_file):
                    ImageProcessingPipeline_postExperiment._link_or_copy(
                        src_file, dst_file
                    )

    @staticmethod
    def _link_or_copy(src: str, dst: str) -> None:
        """Hardlink a file; fall back to copy on filesystems that don't
        support hardlinks (e.g. SMB network shares)."""
        import shutil

        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def run(self):
        # Initialize OmeZarrWriter stream if needed
        if self._writer_is_omezarr and self._events is not None:
            self._init_omezarr_writer()

        unique_fovs = self.df_acquire["fov"].unique()
        max_workers = min(self.n_jobs, len(unique_fovs))  # Limit number of threads
        if self.n_jobs == 1:
            for fov_id in unique_fovs:
                print(f"Processing FOV {fov_id}")
                self.run_on_fov(fov_id)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_on_fov, fov_id): fov_id
                    for fov_id in unique_fovs
                }
                for future in as_completed(futures):
                    fov_id = futures[future]
                    try:
                        future.result()
                        print(f"Finished processing FOV {fov_id}")
                    except Exception as e:
                        print(f"Error processing FOV {fov_id}: {str(e)}")

        # Save events.json to output and close writer
        if self._writer is not None:
            if self._events is not None:
                self._writer.save_events(self._events)
            self._writer.close()

        print("Finished processing all FOVs.")

    def run_on_fov(self, fov_id) -> dict:
        df = (
            self.df_acquire.query("fov ==  @fov_id")
            .drop_duplicates(subset=["fname"])
            .reset_index()
            .copy()
        )

        cell_specific_cols = [
            c
            for c in df.columns
            if c.startswith("mean_intensity_")
            or c.startswith("median_intensity_")
            or c.startswith("cnr_")
            or c.startswith("cnr")
            or c.startswith("norm_")
            or c.startswith("mean_")
            or c in ["x", "y", "area", "label", "particle"]
            or c.startswith("ref_mean_intensity_")
            or c.startswith("ref_median_intensity_")
        ]
        if cell_specific_cols:
            df = df.drop(columns=cell_specific_cols)

        df_old = pd.DataFrame()
        fov_obj = FovState()
        df["fov"] = fov_id
        if self.correct_timestep_jumps:
            # ensure missing timesteps are filled for this FOV by backfilling all missing frames
            # e.g. if timestep 176 exists but many earlier timesteps are missing, add 0,1,2,...,175
            all_timesteps = sorted(self.df_acquire["timestep"].unique())
            existing_ts = sorted(df["timestep"].unique())
            existing_set = set(int(x) for x in existing_ts)
            # consider the full acquisition range and find missing timesteps
            full_range = range(int(min(all_timesteps)), int(max(all_timesteps)) + 1)
            missing_ts = [t for t in full_range if t not in existing_set]
            if len(missing_ts) > 0:
                print(f"Backfilling missing timesteps for FOV {fov_id}: {missing_ts}")
                # representative row per existing timestep (use the row at that timestep as template)
                rep_rows = {int(r["timestep"]): r for _, r in df.iterrows()}
                sorted_existing = sorted(rep_rows.keys())
                fov = int(df["fname"][0].split("_")[0])

                new_rows = []
                for target in missing_ts:
                    # prefer the nearest later existing timestep as template, otherwise use nearest earlier
                    later = [t for t in sorted_existing if t > target]
                    if later:
                        rep_t = later[0]
                    else:
                        earlier = [t for t in sorted_existing if t < target]
                        if earlier:
                            rep_t = earlier[-1]
                        else:
                            # no representative available for this FOV, skip
                            continue

                    rep = rep_rows[rep_t].copy()
                    rep["timestep"] = int(target)
                    # fov_obj is tracked separately now
                    rep["fname"] = f"{fov:03d}_{int(target):05d}"
                    # point fname to the representative image so reads succeed
                    new_rows.append(rep)
                    existing_set.add(target)

                if new_rows:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    df = df.sort_values(by="timestep").reset_index(drop=True)

        for index, row in df.iterrows():
            t = int(row["timestep"])
            if self._use_zarr:
                img = self._read_zarr_raw(t, int(fov_id))
            else:
                img = tifffile.imread(
                    os.path.join(self.img_storage_path, "raw", row["fname"] + ".tiff")
                )
            metadata = row.to_dict()
            metadata["time_acquired"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            shape_img = (img.shape[-2], img.shape[-1])
            metadata["img_shape"] = shape_img
            masks_for_fe = None

            segmentation_results = {}
            if self.use_old_segmentations:
                for seg in self.segmentators:
                    if self._use_zarr:
                        lbl = self._read_zarr_label(seg.name, t, int(fov_id))
                        if lbl is not None:
                            segmentation_results[seg.name] = lbl
                        else:
                            segmentation_results[seg.name] = tifffile.imread(
                                os.path.join(
                                    self.img_storage_path,
                                    seg.name,
                                    row["fname"] + ".tiff",
                                )
                            )
                    else:
                        segmentation_results[seg.name] = tifffile.imread(
                            os.path.join(
                                self.img_storage_path, seg.name, row["fname"] + ".tiff"
                            )
                        )
            else:
                for seg in self.segmentators:
                    segmentation_results[seg.name] = seg.segmentation_class.segment(
                        img[seg.use_channel, :, :]
                    )

            # 1. Extract positions (label, x, y) for tracking
            df_new = build_frame_dataframe(
                self.feature_extractor, segmentation_results, metadata
            )

            # 2. Track
            df_tracked = run_tracking(self.tracker, df_old, df_new, fov_obj)

            # 3. Feature extraction (now has access to df_tracked)
            masks_for_fe = extract_and_merge_features(
                self.feature_extractor, segmentation_results, img, df_tracked, metadata
            )
            df_old = df_tracked
            fov_obj.fov_timestep_counter += 1

            if self.feature_extractor_ref is not None and self.tracker is not None:
                if metadata.get("img_type") == ImgType.IMG_REF or metadata.get(
                    "ref", False
                ):
                    print(
                        f'Adding ref features for timestep {metadata["timestep"]}, fov {fov_id}'
                    )
                    if self._use_zarr:
                        # Ref images fall back to TIFF even with zarr writer
                        ref_tiff = os.path.join(
                            self.img_storage_path, "ref", row["fname"] + ".tiff"
                        )
                        if os.path.exists(ref_tiff):
                            img_ref = tifffile.imread(ref_tiff)
                        else:
                            continue
                    else:
                        img_ref = tifffile.imread(
                            os.path.join(
                                self.img_storage_path, "ref", row["fname"] + ".tiff"
                            )
                        )
                    df_tracked = self.feature_extractor_ref.extract_features(
                        segmentation_results,
                        img_ref,
                        df_tracked,
                        metadata,
                    )

            if masks_for_fe is not None:
                for mask_fe in masks_for_fe:
                    for key, value in mask_fe.items():
                        store_img(
                            value,
                            metadata,
                            self.storage_path,
                            key,
                            writer=self._writer,
                        )

            if not self.use_old_stim_masks and self.stimulator is not None:
                if metadata.get("stim", False):
                    stim_mask = dispatch_stim_mask(
                        self.stimulator,
                        segmentation_results,
                        metadata,
                        img=img,
                    )
                    store_img(
                        stim_mask,
                        metadata,
                        self.storage_path,
                        "stim_mask",
                        writer=self._writer,
                    )
                else:
                    store_img(
                        np.zeros(shape_img, np.uint8),
                        metadata,
                        self.storage_path,
                        "stim_mask",
                        writer=self._writer,
                    )

            save_segmentation_results(
                segmentation_results,
                self.segmentators,
                self.tracker,
                df_tracked,
                metadata,
                self.storage_path,
                save_labels=not self.use_old_segmentations,
                writer=self._writer,
            )

            # Copy/write reused folders to output (raw images are NOT copied —
            # the reanalysis output only contains masks, tracks, and stim data).
            if self._writer is not None:
                # For folders we are reusing (old segmentations, old stim masks),
                # the labels were already written above via save_segmentation_results
                # or store_img. For stim readout images we copy them through the writer.
                if self.use_old_stim_masks:
                    if self._use_zarr:
                        lbl = self._read_zarr_label("stim_mask", t, int(fov_id))
                        if lbl is not None:
                            store_img(
                                lbl,
                                metadata,
                                self.storage_path,
                                "stim_mask",
                                writer=self._writer,
                            )
                    elif not self._writer_is_omezarr:
                        # TIFF→TIFF: hardlink as before
                        for folder in ["stim_mask", "stim"]:
                            src_path = os.path.join(
                                self.img_storage_path, folder, row["fname"] + ".tiff"
                            )
                            dst_path = os.path.join(
                                self.storage_path, folder, row["fname"] + ".tiff"
                            )
                            if os.path.exists(src_path) and not os.path.exists(
                                dst_path
                            ):
                                self._link_or_copy(src_path, dst_path)
            else:
                # Legacy path: no writer — hardlink TIFF files
                if not self._use_zarr:
                    for folder in self.folders_to_move:
                        src_path = os.path.join(
                            self.img_storage_path, folder, row["fname"] + ".tiff"
                        )
                        dst_path = os.path.join(
                            self.storage_path, folder, row["fname"] + ".tiff"
                        )
                        if os.path.exists(src_path):
                            if not os.path.exists(dst_path):
                                self._link_or_copy(src_path, dst_path)

        df_tracked = convert_track_dtypes(df_tracked)

        filename_for_parquet = f"{metadata['fov']}_latest.parquet"
        if "phase_id" in metadata or "phase_name" in metadata:
            metadata["fov_timestep"] = fov_obj.fov_timestep_counter
            filename_for_parquet = (
                f"{metadata['fov']}_phase_{metadata['phase_id']}_latest.parquet"
            )

        df_tracked = self.reduce_df_to_float32(df_tracked)
        df_tracked.to_parquet(
            os.path.join(self.storage_path, "tracks", filename_for_parquet),
            compression="zstd",
        )

        return df_tracked

    def reduce_df_to_float32(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce the memory usage of a DataFrame by converting float64 to float32.
        """
        float64_cols = df.select_dtypes(include="float64").columns
        df[float64_cols] = df[float64_cols].astype("float32")
        return df

    def concat_fovs(self):

        fovs_i_list = os.listdir(os.path.join(self.storage_path, "tracks"))
        fovs_i_list.sort()
        dfs = []

        for fov_i in fovs_i_list:

            track_file = os.path.join(self.storage_path, "tracks", fov_i)
            df = pd.read_parquet(track_file)
            dfs.append(df)

        dfs = pd.concat(dfs)
        dfs = self.reduce_df_to_float32(dfs)

        dfs.to_parquet(
            os.path.join(self.storage_path, "exp_data.parquet"), compression="zstd"
        )
