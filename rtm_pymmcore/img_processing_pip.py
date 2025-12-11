# combines a segmentor, stimulator and tracker into a image processing pipeline.

import os
from typing import List

import numpy as np
import pandas as pd
import tifffile
from useq import MDAEvent

import rtm_pymmcore.segmentation.base_segmentation as base_segmentation
import rtm_pymmcore.stimulation.base_stimulation as base_stimulation
import rtm_pymmcore.tracking.abstract_tracker as abstract_tracker
import rtm_pymmcore.feature_extraction.abstract_fe as abstract_fe
from rtm_pymmcore.data_structures import Fov, ImgType, SegmentationMethod
from rtm_pymmcore.utils import labels_to_particles, create_folders
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import queue


def store_img(img: np.array, metadata, path: str, folder: str):
    """Take the image and store it accordingly. Check the metadata for FOV index and timestamp."""
    fname = metadata["fname"]
    tifffile.imwrite(
        os.path.join(path, folder, fname + ".tiff"),
        img,
        compression="zlib",
        compressionargs={"level": 5},
    )


# Create a new pipeline class that contains a segmentator and a stimulator
class ImageProcessingPipeline:
    def __init__(
        self,
        storage_path: str,
        segmentators: List[SegmentationMethod] = None,
        feature_extractor: abstract_fe.FeatureExtractor = None,
        stimulator: base_stimulation.Stim = None,
        tracker: abstract_tracker.Tracker = None,
        feature_extractor_optocheck: abstract_fe.FeatureExtractor = None,
    ):
        self.segmentators = segmentators
        self.feature_extractor = feature_extractor
        self.stimulator = stimulator
        self.tracker = tracker
        self.feature_extractor_optocheck = feature_extractor_optocheck
        self.storage_path = storage_path
        folders = ["raw", "tracks"]
        if self.stimulator is not None:
            folders.extend(["stim_mask", "stim"])
        if self.tracker is not None:
            folders.append("particles")
        if self.feature_extractor is not None:
            if hasattr(self.feature_extractor, "extra_folders"):
                folders.extend(self.feature_extractor.extra_folders)
        if self.segmentators is not None:
            for seg in self.segmentators:
                folders.append(seg.name)
        if feature_extractor_optocheck is not None:
            folders.append("optocheck")
            if hasattr(feature_extractor_optocheck, "extra_folders"):
                folders.extend(feature_extractor_optocheck.extra_folders)
        create_folders(self.storage_path, folders)

    def run(
        self, img: np.ndarray = None, event: MDAEvent = None, file_path: str = None
    ) -> dict:
        """
        Runs the image processing pipeline on the input image.

        Args:
            img (np.ndarray, optional): The input image to process (loaded in memory).
            event (MDAEvent, optional): The MDAEvent used to capture the image, which also contains the metadata.
            file_path (str, optional): Path to a saved image file (TIFF format). If provided, image is loaded from disk.
                                      Either `img` or `file_path` must be provided, not both.

        Returns:
            dict: A dictionary containing the result of the pipeline.

        Pipeline Steps:
        1. Extract metadata from the event object.
        2. Segment the image using the segmentator.
        3. Extract features from the segmented image.
        4. Add frame-related information to the extracted features.
        5. Initialize (frame 0) or run the tracker.
        6. Remove duplicate tracks in the tracker.
        7. If stimulation is enabled, get the stimulated labels and mask.
        8. Store the intermediate tracks dataframe.
        9. Store the segmented images and labels.
        """

        # Validate inputs: exactly one of img or file_path must be provided
        if (img is None and file_path is None) or (
            img is not None and file_path is not None
        ):
            raise ValueError("Exactly one of 'img' or 'file_path' must be provided")

        # Load image from file if file_path is provided
        if file_path is not None:
            img = tifffile.imread(file_path)

        metadata = event.metadata
        metadata["time_acquired"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        fov_obj: Fov = metadata["fov_object"]
        if self.stimulator.use_labels == False:
            timeout_time = 60
        else:
            timeout_time = 20

        try:
            # First attempt: pull from queue
            df_old = fov_obj.tracks_queue.get(block=True, timeout=timeout_time)

        except Exception as e:
            print("Exception", e)
            # If queue is empty → fallback to file-based recovery
            current_fname = f"{metadata['fname']}.parquet"
            base = current_fname.rsplit("_", 1)[0]

            # Try frame -1, then frame -2
            for offset in (1, 2):
                frame_num = int(metadata["timestep"]) - offset
                if frame_num < 0:
                    continue  # avoid negative filenames

                fname = f"{base}_{str(frame_num).zfill(5)}.parquet"
                file_path = os.path.join(self.storage_path, "tracks", fname)

                try:
                    df_old = pd.read_parquet(file_path)
                    break  # success → exit loop
                except FileNotFoundError:
                    continue  # try the next offset

            else:
                # Neither file exists → return empty DataFrame
                df_old = pd.DataFrame()
                print("Attention df lost")

        if "phase_id" or "phase_name" in metadata:
            metadata["fov_timestep"] = fov_obj.fov_timestep_counter

        if metadata["img_type"] == ImgType.IMG_OPTOCHECK:
            n_optocheck_channels = len(metadata["optocheck_channels"])
            n_channels = len(metadata["channels"])
            img_optocheck = img[n_channels:]
            img = img[:n_channels]

        shape_img = (img.shape[-2], img.shape[-1])

        segmentation_results = {}
        if self.segmentators is not None:
            for seg in self.segmentators:
                segmentation_results[seg.name] = seg.segmentation_class.segment(
                    img[seg.use_channel, :, :]
                )

        # if metadata["stim"] == True:
        #     stim_mask, labels_stim = self.stimulator.get_stim_mask(
        #         segmentation_results, metadata, img
        #     )
        #     fov_obj.stim_mask_queue.put_nowait(stim_mask)
        # if labels_stim is not None:
        #     labels_stim = np.unique(labels_stim[labels_stim > 0])
        #     metadata["stim_labels"] = labels_stim
        # TODO: Reenable, but make exception for stimwholeframe
        # mark in the df which cells have been stimulated
        # stim_index = np.where((df_tracked['frame']==metadata['timestep']) & (df_tracked['label'].isin(labels_stim)))[0]
        # df_tracked.loc[stim_index,'stim']=True

        if self.feature_extractor is None:
            df_new = pd.DataFrame([metadata])
            masks_for_fe = None
        else:
            df_new, masks_for_fe = self.feature_extractor.extract_features(
                segmentation_results, img
            )
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    df_new[key] = pd.Series([value] * len(df_new))
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        df_new[subkey] = [subvalue] * len(df_new)
                else:
                    df_new[key] = value

        if self.tracker is not None:
            df_tracked = self.tracker.track_cells(df_old, df_new, metadata)
        else:
            df_tracked = pd.concat([df_old, df_new], ignore_index=True)

        if metadata["stim"] == True:
            stim_mask, _ = self.stimulator.get_stim_mask(
                label_images=segmentation_results, metadata=metadata, img=img
            )
            if self.stimulator.use_labels:
                fov_obj.stim_mask_queue.put_nowait(stim_mask)

        if metadata["img_type"] == ImgType.IMG_OPTOCHECK:
            if (
                self.feature_extractor_optocheck is not None
                and self.tracker is not None
            ):
                df_tracked = self.feature_extractor_optocheck.extract_features(
                    segmentation_results,
                    img_optocheck,
                    df_tracked,
                    metadata,
                )
        fov_obj.tracks_queue.put(df_tracked)
        fov_obj.fov_timestep_counter += 1

        if not df_tracked.empty:
            df_tracked = df_tracked.drop(
                columns=["fov_object", "img_type", "last_channel"], errors="ignore"
            )

        df_datatypes = {
            "timestep": np.uint32,
            "particle": np.uint32,
            "label": np.uint32,
            "time": np.float32,
            "fov": np.uint16,
            "stim_exposure": np.float32,
        }

        # Only convert columns that exist in the dataframe
        existing_columns = {
            col: dtype
            for col, dtype in df_datatypes.items()
            if col in df_tracked.columns
        }

        try:
            df_tracked = df_tracked.astype(existing_columns)
        except ValueError as e:
            print(e)
            print("Error in converting datatypes. df_tracked:")

        df_tracked.to_parquet(
            os.path.join(self.storage_path, "tracks", f"{metadata['fname']}.parquet")
        )

        if self.stimulator is not None:
            if metadata["stim"]:
                store_img(stim_mask, metadata, self.storage_path, "stim_mask")
            else:
                store_img(
                    np.zeros(shape_img, np.uint8),
                    metadata,
                    self.storage_path,
                    "stim_mask",
                )
                store_img(
                    np.zeros(shape_img, np.uint8), metadata, self.storage_path, "stim"
                )

        if masks_for_fe is not None:
            for mask_fe in masks_for_fe:
                for key, value in mask_fe.items():
                    store_img(value, metadata, self.storage_path, key)

        if self.tracker is None:
            for key, value in segmentation_results.items():
                store_img(value, metadata, self.storage_path, key)
        else:
            for (key, value), segmentator in zip(
                segmentation_results.items(), self.segmentators
            ):
                if segmentator.save_tracked:
                    tracked_label = labels_to_particles(
                        value, df_tracked, metadata=metadata
                    )
                    store_img(tracked_label, metadata, self.storage_path, "particles")
                    store_img(value, metadata, self.storage_path, key)
                else:
                    store_img(value, metadata, self.storage_path, key)

        # cleanup: delete the previous pickled tracks file
        if metadata["timestep"] > 0:
            current_fname = f"{metadata['fname']}.parquet"
            get_last_frame_number = int(metadata["timestep"]) - 1
            get_fname_wo_f_number = current_fname.rsplit("_", 1)[0]
            fname_previous = (
                f"{get_fname_wo_f_number}_{str(get_last_frame_number).zfill(5)}.parquet"
            )
            os.remove(os.path.join(self.storage_path, "tracks", fname_previous))
        return {"result": "STOP"}


class ImageProcessingPipeline_postExperiment:
    def __init__(
        self,
        img_storage_path: str,
        out_path: str,
        df_acquire: pd.DataFrame,
        segmentators: List[SegmentationMethod] = None,
        feature_extractor: abstract_fe.FeatureExtractor = None,
        tracker: abstract_tracker.Tracker = None,
        feature_extractor_optocheck: abstract_fe.FeatureExtractor = None,
        use_old_segmentations: bool = False,
        n_jobs: int = 2,
        correct_timestep_jumps: bool = False,
    ):
        self.segmentators = segmentators
        self.feature_extractor = feature_extractor
        self.tracker = tracker
        self.df_acquire = df_acquire
        self.feature_extractor_optocheck = feature_extractor_optocheck
        self.storage_path = out_path
        self.img_storage_path = img_storage_path
        self.use_old_segmentations = use_old_segmentations
        self.n_jobs = n_jobs
        self.correct_timestep_jumps = correct_timestep_jumps

        folders = ["raw", "tracks"]
        if self.tracker is not None:
            folders.append("particles")
        if self.feature_extractor is not None:
            if hasattr(self.feature_extractor, "extra_folders"):
                folders.extend(self.feature_extractor.extra_folders)
        if self.segmentators is not None:
            for seg in self.segmentators:
                folders.append(seg.name)
        if feature_extractor_optocheck is not None:
            folders.append("optocheck")
            if hasattr(feature_extractor_optocheck, "extra_folders"):
                folders.extend(feature_extractor_optocheck.extra_folders)
        create_folders(self.storage_path, folders)

    def run(self):
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
            or c.startswith("optocheck_mean_intensity_")
            or c.startswith("optocheck_median_intensity_")
        ]
        if cell_specific_cols:
            df = df.drop(columns=cell_specific_cols)

        df_old = pd.DataFrame()
        fov_obj = Fov(0)
        df.loc[:, "fov_object"] = fov_obj
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
                    rep["fov_object"] = fov_obj
                    rep["fname"] = f"{fov:03d}_{int(target):05d}"
                    # point fname to the representative image so reads succeed
                    new_rows.append(rep)
                    existing_set.add(target)

                if new_rows:
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    df = df.sort_values(by="timestep").reset_index(drop=True)

        for index, row in df.iterrows():
            img = tifffile.imread(
                os.path.join(self.img_storage_path, "raw", row["fname"] + ".tiff")
            )
            metadata = row.to_dict()
            shape_img = (img.shape[-2], img.shape[-1])
            segmentation_results = {}
            if self.use_old_segmentations:
                for seg in self.segmentators:
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

            if self.feature_extractor is not None:
                df_new, masks_for_fe = self.feature_extractor.extract_features(
                    segmentation_results, img
                )
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        df_new[key] = pd.Series([value] * len(df_new))
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            df_new[subkey] = [subvalue] * len(df_new)
                    else:
                        df_new[key] = value

            if self.tracker is not None:
                df_tracked = self.tracker.track_cells(df_old, df_new, metadata)
            else:
                df_tracked = pd.concat([df_old, df_new], ignore_index=True)
            df_old = df_tracked

            if (
                self.feature_extractor_optocheck is not None
                and self.tracker is not None
            ):
                if metadata["optocheck"] == True:
                    print(
                        f'Adding optocheck features for timestep {metadata["timestep"]}, fov {fov_id}'
                    )
                    img_optocheck = tifffile.imread(
                        os.path.join(
                            self.img_storage_path, "optocheck", row["fname"] + ".tiff"
                        )
                    )
                    n_channels = len(metadata["channels"])
                    img_optocheck = img_optocheck[n_channels:]
                    df_tracked = self.feature_extractor_optocheck.extract_features(
                        segmentation_results,
                        img_optocheck,
                        df_tracked,
                        metadata,
                    )

            if masks_for_fe is not None:
                for mask_fe in masks_for_fe:
                    for key, value in mask_fe.items():
                        store_img(value, metadata, self.storage_path, key)

            if self.tracker is None:
                for key, value in segmentation_results.items():
                    store_img(value, metadata, self.storage_path, key)
            else:
                for (key, value), segmentator in zip(
                    segmentation_results.items(), self.segmentators
                ):
                    if segmentator.save_tracked:
                        tracked_label = labels_to_particles(value, df_tracked, metadata)
                        store_img(
                            tracked_label, metadata, self.storage_path, "particles"
                        )
                        if not self.use_old_segmentations:
                            store_img(value, metadata, self.storage_path, key)
                    else:
                        if not self.use_old_segmentations:
                            store_img(value, metadata, self.storage_path, key)

        df_tracked.drop(columns=["fov_object"], inplace=True)
        df_tracked.to_parquet(
            os.path.join(self.storage_path, "tracks", f"{metadata['fname']}.parquet"),
            compression="zstd",
        )

        return df_tracked

    def concat_fovs(self):

        fovs_i_list = os.listdir(os.path.join(self.storage_path, "tracks"))
        fovs_i_list.sort()
        dfs = []

        for fov_i in fovs_i_list:

            track_file = os.path.join(self.storage_path, "tracks", fov_i)
            df = pd.read_parquet(track_file)
            dfs.append(df)

        pd.concat(dfs).to_parquet(os.path.join(self.storage_path, "exp_data.parquet"))
