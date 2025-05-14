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
from rtm_pymmcore.data_structures import Fov, ImgType
from rtm_pymmcore.utils import labels_to_particles, create_folders


def store_img(img: np.array, metadata, path: str, folder: str):
    """Take the image and store it accordingly. Check the metadata for FOV index and timestamp."""
    img_type = metadata["img_type"]
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
        segmentators: List[base_segmentation.Segmentator] = None,
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
                folders.append(seg["name"])
        if feature_extractor_optocheck is not None:
            folders.append("optocheck")
            if hasattr(feature_extractor_optocheck, "extra_folders"):
                folders.extend(feature_extractor_optocheck.extra_folders)
        create_folders(self.storage_path, folders)

    def run(self, img: np.ndarray, event: MDAEvent) -> dict:
        """
        Runs the image processing pipeline on the input image.

        Args:
            img (np.ndarray): The input image to process.
            event (MDAEvent): The MDAEvent used to capture the image, which also containins the metadata.

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

        # Rest of the code...

        metadata = event.metadata

        fov_obj: Fov = metadata["fov_object"]
        if metadata["timestep"] > 0:
            df_old = fov_obj.tracks_queue.get(block=True, timeout=360)
        else:
            df_old = pd.DataFrame()

        if metadata["img_type"] == ImgType.IMG_OPTOCHECK:
            n_optocheck_channels = len(metadata["optocheck_channels"])
            n_channels = len(metadata["channels"])
            img_optocheck = img[n_channels:]
            img = img[:n_channels]

        shape_img = (img.shape[-2], img.shape[-1])

        segmentation_results = {}
        for seg in self.segmentators:
            segmentation_results[seg["name"]] = seg["class"].segment(
                img[seg["use_channel"], :, :]
            )

        if metadata["stim"] == True:
            stim_mask, labels_stim = self.stimulator.get_stim_mask(
                segmentation_results, metadata, img
            )
            fov_obj.stim_mask_queue.put_nowait(stim_mask)
            # if labels_stim is not None:
            #     labels_stim = np.unique(labels_stim[labels_stim > 0])
            #     metadata["stim_labels"] = labels_stim
            # TODO: Reenable, but make exception for stimwholeframe
            # mark in the df which cells have been stimulated
            # stim_index = np.where((df_tracked['frame']==metadata['timestep']) & (df_tracked['label'].isin(labels_stim)))[0]
            # df_tracked.loc[stim_index,'stim']=True

        if self.feature_extractor is None:
            df_new = pd.DataFrame([metadata])
            df_new = pd.concat([df_old, df_new], ignore_index=True)
            masks_for_fe = None
        else:
            if metadata["stim"] == True:
                df_new, masks_for_fe = self.feature_extractor.extract_features(
                    segmentation_results, img, stim_mask
                )
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

        if metadata["img_type"] == ImgType.IMG_OPTOCHECK:
            if (
                self.feature_extractor_optocheck is not None
                and self.tracker is not None
            ):
                df_tracked = self.feature_extractor_optocheck.extract_features(
                    segmentation_results, img_optocheck, df_tracked
                )
        fov_obj.tracks_queue.put(df_tracked)

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
                if segmentator.get("save_tracked", False):
                    tracked_label = labels_to_particles(value, df_tracked)
                    store_img(tracked_label, metadata, self.storage_path, "particles")
                    store_img(value, metadata, self.storage_path, key)
                else:
                    store_img(value, metadata, self.storage_path, key)

        # cleanup: delete the previous pickled tracks file
        if metadata["timestep"] > 0:
            fname_previous = f'{str(fov_obj.index).zfill(3)}_{str(metadata["timestep"]-1).zfill(5)}.parquet'
            os.remove(os.path.join(self.storage_path, "tracks", fname_previous))

        return {"result": "STOP"}
