# combines a segmentor, stimulator and tracker into a image processing pipeline.

import inspect
import os
from typing import List

import numpy as np
import pandas as pd
import tifffile
from useq import MDAEvent

import faro.segmentation.base as base_segmentation
import faro.stimulation.base as base_stimulation
from faro.stimulation.base import StimWithImage, StimWithPipeline
import faro.tracking.base as abstract_tracker
import faro.feature_extraction.base as abstract_fe
from faro.core.data_structures import FovState, ImgType, SegmentationMethod
from faro.core.utils import labels_to_particles, create_folders
from datetime import datetime
import queue


def store_img(img: np.array, metadata, path: str, folder: str, *, writer=None):
    """Take the image and store it accordingly. Check the metadata for FOV index and timestamp.

    If *writer* is provided, delegates to the writer backend.
    Otherwise falls back to direct TIFF write (legacy path).
    """
    if isinstance(img, (bool, np.bool_)):
        h, w = metadata["img_shape"]
        img = np.full((h, w), int(img), dtype=np.uint8)
    if writer is not None:
        writer.write(img, metadata, folder)
        return
    fname = metadata["fname"]
    tifffile.imwrite(
        os.path.join(path, folder, fname + ".tiff"),
        img,
        compression="zlib",
        compressionargs={"level": 5},
    )


def build_frame_dataframe(feature_extractor, segmentation_results, metadata):
    """Extract positions from segmentation results and attach metadata columns."""
    if feature_extractor is not None:
        df_new = feature_extractor.extract_positions(segmentation_results)
        for key, value in metadata.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                df_new[key] = pd.Series([value] * len(df_new))
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    df_new[subkey] = [subvalue] * len(df_new)
            else:
                df_new[key] = value
    else:
        df_new = pd.DataFrame([metadata])
    return df_new


def run_tracking(tracker, df_old, df_new, fov_obj):
    """Call tracker.track_cells() or fall back to pd.concat()."""
    if tracker is not None:
        return tracker.track_cells(df_old, df_new, fov_obj)
    return pd.concat([df_old, df_new], ignore_index=True)


def extract_and_merge_features(
    feature_extractor, segmentation_results, img, df_tracked, metadata
):
    """Run feature extraction and merge features into current-frame rows of df_tracked."""
    if feature_extractor is not None:
        features_df, masks_for_fe = feature_extractor.extract_features(
            segmentation_results, img, df_tracked, metadata
        )
        feature_map = features_df.set_index("label")
        current_mask = df_tracked["fname"] == metadata["fname"]
        for col in feature_map.columns:
            df_tracked.loc[current_mask, col] = df_tracked.loc[
                current_mask, "label"
            ].map(feature_map[col])
        return masks_for_fe
    return None


def dispatch_stim_mask(
    stimulator, segmentation_results, metadata, *, img=None, tracks=None
):
    """Dispatch get_stim_mask() based on stimulator type hierarchy."""
    if isinstance(stimulator, StimWithPipeline):
        stim_mask, _ = stimulator.get_stim_mask(
            label_images=segmentation_results,
            metadata=metadata,
            img=img,
            tracks=tracks,
        )
    elif isinstance(stimulator, StimWithImage):
        stim_mask, _ = stimulator.get_stim_mask(
            metadata=metadata,
            img=img,
        )
    else:
        stim_mask, _ = stimulator.get_stim_mask(
            metadata=metadata,
        )
    return stim_mask


def convert_track_dtypes(df):
    """Drop img_type column and cast standard columns to compact dtypes."""
    if not df.empty:
        df = df.drop(columns=["img_type"], errors="ignore")

    df_datatypes = {
        "timestep": np.uint32,
        "particle": np.uint32,
        "label": np.uint32,
        "time": np.float32,
        "fov": np.uint16,
        "stim_exposure": np.float32,
    }

    existing_columns = {
        col: dtype for col, dtype in df_datatypes.items() if col in df.columns
    }

    try:
        df = df.astype(existing_columns)
    except ValueError as e:
        print(e)
        print("Error in converting datatypes. df_tracked:")

    return df


def save_segmentation_results(
    segmentation_results,
    segmentators,
    tracker,
    df,
    metadata,
    storage_path,
    *,
    save_labels=True,
    writer=None,
):
    """Save segmentation masks and optionally tracked-label images."""
    if tracker is None:
        for key, value in segmentation_results.items():
            store_img(value, metadata, storage_path, key, writer=writer)
    else:
        for (key, value), segmentator in zip(
            segmentation_results.items(), segmentators
        ):
            if segmentator.save_tracked:
                tracked_label = labels_to_particles(value, df, metadata=metadata)
                store_img(
                    tracked_label, metadata, storage_path, "particles", writer=writer
                )
                if save_labels:
                    store_img(value, metadata, storage_path, key, writer=writer)
            else:
                if save_labels:
                    store_img(value, metadata, storage_path, key, writer=writer)


# Create a new pipeline class that contains a segmentator and a stimulator
class ImageProcessingPipeline:
    def __init__(
        self,
        storage_path: str,
        segmentators: List[SegmentationMethod] = None,
        feature_extractor: abstract_fe.FeatureExtractor = None,
        stimulator: base_stimulation.Stim = None,
        tracker: abstract_tracker.Tracker = None,
        feature_extractor_ref: abstract_fe.FeatureExtractor = None,
        only_save_every_n_frames: int = 1,
    ):
        self.segmentators = segmentators
        self.feature_extractor = feature_extractor
        self.stimulator = stimulator
        self.tracker = tracker
        self.feature_extractor_ref = feature_extractor_ref
        self.storage_path = storage_path
        self.only_save_every_n_frames = only_save_every_n_frames
        # Only create tracks/ here — image folders are handled by the writer.
        # TiffWriter creates them on demand; OmeZarrWriter uses the zarr store.
        create_folders(self.storage_path, ["tracks"])
        self._analyzer = None  # set by Analyzer.__init__
        self._writer = None  # set by Analyzer or Controller
        self._queue_timeout: float = 20  # seconds; override in tests

    @staticmethod
    def _check_method_against_base(obj, base_cls, method_name: str) -> list[str]:
        """Check that *obj.method_name* accepts the params declared by *base_cls*.

        The base class defines the canonical signature.  Subclasses must accept
        at least the same parameters (extra params are fine, ``**kwargs`` is
        fine).

        Returns a list of warning strings (empty if OK).
        """
        base_method = getattr(base_cls, method_name, None)
        sub_method = getattr(obj, method_name, None)
        if base_method is None or sub_method is None:
            return []

        try:
            base_sig = inspect.signature(base_method)
            sub_sig = inspect.signature(sub_method)
        except (ValueError, TypeError):
            return []

        # If the subclass accepts **kwargs it can handle anything
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sub_sig.parameters.values()
        ):
            return []

        base_params = {
            name for name, p in base_sig.parameters.items() if name != "self"
        }
        sub_params = {name for name, p in sub_sig.parameters.items() if name != "self"}

        missing = base_params - sub_params
        if missing:
            cls_name = type(obj).__name__
            return [
                f"{cls_name}.{method_name}() is missing parameter(s) "
                f"{missing} declared by {base_cls.__name__}"
            ]
        return []

    def validate_events(self, events) -> bool:
        """Alias for validate_pipeline. Use mic.validate_events() for full validation."""
        return self.validate_pipeline(events)

    def validate_pipeline(self, events) -> bool:
        """Validate pipeline components against events.

        Checks:
        1. Component method signatures match their base class contract.
        2. Events carry required metadata declared by each component.

        Returns True if all checks pass, False otherwise.
        Emits warnings for every problem found.
        """
        warnings_list = []

        # --- Signature checks: subclass must accept base class params ---
        if self.segmentators:
            for seg in self.segmentators:
                warnings_list.extend(
                    self._check_method_against_base(
                        seg.segmentation_class,
                        base_segmentation.Segmentator,
                        "segment",
                    )
                )
        if self.tracker:
            warnings_list.extend(
                self._check_method_against_base(
                    self.tracker,
                    abstract_tracker.Tracker,
                    "track_cells",
                )
            )
        if self.feature_extractor:
            warnings_list.extend(
                self._check_method_against_base(
                    self.feature_extractor,
                    abstract_fe.FeatureExtractor,
                    "extract_features",
                )
            )
        if self.stimulator:
            if isinstance(self.stimulator, StimWithPipeline):
                stim_base = StimWithPipeline
            elif isinstance(self.stimulator, StimWithImage):
                stim_base = StimWithImage
            else:
                stim_base = base_stimulation.Stim
            warnings_list.extend(
                self._check_method_against_base(
                    self.stimulator,
                    stim_base,
                    "get_stim_mask",
                )
            )

        # --- Required metadata checks ---
        general_required = set()
        if self.segmentators:
            for seg in self.segmentators:
                general_required |= getattr(
                    seg.segmentation_class, "required_metadata", set()
                )
        if self.tracker:
            general_required |= getattr(self.tracker, "required_metadata", set())
        if self.feature_extractor:
            general_required |= getattr(
                self.feature_extractor, "required_metadata", set()
            )

        stim_required = set()
        if self.stimulator:
            stim_required = getattr(self.stimulator, "required_metadata", set())

        for event in events:
            meta_keys = set(event.metadata.keys())
            # General requirements (all events)
            missing = general_required - meta_keys
            if missing:
                warnings_list.append(
                    f"Event t={event.index.get('t')} p={event.index.get('p')} "
                    f"missing metadata: {missing}"
                )
            # Stim requirements (only stim events)
            if event.stim_channels:
                missing_stim = stim_required - meta_keys
                if missing_stim:
                    warnings_list.append(
                        f"Stim event t={event.index.get('t')} p={event.index.get('p')} "
                        f"missing stim metadata: {missing_stim}"
                    )

        if warnings_list:
            import warnings as w

            for msg in warnings_list:
                w.warn(msg, UserWarning)
        return len(warnings_list) == 0

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
        fov_obj: FovState = self._analyzer.get_fov_state(metadata["fov"])

        frame_idx = event.index.get("t", 0)

        filename_for_parquet = f"{metadata['fov']}_latest.parquet"
        if "phase_id" in metadata or "phase_name" in metadata:
            metadata["fov_timestep"] = frame_idx
            filename_for_parquet = (
                f"{metadata['fov']}_phase_{metadata['phase_id']}_latest.parquet"
            )

        # Split ref channels off the combined image so that segmentation and
        # regular feature extraction only see imaging channels.
        img_ref = None
        if metadata.get("img_type") == ImgType.IMG_REF:
            n_ref_channels = len(metadata.get("ref_channels", ()))
            n_channels = len(metadata.get("channels", ()))
            if n_ref_channels > 0 and img.shape[0] > n_channels:
                img_ref = img[n_channels:]
                img = img[:n_channels]

        shape_img = (img.shape[-2], img.shape[-1])
        metadata["img_shape"] = shape_img

        # --- Segmentation + position extraction (no dependency on previous frame) ---
        # Run BEFORE waiting for df_old so it overlaps with the previous
        # frame's tracking on another worker thread.
        segmentation_results = {}
        if self.segmentators is not None:
            for seg in self.segmentators:
                segmentation_results[seg.name] = seg.segmentation_class.segment(
                    img[seg.use_channel, :, :]
                )

        # 1. Extract positions (label, x, y) for tracking
        fov_obj.n_cells_latest = int(segmentation_results["labels"].max())
        df_new = build_frame_dataframe(
            self.feature_extractor, segmentation_results, metadata
        )

        # --- Wait for previous frame's tracked DataFrame ---
        if self.stimulator is not None and not isinstance(
            self.stimulator, StimWithPipeline
        ):
            timeout_time = self._queue_timeout * 3
        else:
            timeout_time = self._queue_timeout

        try:
            df_old = fov_obj.tracks_queue.get_predecessor(
                frame_idx, timeout=timeout_time
            )
        except queue.Empty as e:
            print("Exception", e)
            # Predecessor never arrived → fallback to file-based recovery
            file_path = os.path.join(self.storage_path, "tracks", filename_for_parquet)
            try:
                df_old = pd.read_parquet(file_path)
            except FileNotFoundError:
                df_old = None
                print("Attention df lost")
        if df_old is None:
            df_old = pd.DataFrame()

        # The dispenser guarantees only one worker is between get_predecessor
        # and put_for_frame per FOV, so setting the legacy counter here is safe
        # for the tracker to read. If anything below raises, the finally block
        # skips both dispensers so downstream workers don't deadlock.
        fov_obj.fov_timestep_counter = frame_idx
        tracked_ok = False
        stim_expected = metadata["stim"] == True and isinstance(
            self.stimulator, StimWithPipeline
        )
        try:
            df_tracked = run_tracking(self.tracker, df_old, df_new, fov_obj)

            masks_for_fe = extract_and_merge_features(
                self.feature_extractor, segmentation_results, img, df_tracked, metadata
            )

            if metadata["stim"] == True:
                stim_mask = dispatch_stim_mask(
                    self.stimulator,
                    segmentation_results,
                    metadata,
                    img=img,
                    tracks=df_tracked,
                )
                if isinstance(self.stimulator, StimWithPipeline):
                    fov_obj.stim_mask_queue.put_for_frame(frame_idx, stim_mask)

            if metadata.get("img_type") == ImgType.IMG_REF:
                if self.feature_extractor_ref is not None and self.tracker is not None:
                    df_tracked = self.feature_extractor_ref.extract_features(
                        segmentation_results,
                        img_ref if img_ref is not None else img,
                        df_tracked,
                        metadata,
                    )
            tracked_ok = True
        finally:
            if tracked_ok:
                fov_obj.tracks_queue.put_for_frame(frame_idx, df_tracked)
            else:
                fov_obj.tracks_queue.skip_frame(frame_idx)
                if stim_expected:
                    fov_obj.stim_mask_queue.skip_frame(frame_idx)

        # --- Parquet save (after unblocking — doesn't mutate df_tracked) ---
        df_to_save = convert_track_dtypes(df_tracked)

        if frame_idx % self.only_save_every_n_frames == 0 or frame_idx == 0:
            with fov_obj.parquet_lock:
                df_to_save.to_parquet(
                    os.path.join(self.storage_path, "tracks", filename_for_parquet),
                    compression="zstd",
                )

        w = self._writer
        if self.stimulator is not None:
            if metadata["stim"]:
                store_img(stim_mask, metadata, self.storage_path, "stim_mask", writer=w)
            else:
                store_img(
                    np.zeros(shape_img, np.uint8),
                    metadata,
                    self.storage_path,
                    "stim_mask",
                    writer=w,
                )
                store_img(
                    np.zeros(shape_img, np.uint8),
                    metadata,
                    self.storage_path,
                    "stim",
                    writer=w,
                )

        if masks_for_fe is not None:
            for mask_fe in masks_for_fe:
                for key, value in mask_fe.items():
                    store_img(value, metadata, self.storage_path, key, writer=w)

        save_segmentation_results(
            segmentation_results,
            self.segmentators,
            self.tracker,
            df_to_save,
            metadata,
            self.storage_path,
            writer=w,
        )

        return {"result": "STOP"}
