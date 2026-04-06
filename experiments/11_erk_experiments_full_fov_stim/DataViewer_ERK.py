# /// script
# requires-python = "<=3.12"
# dependencies = [
#   "napari[pyqt6] == 0.6.1",
#   "pandas",
#   "magicgui",
#   "dask",
#   "tifffile",
#   "scikit-image",
#   "pyarrow",
#   "matplotlib",
#   "numpy",
#   "napari-timestamper",
#   "opencv-python",
#   "zarr>=3"
# ]
# ///

import napari
import pandas as pd
from magicgui import magicgui
from magicgui.widgets import ComboBox
from typing import List
import os
from tifffile import imread as tiff_imread
import dask.array as da
from skimage.io.collection import alphanumeric_key
import skimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from dask import delayed
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import csv
import zarr


RAW_FOLDER = "raw"
STIM_FOLDER = "stim"
MASK_FOLDER = "mask"
LIGHT_MASK_FOLDER = "light_mask"
PARTICLES_FOLDER = "particles"
LABELS_RINGS = "labels_ring"
TRACKS_FOLDER = "tracks"

DEFAULT_FOLDER = "\\\\izbkingston.izb.unibe.ch\\imaging.data\\PertzLab\\optoRTK_CedricZ\\experimental_data"

# OME-Zarr state
_zarr_store = None
_zarr_raw = None
_zarr_axes = None
_zarr_channel_names = None
_zarr_n_imaging_channels = None  # number of non-stim channels


class Layer_Info:
    def __init__(self, folder_name, layer_type, blending, colormap):
        self.folder_name = folder_name
        self.layer_type = layer_type
        self.blending = blending
        self.colormap = colormap


FOLDERS_TO_LOAD = (
    Layer_Info(STIM_FOLDER, "image", "translucent", "gray_r"),
    Layer_Info(RAW_FOLDER, "image", "translucent", "gray_r"),
    Layer_Info(PARTICLES_FOLDER, "labels", "translucent", None),
    Layer_Info(LABELS_RINGS, "labels", "translucent", None),
)

NORM_UNTIL_TIMEPOINT_DEFAULT = 10  # Default timepoint for normalization

exp_df = None
currently_added_layers = []
timestep_variable_name = None


def load_exp_df(project_path):
    global exp_df
    global timestep_variable_name
    time_col = None
    if exp_df is None and os.path.exists(
        os.path.join(project_path, "exp_data.parquet")
    ):
        try:
            available_columns = pq.read_schema(
                os.path.join(project_path, "exp_data.parquet")
            ).names
            required_columns = [
                "cell_line",
                "stim_exposure",
                "stim_timestep",
                "fov",
                "fname",
                "particle",
                "timestep",
                "cnr",
                "cnr_median",
            ]
            if "fov_timestep" in available_columns:
                required_columns.append("fov_timestep")
                timestep_variable_name = "fov_timestep"
            elif "timestep" in available_columns:
                timestep_variable_name = "timestep"
            elif "frame" in available_columns:
                timestep_variable_name = "frame"
                required_columns.append("frame")
            else:
                timestep_variable_name = None

            exp_df = pd.read_parquet(
                os.path.join(project_path, "exp_data.parquet"),
                columns=required_columns,
            )

        except ValueError:
            exp_df = pd.read_parquet(os.path.join(project_path, "exp_data.parquet"))

        # Normalize stim_timestep to native Python types (None / int / tuple[int,...])
        def _normalize_timestep(x):
            if x is None or (np.isscalar(x) and pd.isna(x)):
                return None
            # already a tuple/list/ndarray -> convert to tuple of ints
            if isinstance(x, (list, tuple, np.ndarray)):
                try:
                    return tuple(int(i) for i in x)
                except Exception:
                    return None
            # single numeric value -> int
            try:
                return int(x)
            except Exception:
                return None

        exp_df["stim_timestep"] = exp_df["stim_timestep"].apply(_normalize_timestep)
        # UID-Spalte hinzufügen
        if "uid" not in exp_df.columns:
            exp_df["uid"] = (
                exp_df["fov"].astype("string")
                + "_"
                + exp_df["particle"].astype("string")
            )
        # Normierung nur falls noch nicht vorhanden
        if time_col is not None:
            for col, base in [("cnr_norm", "cnr"), ("cnr_norm_median", "cnr_median")]:
                if col not in exp_df.columns and base in exp_df.columns:
                    mean_cnr = (
                        exp_df[exp_df[time_col] < NORM_UNTIL_TIMEPOINT_DEFAULT]
                        .groupby("uid")[base]
                        .mean()
                    )
                    exp_df[col] = exp_df.apply(
                        lambda row: row[base] / mean_cnr.get(row["uid"], 1), axis=1
                    )
        # Note: Throughout the application, we filter out entries where stim_exposure is NA
        # to avoid issues with missing stimulation exposure data


def get_cell_lines() -> List[str]:
    load_exp_df(project_path)
    return exp_df["cell_line"].unique().tolist()


def get_exposure_times(cell_line: str) -> List[int]:
    # Filter out NA values in stim_exposure
    filtered_df = exp_df[exp_df["cell_line"] == cell_line].dropna(
        subset=["stim_exposure"]
    )
    exposure_times = filtered_df["stim_exposure"].unique().astype(int).tolist()
    return sorted(exposure_times)


def get_stim_timesteps(cell_line: str, stim_exposure: int):
    # Filter out NA values in stim_exposure
    filtered_df = exp_df.dropna(subset=["stim_exposure"])
    raw = (
        filtered_df[
            (filtered_df["cell_line"] == cell_line)
            & (filtered_df["stim_exposure"] == stim_exposure)
        ]["stim_timestep"]
        .unique()
        .tolist()
    )
    # normalize and keep unique preserving order
    seen = set()
    values = []
    for s in raw:
        if s is None:
            continue
        if isinstance(s, (list, tuple, np.ndarray)):
            try:
                val = tuple(int(x) for x in s)
            except Exception:
                continue
        else:
            try:
                val = int(s)
            except Exception:
                continue
        if val in seen:
            continue
        seen.add(val)
        values.append(val)

    def _timestep_label(v):
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ",".join(str(x) for x in v)
        return str(v)

    # Return list of (label, value) pairs for magicgui ComboBox
    choices = []
    for v in values:
        choices.append((_timestep_label(v), v))
    return choices


def get_fov_choices(cell_line: str, stim_exposure: int, stim_timestep) -> List[str]:
    # Filter out NA values in stim_exposure
    filtered_df = exp_df.dropna(subset=["stim_exposure"])
    return (
        filtered_df[
            (filtered_df["cell_line"] == cell_line)
            & (filtered_df["stim_exposure"] == stim_exposure)
            & (filtered_df["stim_timestep"] == stim_timestep)
        ]["fov"]
        .unique()
        .tolist()
    )


def tiff_to_da(folder, filenames, lazy=True, num_workers=8):
    if len(filenames) > 1:
        filenames = sorted(filenames, key=alphanumeric_key)
    filenames = [os.path.join(project_path, folder, fn + ".tiff") for fn in filenames]
    if lazy:
        # open first image to get the shape
        first_image = tiff_imread(filenames[0])
        shape_ = first_image.shape
        dtype_ = first_image.dtype
        # Using Dask's delayed execution model for lazy loading
        lazy_imread = delayed(tiff_imread)  # Using tifffile for lazy reading
        lazy_arrays = [lazy_imread(fn) for fn in filenames]
        stack = da.stack(
            [da.from_delayed(la, shape=shape_, dtype=dtype_) for la in lazy_arrays],
            axis=0,
        )
    else:
        # Use ThreadPoolExecutor to load images concurrently
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Mapping each future to its index to preserve order
            future_to_index = {
                executor.submit(tiff_imread, fn): i for i, fn in enumerate(filenames)
            }
            results = [None] * len(
                filenames
            )  # Pre-allocate the result list to preserve order
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                # Place the result in the correct order
                results[index] = future.result()
            # Stack images into a single array
            stack = np.stack(results, axis=0)

    return stack


def _init_zarr(path):
    """Detect and open an OME-Zarr store in the project directory."""
    global _zarr_store, _zarr_raw, _zarr_axes, _zarr_channel_names, _zarr_n_imaging_channels
    zarr_path = os.path.join(path, "acquisition.ome.zarr")
    if not os.path.isdir(zarr_path):
        _zarr_store = _zarr_raw = _zarr_axes = _zarr_channel_names = None
        _zarr_n_imaging_channels = None
        return False

    store = zarr.open(zarr_path, mode="r")
    _zarr_store = store
    _zarr_raw = store["0"]

    ome = dict(store.attrs).get("ome", {})
    multiscales = ome.get("multiscales", [{}])
    axes = multiscales[0].get("axes", [])
    _zarr_axes = [a["name"] for a in axes]

    channels = ome.get("omero", {}).get("channels", [])
    _zarr_channel_names = [ch.get("label", f"c{i}") for i, ch in enumerate(channels)]

    # Channels named "stim_*" are stim channels; the rest are imaging channels
    _zarr_n_imaging_channels = sum(
        1 for name in _zarr_channel_names if not name.startswith("stim_")
    )
    print(
        f"OME-Zarr detected: axes={_zarr_axes}, "
        f"channels={_zarr_channel_names} "
        f"({_zarr_n_imaging_channels} imaging + "
        f"{len(_zarr_channel_names) - _zarr_n_imaging_channels} stim)"
    )
    return True


def _zarr_has_position_axis():
    return _zarr_axes is not None and "p" in _zarr_axes


def _zarr_load_raw(fov_idx, stim=False):
    """Load raw or stim channels for a given FOV from the zarr store as dask array.

    Returns array with shape (t, c, y, x).
    """
    has_p = _zarr_has_position_axis()
    has_c = "c" in _zarr_axes

    raw_da = da.from_zarr(_zarr_raw)

    if has_p and has_c:
        # (t, p, c, y, x) -> select position
        data = raw_da[:, fov_idx, :, :, :]  # (t, c, y, x)
    elif has_p:
        # (t, p, y, x)
        data = raw_da[:, fov_idx, :, :]  # (t, y, x)
        data = data[:, np.newaxis, :, :]  # (t, 1, y, x)
    elif has_c:
        # (t, c, y, x) - single position
        data = raw_da
    else:
        # (t, y, x)
        data = raw_da[:, np.newaxis, :, :]

    # Split imaging vs stim channels
    if has_c and _zarr_n_imaging_channels is not None:
        if stim:
            data = data[:, _zarr_n_imaging_channels:, :, :]
        else:
            data = data[:, :_zarr_n_imaging_channels, :, :]

    return data


def _zarr_load_label(label_name, fov_idx):
    """Load a label array for a given FOV from the zarr store as dask array.

    Returns array with shape (t, y, x).
    """
    lbl_path = f"labels/{label_name}/0"
    try:
        lbl_arr = _zarr_store[lbl_path]
    except KeyError:
        return None

    lbl_da = da.from_zarr(lbl_arr)
    has_p = _zarr_has_position_axis()

    if has_p:
        return lbl_da[:, fov_idx, :, :]  # (t, y, x)
    return lbl_da  # (t, y, x)


# Mapping from TIFF folder names to zarr label group names
_FOLDER_TO_ZARR_LABEL = {
    PARTICLES_FOLDER: "particles",
    MASK_FOLDER: "labels",
    LABELS_RINGS: "labels_ring",
    LIGHT_MASK_FOLDER: "stim_mask",
}


##
def update_or_add_layer(layer_name, data, colormap, blending, layer_type="image"):
    layer = viewer.layers[layer_name] if layer_name in viewer.layers else None
    if layer is None:
        if layer_type == "image":
            viewer.add_image(
                data, name=layer_name, colormap=colormap, blending=blending
            )
        elif layer_type == "labels":
            viewer.add_labels(data, name=layer_name, blending=blending)
    else:
        if layer in viewer.layers:
            layer.data = data
        elif layer_type == "image":
            viewer.add_image(
                data, name=layer_name, colormap=colormap, blending=blending
            )
        elif layer_type == "labels":
            viewer.add_labels(data, name=layer_name, blending=blending)


@magicgui(
    cell_line={"widget_type": ComboBox, "choices": [], "label": "Cell Line"},
    filter_cell_line={"widget_type": "CheckBox", "label": "use filter", "value": True},
    exposure_time={"widget_type": ComboBox, "choices": [], "label": "StimExposure"},
    filter_exposure_time={
        "widget_type": "CheckBox",
        "label": "use filter",
        "value": True,
    },
    stim_timestep={"widget_type": ComboBox, "choices": [], "label": "StimTimepoint"},
    filter_stim_timestep={
        "widget_type": "CheckBox",
        "label": "use filter",
        "value": True,
    },
    fov={"widget_type": ComboBox, "choices": [], "label": "FOV"},
    filter_fov={"widget_type": "CheckBox", "label": "use filter", "value": True},
    call_button="Load FOV",
    next_fov={"widget_type": "PushButton", "label": "Next FOV ->"},
    previous_fov={"widget_type": "PushButton", "label": "Previous FOV <-"},
    select_data={
        "choices": [folder.folder_name for folder in FOLDERS_TO_LOAD],
        "allow_multiple": True,
        "label": "Layers",
    },
    lazy={"widget_type": "CheckBox", "label": "Lazy Loading"},
)
def selection_widget(
    cell_line: str,
    filter_cell_line: bool,
    exposure_time: int,
    filter_exposure_time: bool,
    stim_timestep,
    filter_stim_timestep: bool,
    fov: str,
    filter_fov: bool,
    select_data: list[str] = [folder.folder_name for folder in FOLDERS_TO_LOAD],
    lazy: bool = True,
    next_fov: bool = False,
    previous_fov: bool = False,
):
    print(
        f"Selected: cell_line={cell_line}({filter_cell_line}), exposure_time={exposure_time}({filter_exposure_time}), stim_timestep={stim_timestep}({filter_stim_timestep}), fov={fov}({filter_fov})"
    )
    global currently_added_layers

    # Dynamisch Filter-Query bauen
    filter_parts = []
    if filter_cell_line:
        filter_parts.append("cell_line == @cell_line")
    if filter_exposure_time:
        filter_parts.append("stim_exposure == @exposure_time")
    if filter_stim_timestep:
        filter_parts.append("stim_timestep == @stim_timestep")
    if filter_fov:
        filter_parts.append("fov == @fov")
    if filter_parts:
        query_str = " and ".join(filter_parts)
        # Filter out NA values in stim_exposure before applying the query
        filtered_df = exp_df.dropna(subset=["stim_exposure"]).query(query_str)
    else:
        # Filter out NA values in stim_exposure
        filtered_df = exp_df.dropna(subset=["stim_exposure"])

    filenames = filtered_df["fname"].unique().tolist()

    layers_added_in_current_call = []

    def update_or_add_layer(
        layer_name,
        data,
        colormap,
        blending,
        layer_type="image",
    ):
        layer = viewer.layers[layer_name] if layer_name in viewer.layers else None
        if layer is None:
            if layer_type == "image":
                viewer.add_image(
                    data, name=layer_name, colormap=colormap, blending=blending
                )
            elif layer_type == "labels":
                viewer.add_labels(data, name=layer_name, blending=blending)
        else:
            if layer in viewer.layers:
                layer.data = data
            elif layer_type == "image":
                viewer.add_image(
                    data, name=layer_name, colormap=colormap, blending=blending
                )
            elif layer_type == "labels":
                viewer.add_labels(data, name=layer_name, blending=blending)

    for folder in select_data:
        folder_info = next(
            (f for f in FOLDERS_TO_LOAD if f.folder_name == folder), None
        )
        if folder_info is None:
            print(f"Folder {folder} not found")
            continue

        # --- load data from zarr or TIFF ---
        data = None
        if _zarr_store is not None:
            fov_idx = int(fov) if fov is not None else 0
            if folder == RAW_FOLDER:
                data = _zarr_load_raw(fov_idx, stim=False)
            elif folder == STIM_FOLDER:
                data = _zarr_load_raw(fov_idx, stim=True)
                if data is not None and data.shape[1] == 0:
                    # no stim channels stored in raw; try stim_mask label
                    data = _zarr_load_label("stim_mask", fov_idx)
            else:
                zarr_label = _FOLDER_TO_ZARR_LABEL.get(folder)
                if zarr_label is not None:
                    data = _zarr_load_label(zarr_label, fov_idx)
        else:
            data = tiff_to_da(folder_info.folder_name, filenames=filenames, lazy=lazy)

        if data is None:
            print(f"No data found for {folder} in {fov}")
            continue
        if folder_info.layer_type == "labels":
            data = data.astype(np.uint32)
        if data.ndim == 4:  # for multi-channel images
            for i in range(data.shape[1]):
                layer_name = f"{folder}_c{i}"
                update_or_add_layer(
                    layer_name,
                    data[:, i, :, :],
                    folder_info.colormap,
                    folder_info.blending,
                    folder_info.layer_type,
                )
                layers_added_in_current_call.append(layer_name)
        else:
            update_or_add_layer(
                folder,
                data,
                folder_info.colormap,
                folder_info.blending,
                folder_info.layer_type,
            )
            layers_added_in_current_call.append(folder)

    # remove layers that were added in the previous call but not in the current call
    for layer in currently_added_layers:
        if layer not in layers_added_in_current_call:
            try:
                viewer.layers.remove(layer)
            except ValueError:
                pass

    currently_added_layers = [layer for layer in layers_added_in_current_call]

    global current_fov
    global current_cell_line
    global current_exposure_time
    global current_stim_timestep

    if cell_line is None:
        cell_line = current_cell_line
    if exposure_time is None:
        exposure_time = current_exposure_time
    if stim_timestep is None:
        stim_timestep = current_stim_timestep
    if fov is None:
        fov = current_fov

    selection_widget.cell_line.choices = get_cell_lines()
    selection_widget.exposure_time.choices = get_exposure_times(cell_line)
    selection_widget.stim_timestep.choices = get_stim_timesteps(
        cell_line, exposure_time
    )
    selection_widget.fov.choices = get_fov_choices(
        cell_line, exposure_time, stim_timestep
    )

    selection_widget.cell_line.value = cell_line
    selection_widget.exposure_time.value = exposure_time
    selection_widget.stim_timestep.value = stim_timestep
    selection_widget.fov.value = fov

    current_fov = fov
    current_cell_line = cell_line
    current_exposure_time = exposure_time
    current_stim_timestep = stim_timestep

    return selection_widget


def set_next(value):
    global current_fov
    # Filter out NA values in stim_exposure
    filtered_exp_df = exp_df.dropna(subset=["stim_exposure"])
    fov_choices = get_fov_choices(
        current_cell_line, current_exposure_time, current_stim_timestep
    )
    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index < len(fov_choices) - 1:
            selection_widget.fov.value = fov_choices[current_index + 1]
    elif current_fov is None:
        selection_widget.fov.value = fov_choices[0]
    # Unset highlighting when changing FOV
    if hasattr(cell_time_series_widget, "selected_particle"):
        cell_time_series_widget.selected_particle = None
        cell_time_series_widget._show_all = True
        cell_time_series_widget.update_plot()
    selection_widget.call_button.clicked.emit()


def set_previous(value):
    global current_fov
    # Filter out NA values in stim_exposure
    filtered_exp_df = exp_df.dropna(subset=["stim_exposure"])
    fov_choices = get_fov_choices(
        current_cell_line, current_exposure_time, current_stim_timestep
    )
    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index > 0:
            selection_widget.fov.value = fov_choices[current_index - 1]
    elif current_fov is None:
        selection_widget.fov.value = fov_choices[0]
    # Unset highlighting when changing FOV
    if hasattr(cell_time_series_widget, "selected_particle"):
        cell_time_series_widget.selected_particle = None
        cell_time_series_widget._show_all = True
        cell_time_series_widget.update_plot()
    selection_widget.call_button.clicked.emit()


def update_exposure_times(event=None):
    exposure_times = get_exposure_times(selection_widget.cell_line.value)
    prev_choices = set(selection_widget.exposure_time.choices)

    if prev_choices != set(exposure_times):
        selection_widget.exposure_time.choices = exposure_times

    if selection_widget.exposure_time.value in exposure_times:
        selection_widget.exposure_time.value = selection_widget.exposure_time.value
    else:
        if exposure_times:
            selection_widget.exposure_time.value = exposure_times[0]
        else:
            # Wenn keine exposure_times vorhanden, setze auf None aber nur wenn erlaubt
            try:
                selection_widget.exposure_time.value = None
            except ValueError:
                # Wenn None nicht erlaubt ist, lasse den aktuellen Wert
                pass

    update_stim_timesteps()


def update_stim_timesteps(event=None):
    stim_timepoints = get_stim_timesteps(
        selection_widget.cell_line.value, selection_widget.exposure_time.value
    )
    selection_widget.stim_timestep.choices = stim_timepoints

    values = [v for _, v in stim_timepoints]
    if selection_widget.stim_timestep.value in values:
        selection_widget.stim_timestep.value = selection_widget.stim_timestep.value
    else:
        if values:
            selection_widget.stim_timestep.value = values[0]
        else:
            # Wenn keine stim_timesteps vorhanden, setze auf None aber nur wenn erlaubt
            try:
                selection_widget.stim_timestep.value = None
            except ValueError:
                # Wenn None nicht erlaubt ist, lasse den aktuellen Wert
                pass

    update_fov()


def update_fov(event=None):
    fov_choices = get_fov_choices(
        selection_widget.cell_line.value,
        selection_widget.exposure_time.value,
        selection_widget.stim_timestep.value,
    )
    prev_choices = set(selection_widget.fov.choices)

    if prev_choices != set(fov_choices):
        selection_widget.fov.choices = fov_choices

    if selection_widget.fov.value in fov_choices:
        selection_widget.fov.value = selection_widget.fov.value
    else:
        if fov_choices:
            selection_widget.fov.value = fov_choices[0]
        else:
            # Wenn keine fovs vorhanden, setze auf None aber nur wenn erlaubt
            try:
                selection_widget.fov.value = None
            except ValueError:
                # Wenn None nicht erlaubt ist, lasse den aktuellen Wert
                pass


# widget to choose the directory
@magicgui(
    directory={"mode": "d", "label": "Experiment: "},
    auto_call=True,
)
def directorypicker(
    directory=Path(DEFAULT_FOLDER),
):
    """Take a directory name and do something with it."""
    print("The directory name is:", directory)
    global project_path
    global exp_df
    global currently_added_layers
    project_path = directory.as_posix().strip()
    exp_df = None
    for layer in viewer.layers:
        try:
            viewer.layers.remove(layer)
        except ValueError:
            pass
    currently_added_layers = []

    _init_zarr(project_path)

    cell_lines = get_cell_lines()
    if cell_lines:
        selection_widget.cell_line.choices = cell_lines
        selection_widget.cell_line.value = cell_lines[0]
        update_exposure_times()
    return directory


def label_to_value(tracks, labels_stack, what, no_normalization=False):
    particles_stack = np.zeros_like(labels_stack, dtype=np.uint16)
    tracks_df_norm = tracks[[timestep_variable_name, "particle", what]].copy()
    tracks_df_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
    tracks_df_norm.dropna(inplace=True)
    particles_stack = np.zeros_like(labels_stack, dtype=np.float64)
    for frame in range(labels_stack.shape[0]):
        labels_f = np.array(labels_stack[frame, :, :])

        tracks_f = tracks_df_norm[tracks_df_norm[timestep_variable_name] == frame]
        from_label = tracks_f["particle"].values
        to_particle = tracks_f[what].to_numpy()
        skimage.util.map_array(
            labels_f, from_label, to_particle, out=particles_stack[frame, :, :]
        )
    return particles_stack


# widget to get ERK-KTR CNr overlay
@magicgui(
    next_fov={"widget_type": "PushButton", "label": "Add CNr layer"},
    auto_call=True,
)
def add_cnr_overlay(next_fov: bool = False):
    # Filter out NA values in stim_exposure
    filtered_exp_df = exp_df.dropna(subset=["stim_exposure"])
    exp_df_current_fov = filtered_exp_df.query(
        "cell_line == @current_cell_line and stim_exposure == @current_exposure_time and fov == @current_fov"
    ).copy()
    if not exp_df_current_fov.columns.str.contains("CNr").any():
        exp_df_current_fov["CNr"] = (
            exp_df_current_fov["mean_intensity_C1_ring"]
            / exp_df_current_fov["mean_intensity_C1_nuc"]
        )
    particles_stack = viewer.layers["particles"].data
    if isinstance(particles_stack, da.Array):
        particles_stack = particles_stack.compute()
    labels_cnr_overlay = label_to_value(
        exp_df_current_fov, particles_stack, "CNr", True
    )
    viewer.add_image(labels_cnr_overlay, name="CNr", colormap="viridis")

    selection_widget.cell_line.choices = get_cell_lines()
    selection_widget.exposure_time.choices = get_exposure_times(current_cell_line)
    selection_widget.stim_timestep.choices = get_stim_timesteps(
        current_cell_line, current_exposure_time
    )
    selection_widget.fov.choices = get_fov_choices(
        current_cell_line, current_exposure_time, current_stim_timestep
    )

    selection_widget.cell_line.value = current_cell_line
    selection_widget.exposure_time.value = current_exposure_time
    selection_widget.stim_timestep.value = current_stim_timestep
    selection_widget.fov.value = current_fov


# widget to get ERK-KTR CNr overlay
@magicgui(
    next_fov={"widget_type": "PushButton", "label": "Add optocheck layer"},
    auto_call=True,
)
def add_optocheck_overlay(next_fov: bool = False):
    # Filter out NA values in stim_exposure
    filtered_exp_df = exp_df.dropna(subset=["stim_exposure"])
    exp_df_current_fov = filtered_exp_df.query(
        "cell_line == @current_cell_line and stim_exposure == @current_exposure_time and fov == @current_fov"
    ).copy()

    particles_stack = viewer.layers["particles"].data
    if isinstance(particles_stack, da.Array):
        particles_stack = particles_stack.compute()
    labels_cnr_overlay = label_to_value(
        exp_df_current_fov, particles_stack, "optocheck_mean_intensity", True
    )
    viewer.add_image(labels_cnr_overlay, name="optocheck", colormap="viridis")

    selection_widget.cell_line.choices = get_cell_lines()
    selection_widget.exposure_time.choices = get_exposure_times(current_cell_line)
    selection_widget.stim_timestep.choices = get_stim_timesteps(
        current_cell_line, current_exposure_time
    )
    selection_widget.fov.choices = get_fov_choices(
        current_cell_line, current_exposure_time, current_stim_timestep
    )

    selection_widget.cell_line.value = current_cell_line
    selection_widget.exposure_time.value = current_exposure_time
    selection_widget.stim_timestep.value = current_stim_timestep
    selection_widget.fov.value = current_fov


current_fov = None
current_cell_line = None
current_exposure_time = None
current_stim_timestep = None
project_path = None

# check if viewer is already open, if not create a new viewer
if napari.current_viewer() is None:
    viewer = napari.Viewer()
else:
    viewer = napari.current_viewer()

# create widgets and add them to napari
dock_widget = viewer.window.add_dock_widget(directorypicker, name="Choose a directory")
viewer.window.add_dock_widget(selection_widget, name="Load FOV")
viewer.window.add_dock_widget(add_cnr_overlay, name="Add CNr overlay")
viewer.window.add_dock_widget(add_optocheck_overlay, name="Add optocheck overlay")

selection_widget.cell_line.changed.connect(update_exposure_times)
selection_widget.exposure_time.changed.connect(update_stim_timesteps)
selection_widget.stim_timestep.changed.connect(update_fov)
selection_widget.next_fov.changed.connect(set_next)
selection_widget.previous_fov.changed.connect(set_previous)


from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QLabel,
    QComboBox,
    QPushButton,
    QToolBar,
    QAction,
    QListWidget,
    QListWidgetItem,
)
from qtpy.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from qtpy.QtCore import QTimer
import csv


class VerticalNavigationToolbar(QToolBar):
    """Custom vertical navigation toolbar for matplotlib canvas"""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.setOrientation(Qt.Vertical)
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)

        # Get the standard navigation toolbar to extract actions
        self._nav_toolbar = NavigationToolbar(canvas, self)
        self._nav_toolbar.hide()  # Hide the original toolbar

        # Create vertical actions based on the standard toolbar
        self._create_actions()

        # Track navigation state
        self.mode = ""

    def _create_actions(self):
        """Create vertical toolbar actions"""
        # Home action
        home_action = QAction("🏠", self)
        home_action.setToolTip("Reset original view")
        home_action.triggered.connect(self._custom_home)
        self.addAction(home_action)

        self.addSeparator()

        # Pan action
        pan_action = QAction("✋", self)
        pan_action.setToolTip("Pan axes with left mouse, zoom with right")
        pan_action.setCheckable(True)
        pan_action.triggered.connect(lambda checked: self._toggle_mode("pan", checked))
        self.addAction(pan_action)
        self._pan_action = pan_action

        # Zoom action
        zoom_action = QAction("🔍", self)
        zoom_action.setToolTip("Zoom to rectangle")
        zoom_action.setCheckable(True)
        zoom_action.triggered.connect(
            lambda checked: self._toggle_mode("zoom", checked)
        )
        self.addAction(zoom_action)
        self._zoom_action = zoom_action

        self.addSeparator()

        # Back action
        back_action = QAction("⬅", self)
        back_action.setToolTip("Back to previous view")
        back_action.triggered.connect(self._nav_toolbar.back)
        self.addAction(back_action)

        # Forward action
        forward_action = QAction("➡", self)
        forward_action.setToolTip("Forward to next view")
        forward_action.triggered.connect(self._nav_toolbar.forward)
        self.addAction(forward_action)

    def _toggle_mode(self, mode, checked):
        """Toggle between pan/zoom modes"""
        if checked:
            # Uncheck other modes
            if mode == "pan":
                self._zoom_action.setChecked(False)
                self._nav_toolbar.pan()
                self.mode = "pan"
            elif mode == "zoom":
                self._pan_action.setChecked(False)
                self._nav_toolbar.zoom()
                self.mode = "zoom"
        else:
            # Deactivate current mode
            if mode == "pan":
                self._nav_toolbar.pan()
            elif mode == "zoom":
                self._nav_toolbar.zoom()
            self.mode = ""

    def _custom_home(self):
        """Custom home action that triggers a proper plot reset"""
        # Find the parent widget that contains the plot
        parent = self.parent()
        while parent and not hasattr(parent, "update_plot"):
            parent = parent.parent()

        # If we found the widget with update_plot method, call it
        if parent and hasattr(parent, "update_plot"):
            parent.update_plot()
        else:
            # Fallback to default home behavior
            self._nav_toolbar.home()


class CellTimeSeriesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Hauptlayout: horizontal (links Controls, mitte Plot, rechts Cell List)
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Linke Spalte für Controls (ca. 1/6 der Breite)
        control_layout = QVBoxLayout()
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setFixedWidth(150)  # ca. 1/6 der typischen Plotbreite

        self.label = QLabel("y-axis:")
        self.combo = QComboBox()
        self.combo.addItems(["cnr", "cnr_median", "cnr_norm", "cnr_norm_median"])
        default_index = self.combo.findText("cnr_norm")
        if default_index != -1:
            self.combo.setCurrentIndex(default_index)
        self.highlight_btn = QPushButton("Highlight Cell")
        self.delete_btn = QPushButton("Remove Cell")
        control_layout.addWidget(self.label)
        control_layout.addWidget(self.combo)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.highlight_btn)
        control_layout.addWidget(self.delete_btn)
        control_layout.addStretch()

        # Plot in der Mitte mit vertikaler Toolbar links
        plot_layout = QHBoxLayout()
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)

        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Vertical navigation toolbar
        self.toolbar = VerticalNavigationToolbar(self.canvas, self)
        self.toolbar.setFixedWidth(40)  # Schmale Toolbar

        # Plot layout: toolbar links, canvas rechts
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        # Rechte Spalte für Cell List
        cell_list_layout = QVBoxLayout()
        cell_list_widget = QWidget()
        cell_list_widget.setLayout(cell_list_layout)
        cell_list_widget.setFixedWidth(200)  # Feste Breite für die Liste

        # Cell list title
        cell_list_title = QLabel("Marked Cells:")
        cell_list_title.setStyleSheet("font-weight: bold;")

        # Cell list widget
        self.cell_list = QListWidget()
        self.cell_list.setMaximumHeight(300)  # Begrenzte Höhe
        self.cell_list.itemClicked.connect(
            self.on_cell_list_selection
        )  # Add selection handler

        # Button to remove selected entry from list
        self.remove_marking_btn = QPushButton("Remove Marking")
        self.remove_marking_btn.clicked.connect(self.remove_selected_marking)

        # Refresh button to update the list
        self.refresh_list_btn = QPushButton("Refresh List")
        self.refresh_list_btn.clicked.connect(self.update_cell_list)

        cell_list_layout.addWidget(cell_list_title)
        cell_list_layout.addWidget(self.cell_list)
        cell_list_layout.addWidget(self.remove_marking_btn)
        cell_list_layout.addWidget(self.refresh_list_btn)
        cell_list_layout.addStretch()

        # Layouts zusammenfügen: Controls links, Plot mitte, Cell List rechts
        main_layout.addWidget(control_widget)
        main_layout.addWidget(plot_widget, stretch=5)
        main_layout.addWidget(cell_list_widget)

        self.combo.currentTextChanged.connect(self.update_plot)
        self.selected_particle = None
        self.lines = []
        self.cid = self.canvas.mpl_connect("pick_event", self.on_pick)
        self.labels_layer = None  # wird später gesetzt

        self.highlight_btn.clicked.connect(self.highlight_cell)
        self.delete_btn.clicked.connect(self.delete_cell)

        # Rechtsklick-Event für "alle Zellen wieder anzeigen"
        self.canvas.mpl_connect("button_press_event", self.on_right_click)

        self._show_all = True  # Flag: alle Zellen anzeigen

        # Dictionary to store cell markings (highlighted/deleted)
        self.cell_markings = {}

    def update_cell_list(self):
        """Update the cell list widget with current markings"""
        self.cell_list.clear()

        if not hasattr(self, "fov") or self.fov is None:
            return

        # Sort markings by particle number for consistent display
        sorted_markings = sorted(self.cell_markings.items())

        for particle, marking in sorted_markings:
            if marking == "highlighted":
                item_text = f"ID {particle} (Highlighted)"
                item = QListWidgetItem(item_text)
                item.setData(
                    1, {"particle": particle, "marking": marking}
                )  # Store data
                item.setBackground(
                    item.background().color().lighter(140)
                )  # Light background
                self.cell_list.addItem(item)
            elif marking == "deleted":
                item_text = f"ID {particle} (Removed)"
                item = QListWidgetItem(item_text)
                item.setData(
                    1, {"particle": particle, "marking": marking}
                )  # Store data
                item.setBackground(
                    item.background().color().darker(120)
                )  # Dark background
                self.cell_list.addItem(item)

    def on_cell_list_selection(self, item):
        """Handle cell list selection - highlight corresponding cell in plot and napari"""
        if item is None:
            return

        item_data = item.data(1)
        if item_data is None:
            return

        particle = item_data["particle"]

        # Highlight in plot
        self.selected_particle = particle
        self._show_all = True  # Keep all cells visible
        self.highlight_particle(particle)

        # Highlight in napari viewer
        if self.labels_layer is not None:

            def set_napari_label():
                try:
                    if hasattr(self.labels_layer, "selected_label"):
                        self.labels_layer.selected_label = particle
                except Exception as e:
                    pass  # Ignore errors if layer is not set up correctly

            QTimer.singleShot(50, set_napari_label)

    def highlight_cell_in_list(self, particle):
        """Highlight a specific cell in the list if it exists"""
        if particle is None:
            return

        # Find the item in the list that corresponds to this particle
        for i in range(self.cell_list.count()):
            item = self.cell_list.item(i)
            item_data = item.data(1)
            if item_data and item_data["particle"] == particle:
                # Select this item in the list
                self.cell_list.setCurrentItem(item)
                # Scroll to make sure it's visible
                self.cell_list.scrollToItem(item)
                break

    def remove_selected_marking(self):
        """Remove the selected marking from the list and CSV file"""
        current_item = self.cell_list.currentItem()
        if current_item is None:
            return

        # Get particle data from the item
        item_data = current_item.data(1)
        if item_data is None:
            return

        particle = item_data["particle"]

        # Remove from CSV file
        self.remove_marking_from_csv(particle)

        # Reload markings and update display
        self.load_cell_markings()
        self.update_cell_list()
        self.update_plot()

    def remove_marking_from_csv(self, particle):
        """Remove a specific particle marking from the CSV file"""
        if self.fov is None:
            return

        uid = f"{self.fov}_{particle}"
        csv_path = os.path.join(project_path, "cell_selection.csv")

        if not os.path.exists(csv_path):
            return

        # Read existing data
        rows = []
        try:
            with open(csv_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Keep all rows except the one we want to remove
                    if str(row.get("uid", "")) != uid:
                        rows.append(row)
                    else:
                        # Found the row to remove - don't add it to the new list
                        print(
                            f"Removing marking for particle {particle} in FOV {self.fov}"
                        )
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return

        # Write back the data without the removed entry
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                if rows:  # Only write if there are rows left
                    fieldnames = [
                        "particle",
                        "label",
                        "fov",
                        "uid",
                        "selected",
                        "deleted",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                else:
                    # If no rows left, create empty file with headers
                    fieldnames = [
                        "particle",
                        "label",
                        "fov",
                        "uid",
                        "selected",
                        "deleted",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
        except Exception as e:
            print(f"Error writing CSV file: {e}")

    def load_cell_markings(self):
        """Load cell markings from CSV file"""
        if self.fov is None:
            return

        csv_path = os.path.join(project_path, "cell_selection.csv")
        self.cell_markings = {}

        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if str(row.get("fov", "")) == str(self.fov):
                            particle = int(row.get("particle", 0))
                            selected = row.get("selected", "False").lower() == "true"
                            deleted = row.get("deleted", "False").lower() == "true"

                            if selected:
                                self.cell_markings[particle] = "highlighted"
                            elif deleted:
                                self.cell_markings[particle] = "deleted"
            except Exception as e:
                print(f"Error loading cell markings: {e}")

    def get_line_style_for_particle(self, particle):
        """Get line style based on particle marking"""
        marking = self.cell_markings.get(particle, "normal")

        if marking == "deleted":
            return {"linestyle": "--", "alpha": 0.2, "linewidth": 1.0}
        elif marking == "highlighted":
            return {"linestyle": "-.", "alpha": 0.8, "linewidth": 2.0}
        else:
            return {"linestyle": "-", "alpha": 0.3, "linewidth": 1.0}

    def set_labels_layer(self, labels_layer):
        self.labels_layer = labels_layer

    def update_data(self, exp_df, fov, y_col="cnr"):
        self.exp_df = exp_df
        self.fov = fov
        self.y_col = y_col
        self.ensure_norm_columns(force_default=True)
        self._show_all = True
        self.load_cell_markings()  # Load markings when data is updated
        self.update_cell_list()  # Update the cell list display
        self.update_plot()

    def ensure_norm_columns(self, force_default=False):
        pass

    def update_norm(self):
        self.ensure_norm_columns(force_default=False)
        self.update_plot()

    def update_plot(self):
        try:
            if not hasattr(self, "exp_df") or self.exp_df is None or self.fov is None:
                return

            y_col = self.combo.currentText()
            self.ax.clear()
            self.lines = []
            df = self.exp_df[self.exp_df["fov"] == self.fov]

            print(timestep_variable_name)
            if timestep_variable_name is not None:
                x = timestep_variable_name
            else:
                self.canvas.draw_idle()
                return

            # Sichere Plot-Erstellung mit Fehlerbehandlung pro Linie
            plotted_count = 0
            max_lines = 500  # Limitiere die Anzahl der Linien für Performance

            for i, (particle, group) in enumerate(df.groupby("particle")):
                try:
                    # Performance-Limit
                    if plotted_count >= max_lines:
                        print(f"Limiting display to {max_lines} lines for performance")
                        break

                    # Filter basierend auf _show_all Flag
                    if (
                        not self._show_all
                        and self.selected_particle is not None
                        and particle != self.selected_particle
                    ):
                        continue

                    # Sichere Datenextraktion
                    if group.empty:
                        continue

                    x_data = group[x]
                    if y_col in group.columns:
                        y_data = group[y_col]
                    elif "cnr" in group.columns:
                        y_data = group["cnr"]
                    else:
                        # Fallback: verwende die erste numerische Spalte
                        numeric_cols = group.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            y_data = group[numeric_cols[0]]
                        else:
                            continue

                    # Bestimme Stil basierend auf Auswahl und Markierung
                    line_style = self.get_line_style_for_particle(particle)

                    if particle == self.selected_particle:
                        # Selected particle gets extra emphasis
                        alpha = 1.0
                        linewidth = 3.0
                        zorder = 10
                    else:
                        # Use marking-based style
                        alpha = line_style["alpha"]
                        linewidth = line_style["linewidth"]
                        zorder = 1

                    (line,) = self.ax.plot(
                        x_data,
                        y_data,
                        label=str(particle),
                        picker=True,
                        pickradius=5,
                        alpha=alpha,
                        linewidth=linewidth,
                        linestyle=line_style["linestyle"],
                        zorder=zorder,
                        rasterized=True,  # Bessere Performance
                    )
                    self.lines.append((line, particle))
                    plotted_count += 1

                except Exception as e:
                    print(f"Error plotting particle {particle}: {e}")
                    continue

            # Blaue Ticks für alle Werte in stim_timestep (nur Tick, keine Linie)
            # Fixme: Not working for multi phase experiments
            try:
                if "stim_timestep" in df.columns and not df.empty:
                    stim_steps = set()
                    for s in df["stim_timestep"].dropna():
                        if isinstance(s, (tuple, list)):
                            for val in s:
                                stim_steps.add(val)
                        else:
                            stim_steps.add(s)
                    stim_ticks = df[df[x].isin(stim_steps)][x].unique()
                    y_min, y_max = self.ax.get_ylim()
                    for tick in stim_ticks:
                        self.ax.plot(
                            tick,
                            y_min,
                            marker="|",
                            color="blue",
                            markersize=12,
                            alpha=0.5,
                            zorder=10,
                        )
            except Exception as e:
                print(f"Error adding stimulus ticks: {e}")

            self.ax.set_xlabel("time [min]")
            self.ax.set_ylabel(y_col)

            # Sichere Canvas-Update
            try:
                self.canvas.draw_idle()
            except Exception as e:
                print(f"Error drawing canvas: {e}")

        except Exception as e:
            print(f"Error updating plot: {e}")
            # Minimaler Fallback
            try:
                self.ax.clear()
                self.lines = []
                self.canvas.draw_idle()
            except Exception as e2:
                print(f"Error in plot fallback: {e2}")

    def on_pick(self, event):
        try:
            # Check if navigation toolbar is in zoom/pan mode
            if hasattr(self, "toolbar") and self.toolbar.mode != "":
                # Don't process pick events when zooming/panning
                return

            # Robustes Picking für überlappende Linien
            found_particle = None

            # Sicherheitscheck: lines müssen existieren
            if not hasattr(self, "lines") or not self.lines:
                return

            # Bei überlappenden Linien kann event.artist eine Liste sein
            picked_artists = (
                event.artist if isinstance(event.artist, list) else [event.artist]
            )

            # Finde die erste passende Linie
            for artist in picked_artists:
                for line, particle in self.lines:
                    if artist == line:
                        found_particle = particle
                        break
                if found_particle is not None:
                    break

            if found_particle is not None:
                # Unterscheide zwischen Single- und Double-Click
                if hasattr(event, "dblclick") and event.dblclick:
                    # Doppelklick: Einzelansicht
                    self.selected_particle = found_particle
                    self._show_all = False
                    self.update_plot()
                else:
                    # Einfachklick: nur hervorheben, alle sichtbar lassen
                    self.selected_particle = found_particle
                    self._show_all = True  # Alle Linien bleiben sichtbar
                    self.highlight_particle(found_particle)

                    # Highlight in cell list if the particle is marked
                    if found_particle in self.cell_markings:
                        self.highlight_cell_in_list(found_particle)

        except Exception as e:
            print(f"Error in pick event: {e}")
            # Sicherer Fallback ohne komplettes Neuzeichnen
            try:
                if hasattr(self, "selected_particle"):
                    self.selected_particle = None
                    self._show_all = True
            except Exception:
                pass

    def on_right_click(self, event):
        # Rechtsklick: alle Zellen wieder anzeigen
        if event.button == 3:  # 3 = Rechtsklick
            try:
                # Reset zu Vollansicht
                self._show_all = True
                old_particle = self.selected_particle
                self.selected_particle = None

                # Nur neu zeichnen wenn sich etwas geändert hat
                if not self._show_all or old_particle is not None:
                    self.update_plot()

                # Sicher prüfen ob labels_layer existiert
                if self.labels_layer is not None:

                    def unset_label():
                        try:
                            if hasattr(self.labels_layer, "show_selected_label"):
                                self.labels_layer.show_selected_label = False
                            if hasattr(self.labels_layer, "selected_label"):
                                self.labels_layer.selected_label = 0
                        except Exception:
                            pass

                    QTimer.singleShot(50, unset_label)
            except Exception as e:
                print(f"Error in right-click: {e}")
                # Minimaler Fallback
                try:
                    self._show_all = True
                    self.selected_particle = None
                except Exception:
                    pass

    def highlight_particle(self, particle):
        try:
            # Sicher prüfen ob lines existieren
            if not hasattr(self, "lines") or not self.lines:
                return

            if particle is None:
                return

            # Batch-Update aller Linien-Eigenschaften
            updates = []
            any_changes = False

            for line, p in self.lines:
                try:
                    # Prüfe aktuelle Eigenschaften vor Änderung
                    current_width = line.get_linewidth()
                    current_alpha = line.get_alpha()

                    if p == particle:
                        # Hervorheben - always make selected particle prominent
                        target_width, target_alpha = 3.0, 1.0
                        target_zorder = 10
                    else:
                        # Normal - use marking-based style
                        line_style = self.get_line_style_for_particle(p)
                        target_width = line_style["linewidth"]
                        target_alpha = line_style["alpha"]
                        target_zorder = 1

                    # Nur ändern wenn nötig
                    if current_width != target_width or current_alpha != target_alpha:
                        updates.append(
                            (line, target_width, target_alpha, target_zorder)
                        )
                        any_changes = True

                except Exception:
                    # Ignoriere ungültige Linien
                    continue

            # Alle Updates auf einmal anwenden (nur wenn nötig)
            if any_changes:
                for line, linewidth, alpha, zorder in updates:
                    try:
                        line.set_linewidth(linewidth)
                        line.set_alpha(alpha)
                        line.set_zorder(zorder)
                    except Exception:
                        continue

                # Nur ein draw-Call am Ende
                self.canvas.draw_idle()

            # Napari-Synchronisation nur wenn layer existiert
            if self.labels_layer is not None:

                def set_label():
                    try:
                        if hasattr(self.labels_layer, "selected_label"):
                            self.labels_layer.selected_label = particle
                        if hasattr(self.labels_layer, "opacity"):
                            self.labels_layer.opacity = 1.0
                        if hasattr(self.labels_layer, "visible"):
                            self.labels_layer.visible = True
                        if hasattr(self.labels_layer, "show_selected_label"):
                            self.labels_layer.show_selected_label = True
                    except Exception:
                        pass

                QTimer.singleShot(50, set_label)

            self.selected_particle = particle

        except Exception as e:
            print(f"Error highlighting particle {particle}: {e}")
            # Minimaler Fallback - setze nur selected_particle
            try:
                self.selected_particle = particle
            except Exception:
                pass

    def highlight_cell(self):
        self.set_flag_for_selected_cell("selected", True)
        self.load_cell_markings()  # Reload markings
        self.update_cell_list()  # Update the cell list display
        self.update_plot()  # Update plot to show new markings

    def delete_cell(self):
        self.set_flag_for_selected_cell("deleted", True)
        self.load_cell_markings()  # Reload markings
        self.update_cell_list()  # Update the cell list display
        self.update_plot()  # Update plot to show new markings

    def set_flag_for_selected_cell(self, flag, value):
        if self.selected_particle is None or self.fov is None:
            return
        uid = f"{self.fov}_{self.selected_particle}"
        csv_path = os.path.join(project_path, "cell_selection.csv")
        entry = {
            "particle": self.selected_particle,
            "label": self.selected_particle,
            "fov": self.fov,
            "uid": uid,
            "selected": False,
            "deleted": False,
        }
        rows = []
        found = False
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if str(row.get("uid", "")) == uid:
                        row[flag] = str(value)
                        found = True
                    rows.append(row)
        if not found:
            entry[flag] = True
            rows.append(entry)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["particle", "label", "fov", "uid", "selected", "deleted"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "particle": row["particle"],
                        "label": row["label"],
                        "fov": row["fov"],
                        "uid": row.get("uid", f"{row['fov']}_{row['particle']}"),
                        "selected": str(row.get("selected", False)),
                        "deleted": str(row.get("deleted", False)),
                    }
                )


# Widget-Instanz erstellen
cell_time_series_widget = CellTimeSeriesWidget()


# Funktion, um das Widget mit aktuellen Daten zu füllen
def update_cell_time_series_widget():
    try:
        if exp_df is None or current_fov is None:
            return

        y_col = cell_time_series_widget.combo.currentText()
        # Daten filtern und NA values in stim_exposure ausschließen
        exp_df_filtered = exp_df.dropna(subset=["stim_exposure"])
        exp_df_filtered = exp_df_filtered[exp_df_filtered["fov"] == current_fov].copy()

        # UID-Spalte hinzufügen, falls nicht vorhanden
        if "uid" not in exp_df_filtered.columns:
            exp_df_filtered["uid"] = (
                exp_df_filtered["fov"].astype("string")
                + "_"
                + exp_df_filtered["particle"].astype("string")
            )

        cell_time_series_widget.update_data(exp_df_filtered, current_fov, y_col)

        # Labels-Layer suchen
        labels_layer = None
        for layer in viewer.layers:
            if layer.name == "particles":
                labels_layer = layer
                break
        cell_time_series_widget.set_labels_layer(labels_layer)

    except Exception as e:
        print(f"Error updating cell time series widget: {e}")


# Callback, wenn FOV oder Layer gewechselt wird
def on_fov_change(event=None):
    update_cell_time_series_widget()


# Callback, wenn in Napari ein Label ausgewählt wird
def on_label_selected(event):
    try:
        particle = event.value
        # Typ angleichen, falls nötig
        try:
            particle = int(particle)
        except Exception:
            pass

        # Nur highlighten wenn das Widget existiert und bereit ist
        if hasattr(cell_time_series_widget, "highlight_particle"):
            cell_time_series_widget.highlight_particle(particle)

            # Also highlight in cell list if the particle is marked
            if particle in cell_time_series_widget.cell_markings:
                cell_time_series_widget.highlight_cell_in_list(particle)
    except Exception as e:
        print(f"Error in label selection: {e}")


# Events verbinden
selection_widget.fov.changed.connect(on_fov_change)
selection_widget.cell_line.changed.connect(on_fov_change)
selection_widget.exposure_time.changed.connect(on_fov_change)
selection_widget.stim_timestep.changed.connect(on_fov_change)
cell_time_series_widget.combo.currentTextChanged.connect(update_cell_time_series_widget)
selection_widget.call_button.clicked.connect(update_cell_time_series_widget)


# Labels-Layer Event verbinden (wenn Layer existiert)
def connect_label_event():
    for layer in viewer.layers:
        if layer.name == "particles":
            try:
                layer.events.selected_label.connect(on_label_selected)
            except Exception:
                pass


connect_label_event()
viewer.layers.events.inserted.connect(lambda event: connect_label_event())

# Initiales Update
update_cell_time_series_widget()


@viewer.bind_key("a", overwrite=True)
def highlight_selected_label_in_plot(event=None):
    # Suche das particles-Layer
    labels_layer = None
    for layer in viewer.layers:
        if layer.name == "particles":
            labels_layer = layer
            break
    if labels_layer is not None:
        particle = labels_layer.selected_label
        if particle is not None and particle != 0:
            try:
                particle = int(particle)
            except Exception:
                pass
            cell_time_series_widget.highlight_particle(particle)

            # Also highlight in cell list if the particle is marked
            if particle in cell_time_series_widget.cell_markings:
                cell_time_series_widget.highlight_cell_in_list(particle)


# Plot-Widget unten andocken (statt als Tab)
viewer.window.add_dock_widget(
    cell_time_series_widget, name="Cell Time Series", area="bottom"
)


# start event loop
napari.run()
