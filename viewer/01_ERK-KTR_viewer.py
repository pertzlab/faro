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
import dask.array as da
from concurrent.futures import ThreadPoolExecutor, as_completed
from dask import delayed
import numpy as np
from pathlib import Path


RAW_FOLDER = "raw"
STIM_FOLDER = "stim"
MASK_FOLDER = "mask"
LIGHT_MASK_FOLDER = "light_mask"
PARTICLES_FOLDER = "particles"
LABELS_RINGS = "labels_ring"
TRACKS_FOLDER = "tracks"

DEFAULT_FOLDER = "\\\\izbkingston.izb.unibe.ch\\imaging.data\\mic01-imaging\\Cedric\\experimental_data"


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

exp_df = None
currently_added_layers = []


def load_exp_df(project_path):
    global exp_df
    if exp_df is None and os.path.exists(
        os.path.join(project_path, "exp_data.parquet")
    ):
        exp_df = pd.read_parquet(os.path.join(project_path, "exp_data.parquet"))
        exp_df["stim_timestep"] = exp_df["stim_timestep"].apply(tuple)


def get_cell_lines() -> List[str]:
    load_exp_df(project_path)
    return exp_df["cell_line"].unique().tolist()


def get_exposure_times(cell_line: str) -> List[int]:
    exposure_times = (
        exp_df[exp_df["cell_line"] == cell_line]["stim_exposure"]
        .unique()
        .astype(int)
        .tolist()
    )
    return sorted(exposure_times)


def get_stim_timesteps(cell_line: str, stim_exposure: int) -> List[str]:
    stim_timesteps = (
        exp_df[
            (exp_df["cell_line"] == cell_line)
            & (exp_df["stim_exposure"] == stim_exposure)
        ]["stim_timestep"]
        .unique()
        .tolist()
    )
    stim_timesteps_dict_map = {
        "choices": stim_timesteps,
        "key": lambda x: ",".join([str(i) for i in x]),
    }
    return stim_timesteps_dict_map


def get_fov_choices(cell_line: str, stim_exposure: int, stim_timestep) -> List[str]:
    return (
        exp_df[
            (exp_df["cell_line"] == cell_line)
            & (exp_df["stim_exposure"] == stim_exposure)
            & (exp_df["stim_timestep"] == stim_timestep)
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
    cell_line={
        "widget_type": ComboBox,
        "choices": [],
        "label": "Cell Line",
    },
    exposure_time={"widget_type": ComboBox, "choices": [], "label": "StimExposure"},
    fov={"widget_type": ComboBox, "choices": [], "label": "FOV"},
    stim_timestep={"widget_type": ComboBox, "choices": [], "label": "StimTimepoint"},
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
    exposure_time: int,
    stim_timestep,
    fov: str,
    select_data: list[str] = [folder.folder_name for folder in FOLDERS_TO_LOAD],
    lazy: bool = True,
    next_fov: bool = False,
    previous_fov: bool = False,
):
    print(
        f"Selected Cell Line: {cell_line}, Exposure Time: {exposure_time}, Selected FOV: {fov}"
    )
    global currently_added_layers

    # Update choices for cell_line, exposure_time, and fov

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

    layers_added_in_current_call = []

    filenames = (
        exp_df.query(
            "cell_line == @cell_line and stim_exposure == @exposure_time and fov == @fov"
        )["fname"]
        .unique()
        .tolist()
    )

    for folder in select_data:
        folder_info = next(
            (f for f in FOLDERS_TO_LOAD if f.folder_name == folder), None
        )
        if folder_info is None:
            print(f"Folder {folder} not found")
            continue
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
    fov_choices = get_fov_choices(
        current_cell_line, current_exposure_time, current_stim_timestep
    )
    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index < len(fov_choices) - 1:
            selection_widget.fov.value = fov_choices[current_index + 1]
    elif current_fov is None:
        selection_widget.fov.value = fov_choices[0]
    selection_widget.call_button.clicked.emit()


def set_previous(value):
    global current_fov
    fov_choices = get_fov_choices(
        current_cell_line, current_exposure_time, current_stim_timestep
    )
    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index > 0:
            selection_widget.fov.value = fov_choices[current_index - 1]
    elif current_fov is None:
        selection_widget.fov.value = fov_choices[0]
    selection_widget.call_button.clicked.emit()


def update_exposure_times(event=None):
    exposure_times = get_exposure_times(selection_widget.cell_line.value)
    prev_choices = set(selection_widget.exposure_time.choices)

    if prev_choices != set(exposure_times):
        selection_widget.exposure_time.choices = exposure_times

    if selection_widget.exposure_time.value in exposure_times:
        selection_widget.exposure_time.value = selection_widget.exposure_time.value
    else:
        selection_widget.exposure_time.value = (
            exposure_times[0] if exposure_times else None
        )

    update_stim_timesteps()


def update_stim_timesteps(event=None):
    stim_timepoints = get_stim_timesteps(
        selection_widget.cell_line.value, selection_widget.exposure_time.value
    )
    selection_widget.stim_timestep.choices = stim_timepoints

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
        selection_widget.fov.value = fov_choices[0] if fov_choices else None


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

    cell_lines = get_cell_lines()
    if cell_lines:
        selection_widget.cell_line.choices = cell_lines
        selection_widget.cell_line.value = cell_lines[0]
        update_exposure_times()
    return directory


def label_to_value(tracks, labels_stack, what, no_normalization=False):
    particles_stack = np.zeros_like(labels_stack, dtype=np.uint16)
    tracks_df_norm = tracks[["timestep", "particle", what]].copy()
    tracks_df_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
    tracks_df_norm.dropna(inplace=True)
    particles_stack = np.zeros_like(labels_stack, dtype=np.float64)
    for frame in range(labels_stack.shape[0]):
        labels_f = np.array(labels_stack[frame, :, :])

        tracks_f = tracks_df_norm[tracks_df_norm["timestep"] == frame]
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
    exp_df_current_fov = exp_df.query(
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
    exp_df_current_fov = exp_df.query(
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


if __name__ == "__main__":
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
    dock_widget = viewer.window.add_dock_widget(
        directorypicker, name="Choose a directory"
    )
    viewer.window.add_dock_widget(selection_widget, name="Load FOV")
    viewer.window.add_dock_widget(add_cnr_overlay, name="Add CNr overlay")
    viewer.window.add_dock_widget(add_optocheck_overlay, name="Add optocheck overlay")

    selection_widget.cell_line.changed.connect(update_exposure_times)
    selection_widget.exposure_time.changed.connect(update_stim_timesteps)
    selection_widget.stim_timestep.changed.connect(update_fov)
    selection_widget.next_fov.changed.connect(set_next)
    selection_widget.previous_fov.changed.connect(set_previous)

    # start event loop
    napari.run()
