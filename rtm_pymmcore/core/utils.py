import numpy as np
import os
from skimage.util import map_array
from rtm_pymmcore.core.data_structures import Fov
import random
import pandas as pd
import dataclasses
import re
from pathlib import Path


def print_configs(mmc):
    """Print all available config groups and their configs as a rich tree."""
    from rich.tree import Tree
    from rich.console import Console

    tree = Tree("[bold]Config Groups")
    for group in mmc.getAvailableConfigGroups():
        configs = list(mmc.getAvailableConfigs(group))
        branch = tree.add(f"[bold cyan]{group}")
        for c in configs:
            branch.add(c)
    Console().print(tree)


def create_folders(path, folders):
    """Create all folders if they don't already exist.

    Keyword arguments:
    path -- location of main folder
    folders -- list of all subfolders
    """

    for folder in folders:
        dir_name = os.path.join(path, folder)
        try:
            os.makedirs(dir_name)
            print("Directory", dir_name, "created ")
        except FileExistsError:
            print("Directory", dir_name, "already exists")


def labels_to_particles(labels, tracks, metadata=None):
    """Takes in a segmentation mask with labels and replaces them with track IDs that are consistent over time."""
    # For every frame
    # labels_stack = np.array(labels_stack)
    particles = np.zeros_like(labels)
    if metadata is None:
        tracks_f = tracks[(tracks["timestep"] == tracks.timestep.max())]
    else:
        tracks_f = tracks[tracks["fname"] == metadata["fname"]]
    from_label = tracks_f["label"].values
    to_particle = tracks_f["particle"].values
    particles = map_array(labels, from_label, to_particle, out=particles)
    return particles


def fix_tuples_in_stim_exposure_list(
    stim_exposures_timesteps,
):
    """Convert any range or list in the stim_exposures_timesteps_before_pause to tuples. Deprecated"""
    for stim_exposure_timestep in stim_exposures_timesteps:
        # Normalize both the timestep and the exposure list to tuples.
        stim_exposure_timestep["stim_timestep"] = _normalize_to_tuple(
            stim_exposure_timestep.get("stim_timestep")
        )
        stim_exposure_timestep["stim_exposure_list"] = _normalize_to_tuple(
            stim_exposure_timestep.get("stim_exposure_list")
        )


def fix_tuples_stim_treatments(
    stim_treatments,
):
    """Convert any range or list in the stim_exposures_timesteps_before_pause to tuples. Deprecated"""
    for stim_treatment in stim_treatments:
        # Normalize stim_timestep and stim_exposure to tuples. If a single int
        # is supplied it becomes a single-element tuple.
        stim_treatment["stim_timestep"] = _normalize_to_tuple(
            stim_treatment.get("stim_timestep")
        )
        stim_treatment["stim_exposure"] = _normalize_to_tuple(
            stim_treatment.get("stim_exposure")
        )

        # Backwards compatibility: some callers may expect 'stim_exposure_list'
        # key (plural). If it's missing but 'stim_exposure' is present, copy it.
        if (
            "stim_exposure_list" not in stim_treatment
            and "stim_exposure" in stim_treatment
        ):
            stim_treatment["stim_exposure_list"] = stim_treatment["stim_exposure"]

        # Keep None as None; helper leaves None unchanged.


def _normalize_to_tuple(value):
    """Normalize a value to a tuple.

    - range -> tuple(range)
    - list/ndarray -> tuple(value)
    - tuple -> unchanged
    - scalar (int/float/str) -> (value,)
    - None -> None
    """
    if value is None:
        return None
    if isinstance(value, range):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    if isinstance(value, (list, np.ndarray)):
        return tuple(value)
    # Treat any other scalar-like value as a single-element tuple
    return (value,)


def add_stim_parameters_to_stim_exposures_timesteps(
    stim_exposures_timesteps,
    stim_power=10,
    stim_channel_name="CyanStim",
    stim_channel_group="TTL_ERK",
    stim_channel_device_name="Spectra",
    stim_channel_power_property_name="Cyan_Level",
):
    """Add general stimulation parameters to each stim_exposures_timesteps_before_pause dict. Deprecated"""
    for stim_exposure_timestep in stim_exposures_timesteps:
        stim_exposure_timestep["stim_power"] = stim_power
        stim_exposure_timestep["stim_channel_name"] = stim_channel_name
        stim_exposure_timestep["stim_channel_group"] = stim_channel_group
        stim_exposure_timestep["stim_channel_device_name"] = stim_channel_device_name
        stim_exposure_timestep["stim_channel_power_property_name"] = (
            stim_channel_power_property_name
        )


def make_baseline_stim_baseline_treatments(
    stim_start,
    stim_end,
    stim_exposure,
    treatment_name="baseline-stim-baseline",
):
    """Create a baseline->stim->baseline treatment list.

    The stim is applied for timesteps in [stim_start, stim_end).
    """
    stim_timestep = tuple(range(stim_start, stim_end))
    stim_exposure_list = tuple([stim_exposure] * len(stim_timestep))
    return [
        {
            "treatment_name": treatment_name,
            "stim_timestep": stim_timestep,
            "stim_exposure_list": stim_exposure_list,
        }
    ]


def print_stim_exposures_timesteps(
    stim_exposures_timesteps,
):
    """Print the stim_exposures_timesteps_before_pause in a readable format. Deprecated"""
    for stim_exposure_timestep in stim_exposures_timesteps:
        print("Pattern Name: ", stim_exposure_timestep["treatment_name"])

        for stim_exp, stim_timestep in zip(
            stim_exposure_timestep["stim_exposure_list"],
            stim_exposure_timestep["stim_timestep"],
        ):
            print(f"{stim_exp} at {stim_timestep}")
        print("")


def print_stim_exposures_timesteps(
    stim_exposures_timesteps,
):
    """Print the stim_exposures_timesteps_before_pause in a readable format."""
    for stim_exposure_timestep in stim_exposures_timesteps:
        print("Pattern Name: ", stim_exposure_timestep["treatment_name"])

        for stim_exp, stim_timestep in zip(
            stim_exposure_timestep["stim_exposure_list"],
            stim_exposure_timestep["stim_timestep"],
        ):
            print(f"{stim_exp} at {stim_timestep}")
        print("")


def print_stim_exposures_timesteps(
    stim_exposures_timesteps,
):
    """Print the stim treatment lists in a readable format."""
    for stim_exposure_timestep in stim_exposures_timesteps:
        print("Pattern Name: ", stim_exposure_timestep.treatment_name)

        for stim_exp, stim_timestep in zip(
            stim_exposure_timestep.stim_exposure_list,
            stim_exposure_timestep.stim_timestep,
        ):
            print(f"{stim_exp} at {stim_timestep}")
        print("")


def _get_mda_from_file(filename):
    import json

    file = os.path.join(filename)
    with open(file, "r") as f:
        data_mda_fovs = json.load(f)
    return data_mda_fovs


def _get_mda_from_viewer(viewer):
    data_mda_fovs = viewer.window._dock_widgets["MDA"].widget().value().stage_positions
    data_mda_fovs_dict = []
    for data_mda in data_mda_fovs:
        data_mda_fovs_dict.append(data_mda.model_dump())
    data_mda_fovs = data_mda_fovs_dict
    return data_mda_fovs

def generate_fov_objects_from_list(mic, data_mda_fovs):
    fovs = []
    for i, fov in enumerate(data_mda_fovs):
        fov_object = Fov(i)
        fov_object.x = fov.get("x")
        fov_object.y = fov.get("y")
        fov_object.z = None if getattr(mic, "ONLY_USE_PFS", False) else fov.get("z")
        fov_object.name = str(i) if fov["name"] is None else fov["name"]
        fovs.append(fov_object)
    return fovs

def generate_fov_objects(mic, viewer=None, filename=None):
    if filename is not None:
        data_mda_fovs = _get_mda_from_file(filename)
    elif viewer is not None:
        data_mda_fovs = _get_mda_from_viewer(viewer)
        if data_mda_fovs is None:
            assert False, "No fovs selected. Please select fovs in the MDA widget"
    else:
        assert False, "Either viewer must be provided or from_file must be True"

    fovs = []
    for i, fov in enumerate(data_mda_fovs):
        fov_object = Fov(i)
        fov_object.x = fov.get("x")
        fov_object.y = fov.get("y")
        fov_object.z = None if getattr(mic, "ONLY_USE_PFS", False) else fov.get("z")
        fov_object.name = str(i) if fov["name"] is None else fov["name"]
        fovs.append(fov_object)
    return fovs


def generate_df_acquire_simple(fovs, n_frames, time_between_timesteps, channels, start_time=0):
    dfs = []
    for fov_index, fov in enumerate(fovs):
        for timestep in range(n_frames):
            dfs.append({
                "fov_object": fov,
                "fov": fov_index,
                "fov_x": fov.x,
                "fov_y": fov.y,
                "fov_z": fov.z,
                "fov_name": fov.name,
                "timestep": timestep,
                "time": start_time + timestep * time_between_timesteps,
                "channels": tuple(dataclasses.asdict(ch) for ch in channels),
                "fname": f"{str(fov_index).zfill(3)}_{str(timestep).zfill(5)}",
            })
    df_acquire = pd.DataFrame(dfs).sort_values(by=["time", "fov"]).reset_index(drop=True)
    print(f"Total Experiment Time: {df_acquire['time'].max() / 3600}h")
    return df_acquire


def generate_df_acquire(
    fovs,
    n_frames,
    time_between_timesteps,
    time_per_fov,
    channels,
    start_time=0,
    channel_optocheck=None,
    optocheck_timepoints=None,
    phase_id=None,
    phase_name=None,
    condition=None,
):
    n_fovs_simultaneously = time_between_timesteps // time_per_fov
    optocheck_timepoints = (
        optocheck_timepoints if optocheck_timepoints is not None else [n_frames - 1]
    )
    timesteps = range(n_frames)
    dfs = []
    for fov_index, fov in enumerate(fovs):
        fov_group = fov_index // n_fovs_simultaneously
        start_time_fov = start_time + fov_group * time_between_timesteps * len(
            timesteps
        )
        if condition is None or len(condition) == 0:
            condition_fov = None
        elif len(condition) == 1:
            condition_fov = condition[0]
        else:
            condition_fov = condition[fov_index]
        for timestep in timesteps:
            if phase_id is not None:
                fname = f"{str(fov_index).zfill(3)}_{str(phase_id).zfill(2)}_{str(timestep).zfill(5)}"
            else:
                fname = f"{str(fov_index).zfill(3)}_{str(timestep).zfill(5)}"
            row = {
                "fov_object": fov,
                "fov": fov_index,
                "fov_x": fov.x,
                "fov_y": fov.y,
                "fov_z": fov.z,
                "fov_name": fov.name,
                "timestep": timestep,
                "time": start_time_fov + timestep * time_between_timesteps,
                "channels": tuple(dataclasses.asdict(channel) for channel in channels),
                "fname": fname,
            }
            if condition_fov is not None:
                row["cell_line"] = condition_fov
            if channel_optocheck is not None:
                row["optocheck"] = True if timestep in optocheck_timepoints else False
                if isinstance(channel_optocheck, list):
                    row["optocheck_channels"] = tuple(
                        dataclasses.asdict(channel) for channel in channel_optocheck
                    )
                else:
                    row["optocheck_channels"] = tuple(
                        [dataclasses.asdict(channel_optocheck)]
                    )
            dfs.append(row)

    df_acquire = pd.DataFrame(dfs)
    if phase_name is not None:
        df_acquire["phase"] = phase_name
    if phase_id is not None:
        df_acquire["phase_id"] = phase_id

    # Sort by time and fov for consistent ordering
    df_acquire = df_acquire.sort_values(by=["time", "fov"]).reset_index(drop=True)

    print(f"Total Experiment Time: {df_acquire['time'].max()/3600}h")
    return df_acquire


def apply_stim_treatments_to_df_acquire(
    df_acquire,
    stim_treatments,
    condition,
    n_fovs_per_well=None,
    add_stim_exposure_group=False,
    regular_spacing_between_stimulations=False,
):
    """Apply stim treatments to the df_acquire dataframe."""

    n_fovs = len(df_acquire["fov"].unique())
    n_stim_treatments = len(stim_treatments)
    if n_stim_treatments > 0:
        n_fovs_per_stim_condition = (
            n_fovs // n_stim_treatments // len(np.unique(condition))
        )
        stim_treatment_tot = []
        random.shuffle(stim_treatments)
        if n_fovs_per_well is None:
            for fov_index in range(0, n_fovs_per_stim_condition + 1):
                stim_treatment_tot.extend(stim_treatments)
            random.shuffle(stim_treatment_tot)
            if n_fovs % n_stim_treatments != 0:
                print(
                    f"Warning: Not equal number of fovs per stim condition. {n_fovs % n_stim_treatments} fovs will have repeated treatment"
                )
                stim_treatment_tot.extend(stim_treatments[: n_fovs % n_stim_treatments])
            print(f"Doing {n_fovs_per_stim_condition} experiment per stim condition")

            if len(condition) != 1:
                stim_treatment_tot = stim_treatment_tot * len(np.unique(condition))

            df_acquire = pd.merge(
                df_acquire,
                pd.DataFrame(stim_treatment_tot),
                left_on="fov",
                right_index=True,
            )
        else:
            stim_treatment_tot = []
            for cell_line in np.unique(condition):
                fovs_for_one_cell_line = df_acquire.query(f"cell_line == @cell_line")[
                    "fov"
                ].unique()
                stim_treat = [
                    stim for stim in stim_treatments for _ in range(n_fovs_per_well)
                ]
                if len(fovs_for_one_cell_line) != len(stim_treat):
                    print(
                        f"Warning: Number of fovs ({len(fovs_for_one_cell_line)}) for cell line {cell_line} does not match number of stim treatments ({len(stim_treat)})."
                    )
                stim_treat = pd.DataFrame(stim_treat)
                stim_treat["fov"] = fovs_for_one_cell_line
                stim_treatment_tot.append(stim_treat)
            stim_treat = pd.concat(stim_treatment_tot, ignore_index=True)
            df_acquire = pd.merge(
                df_acquire, stim_treat, left_on="fov", right_on="fov", how="left"
            )

        df_acquire["stim_exposure"] = np.nan

        for fov in df_acquire["fov"].unique():
            fov_data = df_acquire[df_acquire["fov"] == fov]

            stim_pattern = fov_data.iloc[0]

            if isinstance(stim_pattern["stim_timestep"], tuple) and isinstance(
                stim_pattern["stim_exposure_list"], tuple
            ):
                exposure_map = dict(
                    zip(
                        stim_pattern["stim_timestep"],
                        stim_pattern["stim_exposure_list"],
                    )
                )

                for timestep in fov_data["timestep"]:
                    if timestep in exposure_map:
                        mask = (df_acquire["fov"] == fov) & (
                            df_acquire["timestep"] == timestep
                        )
                        df_acquire.loc[mask, "stim_exposure"] = exposure_map[timestep]

        df_acquire["stim"] = df_acquire.apply(
            lambda row: (
                row["timestep"] in row["stim_timestep"] and row["stim_exposure"] > 0
            ),
            axis=1,
        )

    df_acquire = df_acquire.sort_values(by=["time", "fov"]).reset_index(drop=True)
    df_acquire = df_acquire.dropna(axis=1, how="all")
    if add_stim_exposure_group and regular_spacing_between_stimulations:
        spacing_interval = (
            df_acquire["stim_timestep"][0][1] - df_acquire["stim_timestep"][0][0]
        )
        for start in range(0, df_acquire["timestep"].max(), spacing_interval):
            end = start + spacing_interval
            mask = (df_acquire["timestep"] >= start) & (df_acquire["timestep"] < end)
            window = df_acquire.loc[mask, "stim_exposure"]
            value = window.dropna().iloc[0] if window.dropna().size > 0 else np.nan
            df_acquire.loc[mask, "stim_exposure"] = value

    else:
        df_acquire["stim_exposure"] = df_acquire["stim_exposure"].fillna(0)

    return df_acquire


def parse_filename(fname):
    stem = Path(fname).stem
    nums = re.findall(r"\d+", stem)
    if len(nums) >= 3:
        fov = int(nums[0])
        phase = int(nums[1])
        timestep = int(nums[2])
        return {"fname": fname, "fov": fov, "phase": phase, "timestep": timestep}
    elif len(nums) == 2:
        fov = int(nums[0])
        phase = None
        timestep = int(nums[1])
        return {"fname": fname, "fov": fov, "phase": phase, "timestep": timestep}
    elif len(nums) == 1:
        # fallback: treat as fov only
        fov = int(nums[0])
        return {"fname": fname, "fov": fov, "phase": None, "timestep": None}
    else:
        return {"fname": fname, "fov": None, "phase": None, "timestep": None}


def generate_exp_data_from_tracks(path):
    tracks_dir = Path(path) / "tracks"
    all_files = [p.name for p in tracks_dir.glob("*.parquet")]

    infos = [parse_filename(f) for f in all_files]
    # group by fov
    from collections import defaultdict

    fov_groups = defaultdict(list)
    for info in infos:
        if info["fov"] is None:
            continue
        fov_groups[info["fov"]].append(info)

    selected_files = []
    for fov, items in sorted(fov_groups.items()):
        has_phase = any(it["phase"] is not None for it in items)
        if has_phase:
            # choose highest phase for this fov, return all files in that phase
            max_phase = max(it["phase"] for it in items if it["phase"] is not None)
            chosen = [it["fname"] for it in items if it["phase"] == max_phase]
            reason = f"phase {max_phase} (highest)"
        else:
            # no phase info: choose files with highest timestep (likely one file)
            timesteps = [it["timestep"] for it in items if it["timestep"] is not None]
            if timesteps:
                max_ts = max(timesteps)
                chosen = [it["fname"] for it in items if it["timestep"] == max_ts]
                reason = f"timestep {max_ts} (highest)"
            else:
                chosen = [it["fname"] for it in items]
                reason = "no timestep/phase data"

        selected_files.extend(chosen)

    selected_files = sorted(selected_files)
    dfs = []
    for fov_i in selected_files:
        track_file = os.path.join(path, "tracks", fov_i)
        df = pd.read_parquet(track_file)
        dfs.append(df)
    pd.concat(dfs).to_parquet(os.path.join(path, "exp_data.parquet"))
