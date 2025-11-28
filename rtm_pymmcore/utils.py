import numpy as np
import os
from skimage.util import map_array
from rtm_pymmcore.data_structures import Fov
import random
import pandas as pd
import dataclasses
import re
from pathlib import Path


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
        fov_object.z = fov.get("z") if not mic.USE_ONLY_PFS else None
        fov_object.name = str(i) if fov["name"] is None else fov["name"]
        fovs.append(fov_object)
    return fovs


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
            for fov_index in range(0, n_fovs_per_stim_condition):
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


def get_rotating_fov_indices(n_fovs, n_fovs_per_batch, phase_idx):
    """
    Get FOV indices for a specific phase with rotation.

    FOVs are divided into batches and rotated so that each batch is used once
    before any batch is reused.

    Args:
        n_fovs: Total number of FOVs
        n_fovs_per_batch: Number of FOVs per batch
        phase_idx: Phase index (0-based)

    Returns:
        list: FOV indices for this phase

    Example:
        n_fovs=20, n_fovs_per_batch=5 creates 4 batches:
        - Phase 0: [0, 1, 2, 3, 4]
        - Phase 1: [5, 6, 7, 8, 9]
        - Phase 2: [10, 11, 12, 13, 14]
        - Phase 3: [15, 16, 17, 18, 19]
        - Phase 4: [0, 1, 2, 3, 4] (cycle repeats)
    """
    n_batches = n_fovs // n_fovs_per_batch
    batch_idx = phase_idx % n_batches

    start_idx = batch_idx * n_fovs_per_batch
    end_idx = start_idx + n_fovs_per_batch

    return list(range(start_idx, end_idx)), batch_idx


def generate_df_acquire_single_phase(
    fovs,
    channels,
    phase_id,
    phase_name,
    fov_indices,
    n_frames,
    imaging_interval=15,
    channel_optocheck=None,
    optocheck_timepoints=None,
    condition=None,
    stim_exposure=None,
    stim_cell_percentage=None,
    stim_edge_distance=None,
    stim_pattern=None,
    stim_power=10,
    stim_channel_name="CyanStim",
    stim_channel_group="TTL_ERK",
    stim_channel_device_name="Spectra",
    stim_channel_power_property_name="Cyan_Level",
):
    """
    Generate df_acquire for a single phase with specific FOVs and stimulation parameters.

    Note: timestep and time always start at 0 for each phase. This is intentional as each
    phase is acquired and analyzed independently.

    Args:
        fovs: List of all FOV objects
        channels: List of imaging channels
        phase_id: Unique identifier for this phase
        phase_name: Human-readable name for this phase
        fov_indices: List of FOV indices to image in this phase
        n_frames: Number of frames to acquire in this phase
        imaging_interval: Time between frames in seconds
        channel_optocheck: Optional optocheck channel
        optocheck_timepoints: Timesteps for optocheck (relative to this phase)
        condition: List of conditions per FOV
        stim_exposure: Stimulation exposure in ms (None = no stim)
        stim_cell_percentage: Cell percentage for stimulation (0-1)
        stim_edge_distance: Edge distance for stimulation in pixels
        stim_pattern: Stimulation pattern ("every_frame", "every_2nd", "every_4th")
        stim_power: Stimulation power level
        stim_channel_name: Stimulation channel name
        stim_channel_group: Stimulation channel group
        stim_channel_device_name: Stimulation device name
        stim_channel_power_property_name: Power property name

    Returns:
        pd.DataFrame: Acquisition dataframe for this phase
    """
    dfs = []
    timestep = 0  # Always start at 0 for each phase
    current_time = 0  # Always start at 0 for each phase

    for frame_idx in range(n_frames):
        for fov_idx in fov_indices:
            fov = fovs[fov_idx]

            # Determine condition for this FOV
            if condition is None or len(condition) == 0:
                condition_fov = None
            elif len(condition) == 1:
                condition_fov = condition[0]
            else:
                condition_fov = condition[fov_idx]

            fname = f"{str(fov_idx).zfill(3)}_{str(phase_id).zfill(2)}_{str(timestep).zfill(5)}"

            row = {
                "fov_object": fov,
                "fov": fov_idx,
                "fov_x": fov.x,
                "fov_y": fov.y,
                "fov_z": fov.z,
                "fov_name": fov.name,
                "timestep": timestep,
                "fov_timestep": frame_idx,
                "time": current_time,
                "channels": tuple(dataclasses.asdict(channel) for channel in channels),
                "fname": fname,
                "phase": phase_name,
                "phase_id": phase_id,
            }

            if condition_fov is not None:
                row["cell_line"] = condition_fov

            if channel_optocheck is not None:
                if optocheck_timepoints is not None:
                    row["optocheck"] = timestep in optocheck_timepoints
                else:
                    row["optocheck"] = False

                if isinstance(channel_optocheck, list):
                    row["optocheck_channels"] = tuple(
                        dataclasses.asdict(channel) for channel in channel_optocheck
                    )
                else:
                    row["optocheck_channels"] = tuple(
                        [dataclasses.asdict(channel_optocheck)]
                    )

            # Add stimulation parameters if specified
            if stim_exposure is not None:
                row["stim_exposure"] = stim_exposure
                row["stim_power"] = stim_power
                row["stim_channel_name"] = stim_channel_name
                row["stim_channel_group"] = stim_channel_group
                row["stim_channel_device_name"] = stim_channel_device_name
                row["stim_channel_power_property_name"] = (
                    stim_channel_power_property_name
                )
                row["stim_cell_percentage"] = (
                    stim_cell_percentage if stim_cell_percentage is not None else 0.3
                )
                row["stim_edge_distance"] = (
                    stim_edge_distance if stim_edge_distance is not None else 5
                )
                row["stim_pattern"] = (
                    stim_pattern if stim_pattern is not None else "every_frame"
                )

                # Determine if this frame should be stimulated
                should_stim = False
                if stim_pattern == "every_frame":
                    should_stim = True
                elif stim_pattern == "every_2nd" and frame_idx % 2 == 0:
                    should_stim = True
                elif stim_pattern == "every_4th" and frame_idx % 4 == 0:
                    should_stim = True

                row["stim"] = should_stim
            else:
                row["stim"] = False
                row["stim_exposure"] = 0

            dfs.append(row)
            timestep += 1

        current_time += imaging_interval

    df_phase = pd.DataFrame(dfs)

    # If stimulation is enabled, create stim_timestep and stim_exposure_list tuples
    if stim_exposure is not None:
        stim_timesteps = df_phase[df_phase["stim"]]["timestep"].unique()
        stim_timestep_tuple = tuple(int(x) for x in sorted(stim_timesteps))
        stim_exposure_list_tuple = tuple([stim_exposure] * len(stim_timestep_tuple))

        # Initialize columns first
        df_phase["stim_timestep"] = None
        df_phase["stim_exposure_list"] = None

        # Then set values
        for idx in df_phase.index:
            df_phase.at[idx, "stim_timestep"] = stim_timestep_tuple
            df_phase.at[idx, "stim_exposure_list"] = stim_exposure_list_tuple

    return df_phase


def generate_df_acquire_rotating_fovs(
    fovs,
    channels,
    n_fovs_per_batch=5,
    imaging_interval=15,  # seconds between frames
    batch_duration=1800,  # 30 minutes in seconds
    n_cycles=5,  # number of times to repeat the entire rotation
    start_time=0,
    channel_optocheck=None,
    optocheck_timepoints=None,
    condition=None,
):
    """
    Generate a df_acquire for rotating FOV batches.

    Args:
        fovs: List of FOV objects (should be 20 total: 10 mcherry, 10 lifeact)
        channels: List of channel configurations
        n_fovs_per_batch: Number of FOVs to image simultaneously (default: 5)
        imaging_interval: Time between frames in seconds (default: 15s)
        batch_duration: Duration to image each batch in seconds (default: 1800s = 30min)
        n_cycles: Number of times to repeat the entire rotation (default: 5)
        start_time: Starting time in seconds (default: 0)
        channel_optocheck: Optional optocheck channel configuration
        optocheck_timepoints: List of global timesteps for optocheck
        condition: List of conditions (cell lines) per FOV

    Returns:
        pd.DataFrame: Acquisition dataframe with rotating FOV batches

    Note: Each batch is a phase. The terms are used interchangeably in the code.
    """
    n_fovs = len(fovs)
    n_batches = n_fovs // n_fovs_per_batch
    frames_per_batch = int(batch_duration / imaging_interval)

    # Create FOV batches
    fov_batches = []
    for i in range(n_batches):
        batch_start = i * n_fovs_per_batch
        batch_end = batch_start + n_fovs_per_batch
        fov_batches.append(list(range(batch_start, batch_end)))

    print(f"Created {n_batches} batches with {n_fovs_per_batch} FOVs each")
    print(
        f"Each batch will be imaged for {batch_duration/60} minutes ({frames_per_batch} frames)"
    )
    print(f"Total cycle duration: {n_batches * batch_duration / 3600} hours")
    print(f"Repeating {n_cycles} times")

    dfs = []
    global_timestep = 0
    current_time = start_time

    for cycle in range(n_cycles):
        for batch_idx, fov_indices in enumerate(fov_batches):
            phase_id = (
                cycle * n_batches + batch_idx
            )  # phase_id = unique identifier for each batch
            phase_name = f"cycle{cycle}_batch{batch_idx}"

            # Image this batch for the specified duration
            for frame_in_batch in range(frames_per_batch):
                for fov_idx in fov_indices:
                    fov = fovs[fov_idx]

                    # Determine condition for this FOV
                    if condition is None or len(condition) == 0:
                        condition_fov = None
                    elif len(condition) == 1:
                        condition_fov = condition[0]
                    else:
                        condition_fov = condition[fov_idx]

                    fname = f"{str(fov_idx).zfill(3)}_{str(phase_id).zfill(2)}_{str(global_timestep).zfill(5)}"

                    row = {
                        "fov_object": fov,
                        "fov": fov_idx,
                        "fov_x": fov.x,
                        "fov_y": fov.y,
                        "fov_z": fov.z,
                        "fov_name": fov.name,
                        "timestep": global_timestep,
                        "fov_timestep": frame_in_batch,  # Frame within this batch/phase
                        "time": current_time,
                        "channels": tuple(
                            dataclasses.asdict(channel) for channel in channels
                        ),
                        "fname": fname,
                        "phase": phase_name,
                        "phase_id": phase_id,
                        "batch": batch_idx,  # batch within current cycle
                        "cycle": cycle,
                    }

                    if condition_fov is not None:
                        row["cell_line"] = condition_fov

                    if channel_optocheck is not None:
                        if optocheck_timepoints is not None:
                            row["optocheck"] = global_timestep in optocheck_timepoints
                        else:
                            row["optocheck"] = False

                        if isinstance(channel_optocheck, list):
                            row["optocheck_channels"] = tuple(
                                dataclasses.asdict(channel)
                                for channel in channel_optocheck
                            )
                        else:
                            row["optocheck_channels"] = tuple(
                                [dataclasses.asdict(channel_optocheck)]
                            )

                    dfs.append(row)
                    global_timestep += 1

                current_time += imaging_interval

    df_acquire = pd.DataFrame(dfs)
    df_acquire = df_acquire.sort_values(by=["time", "fov"]).reset_index(drop=True)

    total_hours = df_acquire["time"].max() / 3600
    print(f"\nTotal Experiment Time: {total_hours:.2f} hours")
    print(f"Total frames per FOV: {len(df_acquire[df_acquire['fov'] == 0])}")
    print(f"Total frames: {len(df_acquire)}")

    return df_acquire


def apply_stim_to_rotating_fovs(
    df_acquire,
    stim_exposures=[50, 250, 500],
    stim_cell_percentages=[0.1, 0.2, 0.3],
    stim_edge_distances=[0, 5, 10],
    stim_patterns=["every_frame", "every_2nd", "every_4th"],
    stim_power=10,
    stim_channel_name="CyanStim",
    stim_channel_group="TTL_ERK",
    stim_channel_device_name="Spectra",
    stim_channel_power_property_name="Cyan_Level",
    use_condition_factor=False,
):
    """
    Apply stimulation conditions to rotating FOV dataframe.
    Each unique batch (phase) gets a different combination of stimulation parameters.

    Args:
        df_acquire: DataFrame from generate_df_acquire_rotating_fovs
        stim_exposures: List of stimulation exposure values in ms (default: [50, 250, 500])
        stim_cell_percentages: List of cell percentage values 0-1 (default: [0.1, 0.2, 0.3])
        stim_edge_distances: List of edge distances in pixels (default: [0, 5, 10])
        stim_patterns: List of stimulation patterns (default: ["every_frame", "every_2nd", "every_4th"])
                      - "every_frame": stimulate every frame
                      - "every_2nd": stimulate every 2nd frame (e.g., every 30s if imaging_interval=15s)
                      - "every_4th": stimulate every 4th frame (e.g., every minute if imaging_interval=15s)
        stim_power: Stimulation power level (default: 10)
        stim_channel_name: Name of stimulation channel (default: "CyanStim")
        stim_channel_group: Channel group (default: "TTL_ERK")
        stim_channel_device_name: Device name (default: "Spectra")
        stim_channel_power_property_name: Property name for power (default: "Cyan_Level")

    Returns:
        pd.DataFrame: DataFrame with stimulation parameters added

    Note: With default parameters (3×3×3×3), there are 81 possible combinations.
          The function cycles through these combinations for each unique batch/phase.
    """
    import itertools

    # Generate base parameter combinations (without condition)
    base_param_combos = list(
        itertools.product(
            stim_exposures, stim_cell_percentages, stim_edge_distances, stim_patterns
        )
    )

    unique_phases = sorted(df_acquire["phase_id"].unique())
    n_phases = len(unique_phases)

    # Optionally incorporate condition as an additional factor
    if use_condition_factor:
        if "cell_line" not in df_acquire.columns:
            raise ValueError(
                "use_condition_factor=True aber Spalte 'cell_line' fehlt im DataFrame."
            )
        unique_conditions = sorted(df_acquire["cell_line"].dropna().unique())
        n_conditions = len(unique_conditions)
        # Expand combinations by condition dimension
        all_combinations = [
            (*params, cond)
            for params in base_param_combos
            for cond in unique_conditions
        ]
        print(
            "Condition-Faktor aktiv: Kombinationen werden über Bedingungen erweitert."
        )
    else:
        unique_conditions = []
        n_conditions = 0
        all_combinations = base_param_combos

    n_combinations = len(all_combinations)
    print(f"Total stimulation combinations: {n_combinations}")
    print(
        f"Parameter-Kombinationen: {len(stim_exposures)} exposures × {len(stim_cell_percentages)} cell% × "
        f"{len(stim_edge_distances)} edge distances × {len(stim_patterns)} patterns"
        + (f" × {n_conditions} conditions" if use_condition_factor else "")
    )
    print(
        f"Phasen: {n_phases}"
        + (f", Bedingungen: {n_conditions}" if use_condition_factor else "")
    )

    # Build assignment groups: either per phase or per (phase, condition)
    if use_condition_factor:
        assignment_groups = [
            (phase_id, cond) for phase_id in unique_phases for cond in unique_conditions
        ]
    else:
        assignment_groups = [(phase_id, None) for phase_id in unique_phases]
    n_groups = len(assignment_groups)
    print(f"Zu belegende Gruppen: {n_groups}")

    if n_groups < n_combinations:
        print(
            f"WARNING: Weniger Gruppen ({n_groups}) als Kombinationen ({n_combinations})."
        )
        print(f"         Nur die ersten {n_groups} Kombinationen werden genutzt.")
    elif n_groups > n_combinations:
        print(f"INFO: Mehr Gruppen ({n_groups}) als Kombinationen ({n_combinations}).")
        print("      Kombinationen werden zyklisch wiederholt.")

    # Initialize columns
    df_acquire["stim"] = False
    df_acquire["stim_exposure"] = 0
    df_acquire["stim_power"] = stim_power
    df_acquire["stim_channel_name"] = stim_channel_name
    df_acquire["stim_channel_group"] = stim_channel_group
    df_acquire["stim_channel_device_name"] = stim_channel_device_name
    df_acquire["stim_channel_power_property_name"] = stim_channel_power_property_name
    df_acquire["stim_cell_percentage"] = 0.0
    df_acquire["stim_edge_distance"] = 0
    df_acquire["stim_pattern"] = "none"
    df_acquire["stim_timestep"] = None
    df_acquire["stim_exposure_list"] = None

    # Assign combinations to each phase
    for group_idx, (phase_id, cond) in enumerate(assignment_groups):
        combo_idx = group_idx % n_combinations
        if use_condition_factor:
            exposure, cell_pct, edge_dist, pattern, combo_condition = all_combinations[
                combo_idx
            ]
        else:
            exposure, cell_pct, edge_dist, pattern = all_combinations[combo_idx]
            combo_condition = cond  # None

        phase_mask = df_acquire["phase_id"] == phase_id
        if use_condition_factor:
            group_mask = phase_mask & (df_acquire["cell_line"] == cond)
        else:
            group_mask = phase_mask

        if group_mask.sum() == 0:
            print(
                f"WARN: Gruppe (phase={phase_id}, condition={cond}) hat keine Zeilen – übersprungen."
            )
            continue

        df_acquire.loc[group_mask, "stim_power"] = stim_power
        df_acquire.loc[group_mask, "stim_channel_name"] = stim_channel_name
        df_acquire.loc[group_mask, "stim_channel_group"] = stim_channel_group
        df_acquire.loc[group_mask, "stim_channel_device_name"] = (
            stim_channel_device_name
        )
        df_acquire.loc[group_mask, "stim_channel_power_property_name"] = (
            stim_channel_power_property_name
        )
        df_acquire.loc[group_mask, "stim_cell_percentage"] = cell_pct
        df_acquire.loc[group_mask, "stim_edge_distance"] = edge_dist
        df_acquire.loc[group_mask, "stim_pattern"] = pattern

        group_data = df_acquire[group_mask].copy()
        all_stim_timesteps = []
        for fov in group_data["fov"].unique():
            fov_group_mask = group_mask & (df_acquire["fov"] == fov)
            fov_data = df_acquire[fov_group_mask]
            fov_timesteps = fov_data["fov_timestep"].values
            global_timesteps = fov_data["timestep"].values
            if pattern == "every_frame":
                stim_indices = range(len(fov_timesteps))
            elif pattern == "every_2nd":
                stim_indices = range(0, len(fov_timesteps), 2)
            elif pattern == "every_4th":
                stim_indices = range(0, len(fov_timesteps), 4)
            else:
                stim_indices = []
            for idx in stim_indices:
                fov_ts = fov_timesteps[idx]
                global_ts = global_timesteps[idx]
                frame_mask = fov_group_mask & (df_acquire["fov_timestep"] == fov_ts)
                df_acquire.loc[frame_mask, "stim"] = True
                df_acquire.loc[frame_mask, "stim_exposure"] = exposure
                if global_ts not in all_stim_timesteps:
                    all_stim_timesteps.append(global_ts)
        all_stim_timesteps.sort()
        # ensure pure Python int (avoid numpy.int64 leaking into tuple)
        stim_timestep_tuple = tuple(int(x) for x in all_stim_timesteps)
        stim_exposure_list_tuple = tuple([exposure] * len(all_stim_timesteps))
        for idx in df_acquire.index[group_mask]:
            df_acquire.at[idx, "stim_timestep"] = stim_timestep_tuple
            df_acquire.at[idx, "stim_exposure_list"] = stim_exposure_list_tuple
        print(
            f"Phase {phase_id}, Condition {cond if cond is not None else '-'}: combo={combo_idx}, exposure={exposure}ms, cell%={cell_pct*100:.0f}%, edge_dist={edge_dist}px, pattern={pattern}, stim_frames={len(all_stim_timesteps)}"
        )

    # Summary statistics
    n_stim_frames = df_acquire["stim"].sum()
    n_total_frames = len(df_acquire)
    print(
        f"\nTotal frames with stimulation: {n_stim_frames}/{n_total_frames} "
        f"({n_stim_frames/n_total_frames*100:.1f}%)"
    )

    return df_acquire


def generate_dummy_fovs(n_fovs=20, x_start=0, y_start=0, spacing=500):
    """
    Generate dummy FOV objects for testing.

    Args:
        n_fovs: Number of FOVs to create
        x_start: Starting X position
        y_start: Starting Y position
        spacing: Spacing between FOVs in microns

    Returns:
        List of Fov objects
    """
    fovs = []
    for i in range(n_fovs):
        fov_object = Fov(i)
        fov_object.x = x_start + (i % 5) * spacing  # 5 FOVs per row
        fov_object.y = y_start + (i // 5) * spacing
        fov_object.z = None
        fov_object.name = f"FOV_{i}"
        fovs.append(fov_object)
    return fovs


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
