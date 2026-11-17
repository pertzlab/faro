from __future__ import annotations

import numpy as np
import os
from collections import namedtuple, defaultdict
from skimage.util import map_array
from faro.core.data_structures import (
    Channel,
    PowerChannel,
    FovState,
    RTMEvent,
    RTMSequence,
    WaitEvent,
)
import math
import random
import pandas as pd
import dataclasses
import re
from pathlib import Path

_TRACK_COLS_FOR_PARTICLES = frozenset({"fname", "label", "particle"})

# plate_row/plate_col are the useq well indices (0-based) when the position
# was assigned to a well in the position table / stage map; None otherwise.
FovPosition = namedtuple(
    "FovPosition",
    ["x", "y", "z", "name", "plate_row", "plate_col"],
    defaults=(None, None),
)


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


def validate_hardware(events, mmc, *, power_properties=None) -> bool:
    """Validate that event channels exist on the microscope and params are in range.

    Checks:
    1. All channel configs (imaging + stim) exist in a config group.
    2. Exposure values are within the camera's allowed range.
    3. Device property values (e.g. laser power) are within device limits.

    Returns True if all checks pass, False otherwise.
    Emits warnings for every problem found.
    """
    import warnings

    problems: list[str] = []

    # Build map: config_name → [group, ...]
    available: dict[str, list[str]] = {}
    for group in mmc.getAvailableConfigGroups():
        for config in mmc.getAvailableConfigs(group):
            available.setdefault(config, []).append(group)

    # Collect unique channels across all events (compatible with MDAEvent too)
    seen: dict[str, tuple] = {}  # config → (Channel, "imaging"|"stim")
    for event in events:
        for ch in getattr(event, "channels", ()):
            if ch.config not in seen:
                seen[ch.config] = (ch, "imaging")
        for ch in getattr(event, "stim_channels", ()):
            if ch.config not in seen:
                seen[ch.config] = (ch, "stim")
        for ch in getattr(event, "ref_channels", ()):
            if ch.config not in seen:
                seen[ch.config] = (ch, "ref")

    # 1. Check config existence
    for name, (ch, label) in seen.items():
        if name not in available:
            problems.append(
                f"{label.capitalize()} channel config '{name}' not found on "
                f"microscope. Available configs: {sorted(available.keys())}"
            )

    # 2. Check exposure against camera limits
    try:
        camera = mmc.getCameraDevice()
        if camera and mmc.hasPropertyLimits(camera, "Exposure"):
            lo = mmc.getPropertyLowerLimit(camera, "Exposure")
            hi = mmc.getPropertyUpperLimit(camera, "Exposure")
            checked_exposures: set[tuple[str, int]] = set()
            for event in events:
                for ch in (
                    *getattr(event, "channels", ()),
                    *getattr(event, "stim_channels", ()),
                    *getattr(event, "ref_channels", ()),
                ):
                    if ch.exposure is None:
                        continue
                    key = (ch.config, ch.exposure)
                    if key in checked_exposures:
                        continue
                    checked_exposures.add(key)
                    if ch.exposure < lo:
                        problems.append(
                            f"Channel '{ch.config}' exposure {ch.exposure} ms "
                            f"is below camera minimum ({lo} ms)"
                        )
                    if hi > 0 and ch.exposure > hi:
                        problems.append(
                            f"Channel '{ch.config}' exposure {ch.exposure} ms "
                            f"exceeds camera maximum ({hi} ms)"
                        )
    except Exception:
        pass  # camera not set or property unavailable

    # 3. Check device property limits (e.g. laser power)
    # PowerChannel has .power; the mapping config→(device, property) comes
    # from the microscope via power_properties.
    _pprops = power_properties or {}
    checked_props: set[tuple] = set()
    for event in events:
        for ch in (
            *getattr(event, "channels", ()),
            *getattr(event, "stim_channels", ()),
            *getattr(event, "ref_channels", ()),
        ):
            power = getattr(ch, "power", None)
            if power is None:
                continue
            mapping = _pprops.get(ch.config)
            if mapping is None:
                # A channel asks for a specific power but nothing maps its
                # config to a device/property -> the power is silently dropped
                # and the light source stays at its current value. Flag it so
                # this surfaces before the run instead of as a flat exposure.
                problems.append(
                    f"Channel '{ch.config}' sets power={power} but has no "
                    f"power-property mapping (not auto-detected, not in "
                    f"POWER_PROPERTIES). The power will NOT be applied. Add it "
                    f"to the microscope's POWER_PROPERTIES, e.g. "
                    f"{{'{ch.config}': ('<device>', '<Color>_Level')}}."
                )
                continue
            device_name, property_name = mapping
            key = (device_name, property_name, power)
            if key in checked_props:
                continue
            checked_props.add(key)
            try:
                if not mmc.hasPropertyLimits(device_name, property_name):
                    continue
                lo = mmc.getPropertyLowerLimit(device_name, property_name)
                hi = mmc.getPropertyUpperLimit(device_name, property_name)
                if power < lo:
                    problems.append(
                        f"Channel '{ch.config}': {property_name}={power} "
                        f"is below device minimum ({lo})"
                    )
                if hi > 0 and power > hi:
                    problems.append(
                        f"Channel '{ch.config}': {property_name}={power} "
                        f"exceeds device maximum ({hi})"
                    )
            except Exception:
                pass  # device/property not found

    if problems:
        for msg in problems:
            warnings.warn(msg, UserWarning)
    return len(problems) == 0


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
    particles = np.zeros_like(labels)
    if tracks.empty or not _TRACK_COLS_FOR_PARTICLES.issubset(tracks.columns):
        return particles
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
    try:
        mda_widget = viewer.window.dock_widgets["MDA"]
    except KeyError as e:
        raise KeyError(
            "MDA dock widget not registered. Click the 'MDA' button in the "
            "napari-micromanager toolbar (or call "
            "`main_window._show_dock_widget('MDA')` programmatically) "
            "before reading FOVs from the viewer."
        ) from e
    data_mda_fovs = mda_widget.value().stage_positions
    return [pos.model_dump() for pos in data_mda_fovs]


def generate_fov_positions_from_list(mic, data_mda_fovs):
    """Create FovPosition namedtuples from a list of position dicts."""
    fovs = []
    for i, fov in enumerate(data_mda_fovs):
        z = None if getattr(mic, "USE_ONLY_PFS", False) else fov.get("z")
        name = str(i) if fov.get("name") is None else fov["name"]
        fovs.append(
            FovPosition(
                x=fov.get("x"),
                y=fov.get("y"),
                z=z,
                name=name,
                plate_row=fov.get("plate_row"),
                plate_col=fov.get("plate_col"),
            )
        )
    return fovs


# Backwards-compat alias
generate_fov_objects_from_list = generate_fov_positions_from_list


def _well_columns(fov) -> dict:
    """Return plate_row/plate_col df_acquire columns for a FOV.

    Empty dict when the FOV has no well assignment, so runs without wells get
    no well columns at all (rows from FOVs without a well hold NaN in mixed
    runs).
    """
    plate_row = getattr(fov, "plate_row", None)
    plate_col = getattr(fov, "plate_col", None)
    if plate_row is None or plate_col is None:
        return {}
    return {"plate_row": plate_row, "plate_col": plate_col}


def generate_fov_positions(mic, viewer=None, filename=None, fake_fovs=None):
    """Create FovPosition namedtuples from viewer or file."""
    if fake_fovs is not None:
        return [FovPosition(x=0, y=0, z=None, name=str(i)) for i in range(fake_fovs)]
    elif filename is not None:
        data_mda_fovs = _get_mda_from_file(filename)
    elif viewer is not None:
        data_mda_fovs = _get_mda_from_viewer(viewer)
        if data_mda_fovs is None:
            assert False, "No fovs selected. Please select fovs in the MDA widget"
    else:
        assert False, "Either viewer, filename, or fake_fovs must be provided"

    return generate_fov_positions_from_list(mic, data_mda_fovs)


# Backwards-compat alias
generate_fov_objects = generate_fov_positions


def generate_df_acquire_simple(
    fovs, n_frames, time_between_timesteps, channels, start_time=0
):
    dfs = []
    for fov_index, fov in enumerate(fovs):
        well = _well_columns(fov)
        for timestep in range(n_frames):
            dfs.append(
                {
                    "fov": fov_index,
                    "fov_x": fov.x,
                    "fov_y": fov.y,
                    "fov_z": fov.z,
                    "fov_name": fov.name,
                    **well,
                    "timestep": timestep,
                    "time": start_time + timestep * time_between_timesteps,
                    "channels": tuple(dataclasses.asdict(ch) for ch in channels),
                    "fname": f"{str(fov_index).zfill(3)}_{str(timestep).zfill(5)}",
                }
            )
    df_acquire = (
        pd.DataFrame(dfs).sort_values(by=["time", "fov"]).reset_index(drop=True)
    )
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
    first_fov_index=0,
):
    """One row per (FOV, timestep) of a legacy df_acquire.

    The FOVs get the indices ``first_fov_index``, ``first_fov_index + 1``, ...
    in list order. Pass the offset when a FOV group is a slice of a larger
    list, e.g. ``generate_df_acquire(fovs[21:42], ..., first_fov_index=21)``,
    so the groups keep distinct FOV indices. ``channel_optocheck`` is acquired
    as reference channel(s) on ``optocheck_timepoints`` (default: last frame).
    """
    n_fovs_simultaneously = time_between_timesteps // time_per_fov
    optocheck_timepoints = (
        optocheck_timepoints if optocheck_timepoints is not None else [n_frames - 1]
    )
    timesteps = range(n_frames)
    dfs = []
    for i, fov in enumerate(fovs):
        fov_index = first_fov_index + i
        fov_group = i // n_fovs_simultaneously
        start_time_fov = start_time + fov_group * time_between_timesteps * len(
            timesteps
        )
        if condition is None or len(condition) == 0:
            condition_fov = None
        elif len(condition) == 1:
            condition_fov = condition[0]
        else:
            condition_fov = condition[fov_index]
        well = _well_columns(fov)
        for timestep in timesteps:
            if phase_id is not None:
                fname = f"{str(fov_index).zfill(3)}_{str(phase_id).zfill(2)}_{str(timestep).zfill(5)}"
            else:
                fname = f"{str(fov_index).zfill(3)}_{str(timestep).zfill(5)}"
            row = {
                "fov": fov_index,
                "fov_x": fov.x,
                "fov_y": fov.y,
                "fov_z": fov.z,
                "fov_name": fov.name,
                **well,
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
    randomize=False,
):
    """Apply stim treatments to the df_acquire dataframe."""

    n_fovs = len(df_acquire["fov"].unique())
    n_stim_treatments = len(stim_treatments)
    if n_stim_treatments > 0:
        n_fovs_per_stim_condition = (
            n_fovs // n_stim_treatments // len(np.unique(condition))
        )
        stim_treatment_tot = []
        if randomize:
            random.shuffle(stim_treatments)
        if n_fovs_per_well is None:
            for fov_index in range(0, n_fovs_per_stim_condition + 1):
                stim_treatment_tot.extend(stim_treatments)
            if randomize:
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


# ---------------------------------------------------------------------------
# RTMEvent-based helpers
# ---------------------------------------------------------------------------


def events_to_dataframe(events: list) -> pd.DataFrame:
    """Convert RTMEvent (or MDAEvent) list to summary DataFrame.

    Each row = one timepoint with channels + stim info merged.
    Compatible with both RTMEvent and plain useq.MDAEvent objects.
    """
    rows = []
    for e in events:
        if isinstance(e, WaitEvent):
            continue  # timed gap, not an acquired frame
        channels = getattr(e, "channels", ())
        stim_channels = getattr(e, "stim_channels", ())
        ref_channels = getattr(e, "ref_channels", ())

        # Fallback for plain MDAEvent: build from .channel + .exposure
        if not channels and getattr(e, "channel", None):
            channels = (Channel(config=e.channel.config, exposure=e.exposure or 0),)

        row = {
            "fov": e.index.get("p", 0),
            "timestep": e.index.get("t", 0),
            "time": e.min_start_time or 0,
            "x_pos": e.x_pos,
            "y_pos": e.y_pos,
            "z_pos": e.z_pos,
            "channels": tuple(dataclasses.asdict(ch) for ch in channels),
            "stim_channels": tuple(dataclasses.asdict(ch) for ch in stim_channels),
            "ref_channels": tuple(dataclasses.asdict(ch) for ch in ref_channels),
            "stim": len(stim_channels) > 0,
            "ref": len(ref_channels) > 0,
            **e.metadata,
        }
        if stim_channels:
            row["stim_power"] = getattr(stim_channels[0], "power", None)
            row["stim_exposure"] = stim_channels[0].exposure
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by=["timestep", "fov"]).reset_index(drop=True)


def merge_rtm_sequences(
    sequences: list[RTMSequence],
    time_per_fov: float = 0,
) -> list[RTMEvent]:
    """Merge multiple RTMSequences into a single event list, batching FOVs in parallel.

    Determines how many FOVs can be imaged within one timepoint interval
    (``interval // time_per_fov``) and groups them into parallel batches.
    FOVs within the same batch share timepoints (no time offset).  Overflow
    FOVs go into the next batch, which starts after the previous batch
    finishes.

    Example: 31 FOVs, ``time_per_fov=2``, interval=60 s → 30 FOVs fit per
    batch.  The first 30 run in parallel, FOV 31 runs after they finish.

    Args:
        sequences: RTMSequence objects to merge. Each may contain one or
            more FOVs.
        time_per_fov: Time (in seconds) to image one FOV.  When 0, all FOVs
            are merged in parallel with no batching.

    Returns:
        Flat list of RTMEvent with re-indexed FOVs, sequential timepoints
        per batch, and adjusted times.
    """
    if not sequences:
        return []

    # 1. Collect per-FOV event lists, re-indexing p globally
    fov_event_lists: list[list[RTMEvent]] = []
    global_p = 0
    for seq in sequences:
        events = list(seq)
        local_fovs = sorted({e.index.get("p", 0) for e in events})
        for lp in local_fovs:
            fov_evs = [e for e in events if e.index.get("p", 0) == lp]
            fov_event_lists.append(
                [
                    ev.model_copy(update={"index": {**dict(ev.index), "p": global_p}})
                    for ev in fov_evs
                ]
            )
            global_p += 1

    total_fovs = len(fov_event_lists)

    # 2. Determine how many FOVs fit in one interval
    if time_per_fov > 0:
        first_fov = fov_event_lists[0]
        unique_times = sorted({e.min_start_time or 0 for e in first_fov})
        if len(unique_times) >= 2:
            interval = unique_times[1] - unique_times[0]
        else:
            interval = 0
        n_parallel = (
            max(1, int(interval // time_per_fov)) if interval > 0 else total_fovs
        )
    else:
        n_parallel = total_fovs

    # 3. Group FOVs into parallel batches.
    # Wall-clock (min_start_time) is offset per batch so events stay in
    # chronological order; the ``t`` index is *not* offset, so every FOV
    # writes to its own per-FOV-relative t in the zarr store and overflow
    # batches stack along p instead of along t.
    result: list[RTMEvent] = []
    time_offset = 0.0

    for batch_start in range(0, total_fovs, n_parallel):
        batch = fov_event_lists[batch_start : batch_start + n_parallel]

        for fov_evs in batch:
            for ev in fov_evs:
                new_time = (ev.min_start_time or 0) + time_offset
                result.append(ev.model_copy(update={"min_start_time": new_time}))

        # Offset for next batch: last timepoint start + time to image batch FOVs
        batch_max_time = max(e.min_start_time or 0 for fov in batch for e in fov)
        time_offset += batch_max_time + len(batch) * time_per_fov

    result.sort(key=lambda e: (e.min_start_time or 0, e.index.get("p", 0)))
    return result


# ---------------------------------------------------------------------------
# Parallelisation helpers
# ---------------------------------------------------------------------------


def _infer_interval(events: list[RTMEvent]) -> float:
    """Infer the timepoint interval from events (time gap between first two unique times)."""
    unique_times = sorted({e.min_start_time or 0 for e in events})
    if len(unique_times) >= 2:
        return unique_times[1] - unique_times[0]
    return 0


def _resolve_n_parallel(
    events: list[RTMEvent],
    time_per_fov: float,
    n_parallel: int | None,
) -> int:
    """Return *n_parallel*, computing it from the interval if not given."""
    if n_parallel is not None:
        return n_parallel
    interval = _infer_interval(events)
    if interval > 0 and time_per_fov > 0:
        return max(1, int(interval // time_per_fov))
    return len({e.index.get("p", 0) for e in events})


def check_fov_batching(
    events: list[RTMEvent],
    time_per_fov: float,
    n_parallel: int | None = None,
) -> bool:
    """Check whether FOVs in an event list can be imaged in parallel.

    Args:
        events: Flat list of RTMEvent.
        time_per_fov: Time (in seconds) to image one FOV.
        n_parallel: Max FOVs per batch.  If *None*, computed from
            ``time_per_fov`` and the inferred timepoint interval.
    """
    n_parallel = _resolve_n_parallel(events, time_per_fov, n_parallel)
    n_fovs = len({e.index.get("p", 0) for e in events})
    if n_fovs <= n_parallel:
        print(
            f"Parallelisation OK: {n_fovs} FOV(s) fit in "
            f"{n_parallel} parallel slot(s)."
        )
        return True
    n_batches = math.ceil(n_fovs / n_parallel)
    print(
        f"Parallelisation NOT possible in one batch: {n_fovs} FOV(s) "
        f"need {n_batches} batch(es) of up to {n_parallel}. "
        f"Use apply_fov_batching() to adjust timing."
    )
    return False


def apply_fov_batching(
    events: list[RTMEvent],
    time_per_fov: float,
    n_parallel: int | None = None,
    offset_min_start_time: bool = True,
) -> list[RTMEvent]:
    """Adjust timing so that overflow FOVs run in subsequent batches.

    FOVs 0 .. ``n_parallel-1`` keep their original timing (batch 0).
    FOVs ``n_parallel`` .. ``2*n_parallel-1`` are offset so they start
    after batch 0 finishes, and so on.

    Args:
        events: Flat list of RTMEvent (e.g. from ``list(sequence)`` or
            ``merge_rtm_sequences``).
        time_per_fov: Time (in seconds) to image one FOV.
        n_parallel: Max FOVs per batch.  If *None*, computed from
            ``time_per_fov`` and the inferred timepoint interval.
        offset_min_start_time: When True (default), stagger each FOV's
            ``min_start_time`` by its position within its batch times
            ``time_per_fov``. FOVs in a batch are imaged sequentially,
            not simultaneously, so the k-th FOV of a batch only starts
            ~``k * time_per_fov`` after the first. Encoding that in
            ``min_start_time`` keeps the scheduled per-FOV frame interval
            consistent and makes lag measurement meaningful. The first
            FOV of every batch gets 0 offset (its batch wall-clock
            offset still applies for batches > 0).

    Returns:
        New list of RTMEvent with adjusted ``min_start_time`` and ``t``
        indices for overflow batches.
    """
    n_parallel = _resolve_n_parallel(events, time_per_fov, n_parallel)
    fov_ids = sorted({e.index.get("p", 0) for e in events})
    n_fovs = len(fov_ids)

    # A single batch with no per-FOV stagger requested is a no-op.
    if n_fovs <= n_parallel and not offset_min_start_time:
        return list(events)

    # Sorted position of each FOV: batch = pos // n_parallel,
    # within-batch slot = pos % n_parallel.
    fov_pos = {fov: i for i, fov in enumerate(fov_ids)}

    # Per-batch wall-clock offset — the ``t`` index stays per-FOV
    # relative (every FOV uses 0..N-1) so the writer's time axis is
    # aligned across batches instead of concatenated. Without this each
    # batch was mapped to a disjoint slab of t and the zarr store had
    # n_batches * N empty rows per FOV.
    batch0_events = [
        e for e in events if fov_pos[e.index.get("p", 0)] // n_parallel == 0
    ]
    max_time_batch0 = max((e.min_start_time or 0 for e in batch0_events), default=0)
    batch_duration = max_time_batch0 + n_parallel * time_per_fov

    result: list[RTMEvent] = []
    for ev in events:
        fov = ev.index.get("p", 0)
        pos = fov_pos[fov]
        batch = pos // n_parallel
        slot = pos % n_parallel

        offset = batch * batch_duration
        if offset_min_start_time:
            offset += slot * time_per_fov

        if offset == 0:
            result.append(ev)
        else:
            new_time = (ev.min_start_time or 0) + offset
            result.append(ev.model_copy(update={"min_start_time": new_time}))

    result.sort(key=lambda e: (e.min_start_time or 0, e.index.get("p", 0)))
    return result
