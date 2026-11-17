"""Assertions and helpers for the output of a hardware test run."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from faro.stimulation.base import StimWithImage


def parse_offset_env(raw: str) -> list[tuple[float, float]]:
    """Parse ``"dx0,dy0;dx1,dy1;..."`` into a list of (dx, dy) tuples."""
    offsets: list[tuple[float, float]] = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        dx_str, dy_str = pair.split(",")
        offsets.append((float(dx_str), float(dy_str)))
    return offsets


def assert_clean_run(controller, tmp_path: Path, *, expect_tracks: bool) -> None:
    """Post-run smoke checks shared by every hardware test.

    Fails loudly on any background-thread error (experiments log-and-
    continue, but a hardware test is meaningless if it swallows them)
    and confirms the run produced a napari-loadable OME-Zarr store.
    """
    if controller.background_errors:
        # Print full tracebacks to stdout so they appear in pytest -s
        # output but don't bloat the assertion message.
        for e in controller.background_errors:
            print(f"\n--- [{e.source}] {e.exc_type} ---\n{e.traceback}")
        summary = "\n".join(
            f"  [{e.source}] {e.exc_type}: {e.message}"
            for e in controller.background_errors
        )
        raise AssertionError(
            f"Background errors during acquisition:\n{summary}\n"
            f"(full tracebacks printed above)"
        )

    zarr_path = tmp_path / "acquisition.ome.zarr"
    assert zarr_path.is_dir(), f"OME-Zarr store not created at {zarr_path}"

    grp = zarr.open_group(str(zarr_path), mode="r")
    assert (
        "ome" in grp.attrs
    ), "OME metadata missing on root group — store will not load in napari"

    assert (tmp_path / "events.json").is_file(), "events.json not written"

    if expect_tracks:
        tracks_dir = tmp_path / "tracks"
        assert tracks_dir.is_dir(), "tracks/ folder not created"
        assert list(tracks_dir.glob("*.parquet")), "no track parquet files written"


def open_stim_channel_array(zarr_path):
    """Locate the first stim readout channel in a hardware-test OME-Zarr.

    Writer appends stim channels after imaging channels and labels them
    ``stim_0``, ``stim_1``, ... Multi-position runs use the direct-write
    path (single ``(t,p,c,y,x)`` array at ``"0"``); the ``omero``
    channel list is a sibling of ``multiscales`` under ``attrs["ome"]``.
    Returns ``(array, stim_channel_index)``.
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    ome = root.attrs["ome"]
    multiscales = ome["multiscales"][0]
    channels = ome["omero"]["channels"]
    stim_idx = next(i for i, c in enumerate(channels) if c["label"].startswith("stim_"))
    arr = root[multiscales["datasets"][0]["path"]]
    return arr, stim_idx


class DelayedMaskStim(StimWithImage):
    """StimWithImage stub that sleeps before returning a full-image mask.

    Used by lag / timeout tests that need the mask request to go
    through the Analyzer's async queue (not the synchronous base-Stim
    path) so sleep time directly translates into pipeline lag. The
    ``metadata["img_shape"]`` key isn't populated on the storage-worker
    path, so we derive shape from the raw image.
    """

    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def get_stim_mask(self, metadata: dict, img):
        time.sleep(self.delay_s)
        h, w = img.shape[-2], img.shape[-1]
        return np.ones((h, w), dtype=np.uint8), None


def load_tracks_df(tmp_path: Path) -> pd.DataFrame:
    """Concatenate all per-FOV tracks parquets into a single DataFrame.

    Returns an empty DataFrame if no parquet files exist; the caller
    decides whether that's an error.
    """
    parts = sorted((tmp_path / "tracks").glob("*_latest.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def assert_timestep_ordering(df: pd.DataFrame) -> None:
    """PR #6 / TracksDispenser contract: ``timestep`` equals ``fov_timestep``.

    Under concurrent pipeline workers a thread-order race would desync
    the MDA timestep from the per-FOV counter. The fix (TracksDispenser)
    serializes handoff in acquisition order, making the two columns
    match row-for-row.
    """
    if "fov_timestep" not in df.columns:
        return
    mismatched = df[df["timestep"] != df["fov_timestep"]]
    assert mismatched.empty, (
        f"{len(mismatched)} rows where timestep != fov_timestep — "
        f"TracksDispenser ordering broken. Sample:\n"
        f"{mismatched[['fov', 'timestep', 'fov_timestep']].head()}"
    )
