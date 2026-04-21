"""Local smoke test for OmeZarrWriter / OmeZarrWriterPlate.

Runs the writers against synthetic numpy frames — no Micro-Manager, no
microscope, no Controller. Exercises all three storage layouts:

  * multi-position direct  (OmeZarrWriter, n_pos > 1)
  * single-position stream (OmeZarrWriter, n_pos == 1, via ome-writers)
  * plate layout           (OmeZarrWriterPlate)

For each scenario: init_stream -> write raw + stim + label frames ->
close -> reopen with zarr and assert shapes/axes. With ``--napari`` the
script also opens each store in a napari.Viewer via napari-ome-zarr and
prints the resulting layers (names, shapes, dtypes).

Not a pytest file — run directly::

    uv run python tests/manual_smoke_writers.py
    uv run python tests/manual_smoke_writers.py --napari
    uv run python tests/manual_smoke_writers.py --scenario plate --napari
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from faro.core.writers import OmeZarrWriter, OmeZarrWriterPlate


IMAGE_H = 128
IMAGE_W = 128
N_T = 4
N_IMG_CH = 2
N_STIM_CH = 1
IMG_CHANNEL_NAMES = ["DAPI", "FITC"]


@dataclass
class Scenario:
    name: str
    n_pos: int
    writer_cls: type
    skip_napari_labels: str | None = None


# Plate-label napari skip rationale — unskip when both upstream PRs land:
#   ome/ome-zarr-py reader.py has ``# self.add(zarr, plate_labels=True)``
#   commented out (line ~58). PR ome/ome-zarr-py#207 closed unmerged;
#   companion PR ome/napari-ome-zarr#54 still open. Tracking issue
#   ome/ome-zarr-py#65.
_PLATE_LABELS_SKIP = (
    "napari-ome-zarr does not surface per-well label groups; "
    "see ome/ome-zarr-py#65, ome/napari-ome-zarr#54"
)

SCENARIOS: dict[str, Scenario] = {
    "direct": Scenario("multi-pos direct", n_pos=3, writer_cls=OmeZarrWriter),
    "stream": Scenario("single-pos stream", n_pos=1, writer_cls=OmeZarrWriter),
    "plate": Scenario(
        "plate",
        n_pos=3,
        writer_cls=OmeZarrWriterPlate,
        skip_napari_labels=_PLATE_LABELS_SKIP,
    ),
}


def _synthetic_raw(t: int, p: int) -> np.ndarray:
    """All imaging channels for one (t, p) — shape (C, Y, X) uint16."""
    rng = np.random.default_rng(seed=1000 * p + t)
    img = rng.integers(0, 65535, size=(N_IMG_CH, IMAGE_H, IMAGE_W), dtype=np.uint16)
    img[:, :16, :16] = (t + 1) * 5000
    img[:, :16, -16:] = (p + 1) * 5000
    return img


def _synthetic_stim(t: int, p: int) -> np.ndarray:
    """Stim readout, (Y, X) uint16."""
    frame = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint16)
    frame[t * 16 : (t + 1) * 16, p * 16 : (p + 1) * 16] = 50_000
    return frame


def _synthetic_label(t: int, p: int) -> np.ndarray:
    """Segmentation labels, (Y, X) int32."""
    arr = np.zeros((IMAGE_H, IMAGE_W), dtype=np.int32)
    arr[20:40, 20:40] = t + 1
    arr[60:80, 60:80] = (p + 1) * 10
    return arr


def run_scenario(scenario: Scenario, out_dir: Path) -> Path:
    print(f"\n=== {scenario.name} ({scenario.writer_cls.__name__}, n_pos={scenario.n_pos}) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    position_names = [f"Pos{i}" for i in range(scenario.n_pos)]
    writer = scenario.writer_cls(
        storage_path=str(out_dir),
        store_stim_images=True,
        n_timepoints=N_T,
    )
    writer.init_stream(
        position_names=position_names,
        channel_names=IMG_CHANNEL_NAMES,
        image_height=IMAGE_H,
        image_width=IMAGE_W,
        n_timepoints=N_T,
        n_stim_channels=N_STIM_CH,
    )

    for t in range(N_T):
        for p in range(scenario.n_pos):
            meta = {"timestep": t, "fov": p, "stim": (t % 2 == 0)}
            writer.write(_synthetic_raw(t, p), meta, "raw")
            if t % 2 == 0:
                writer.write(
                    _synthetic_stim(t, p),
                    {**meta, "stim": True},
                    "stim",
                )
            writer.write(_synthetic_label(t, p), meta, "labels")

    writer.close()
    print(f"  wrote {out_dir}")
    return Path(writer._zarr_path)


def validate_direct_store(zarr_path: Path, n_pos: int) -> None:
    print(f"  [validate] direct store {zarr_path.name}")
    root = zarr.open(str(zarr_path), mode="r")
    print(root.tree())

    raw = root["0"]
    expected_shape = (
        (N_T, n_pos, N_IMG_CH + N_STIM_CH, IMAGE_H, IMAGE_W)
        if n_pos > 1
        else (N_T, N_IMG_CH + N_STIM_CH, IMAGE_H, IMAGE_W)
    )
    assert raw.shape == expected_shape, f"raw shape {raw.shape} != {expected_shape}"
    assert str(raw.dtype) == "uint16"

    ome = root.attrs["ome"]
    axes = [a["name"] for a in ome["multiscales"][0]["axes"]]
    expected_axes = (
        ["t", "p", "c", "y", "x"] if n_pos > 1 else ["t", "c", "y", "x"]
    )
    assert axes == expected_axes, f"axes {axes} != {expected_axes}"

    channel_labels = [c["label"] for c in ome["omero"]["channels"]]
    assert channel_labels == IMG_CHANNEL_NAMES + ["stim_0"]

    labels_grp = root["labels"]["labels"]["0"]
    expected_label_shape = (
        (N_T, n_pos, IMAGE_H, IMAGE_W) if n_pos > 1 else (N_T, IMAGE_H, IMAGE_W)
    )
    assert labels_grp.shape == expected_label_shape
    print(f"  [validate] OK  raw={raw.shape} labels={labels_grp.shape} axes={axes}")


def validate_plate_store(zarr_path: Path, n_pos: int) -> None:
    print(f"  [validate] plate store {zarr_path.name}")
    root = zarr.open(str(zarr_path), mode="r")
    print(root.tree())

    for p in range(n_pos):
        well = root[f"A/{p + 1}/0"]
        raw = well["0"]
        assert raw.shape[-2:] == (IMAGE_H, IMAGE_W)
        assert raw.shape[0] == N_T, f"well A/{p + 1} t={raw.shape[0]} != {N_T}"
        assert raw.shape[-3] == N_IMG_CH + N_STIM_CH
        lbl = well["labels"]["labels"]["0"]
        assert lbl.shape[-2:] == (IMAGE_H, IMAGE_W)
    print(f"  [validate] OK  {n_pos} wells, each with labels")


def validate(zarr_path: Path, scenario: Scenario) -> None:
    if scenario.writer_cls is OmeZarrWriterPlate:
        validate_plate_store(zarr_path, scenario.n_pos)
    else:
        validate_direct_store(zarr_path, scenario.n_pos)


def _layer_shape(layer) -> tuple[int, ...]:
    """Highest-resolution shape of a napari layer (handles multiscale)."""
    data = layer.data
    if isinstance(data, list):
        return tuple(data[0].shape)
    return tuple(data.shape)


def _expected_layer_shapes(scenario: Scenario) -> dict[str, tuple[int, ...]]:
    """Per-layer shape after napari's channel_axis split.

    Image layers drop the channel axis (split into one layer each).
    Plate tiles wells along x: width becomes ``IMAGE_W * n_pos``.
    """
    if scenario.writer_cls is OmeZarrWriterPlate:
        img_shape = (N_T, IMAGE_H, IMAGE_W * scenario.n_pos)
        return {name: img_shape for name in IMG_CHANNEL_NAMES + ["stim_0"]}

    if scenario.n_pos > 1:
        img_shape = (N_T, scenario.n_pos, IMAGE_H, IMAGE_W)
    else:
        img_shape = (N_T, IMAGE_H, IMAGE_W)
    return {
        **{name: img_shape for name in IMG_CHANNEL_NAMES + ["stim_0"]},
        "labels": img_shape,
    }


def open_in_napari(zarr_path: Path, scenario: Scenario, keep_open: bool) -> None:
    try:
        import napari
        from napari_ome_zarr import napari_get_reader
    except ImportError as e:
        print(f"  [napari] skipped: {e}")
        return

    print(f"  [napari] opening {zarr_path}")
    reader = napari_get_reader(str(zarr_path))
    if reader is None:
        print("  [napari] napari-ome-zarr refused the path")
        return
    layer_tuples = reader(str(zarr_path))

    print(f"  [napari] reader returned {len(layer_tuples)} layer(s):")
    for i, lt in enumerate(layer_tuples):
        data, meta, ltype = (lt + (None,))[:3] if len(lt) < 3 else lt
        name = meta.get("name", f"layer_{i}") if isinstance(meta, dict) else f"layer_{i}"
        if isinstance(data, list):
            shapes = [getattr(d, "shape", None) for d in data]
            dtypes = [str(getattr(d, "dtype", "?")) for d in data]
            print(f"    [{i}] {name!r} type={ltype} multiscale={shapes} dtypes={dtypes}")
        else:
            print(f"    [{i}] {name!r} type={ltype} shape={getattr(data, 'shape', None)} "
                  f"dtype={getattr(data, 'dtype', '?')}")

    viewer = napari.Viewer(show=True)
    for lt in layer_tuples:
        data, meta, ltype = (lt + (None,))[:3] if len(lt) < 3 else lt
        meta = meta or {}
        if ltype == "labels":
            viewer.add_labels(data, **{k: v for k, v in meta.items() if k in ("name", "scale")})
        else:
            viewer.add_image(data, **{k: v for k, v in meta.items() if k in ("name", "channel_axis", "colormap", "scale")})

    layer_names = [layer.name for layer in viewer.layers]
    print(f"  [napari] viewer has {len(viewer.layers)} layer(s): {layer_names}")

    screenshot_path = zarr_path.parent / f"{scenario.writer_cls.__name__}_{scenario.n_pos}pos.png"
    viewer.screenshot(str(screenshot_path), canvas_only=False)
    print(f"  [napari] screenshot -> {screenshot_path}")

    expected_shapes = _expected_layer_shapes(scenario)
    for layer in viewer.layers:
        if layer.name not in expected_shapes:
            continue
        actual = _layer_shape(layer)
        expected = expected_shapes[layer.name]
        if actual != expected:
            viewer.close()
            raise AssertionError(
                f"{scenario.name}: layer {layer.name!r} shape {actual} "
                f"!= expected {expected}"
            )
        print(f"  [napari] shape OK  {layer.name!r}: {actual}")

    labels_loaded = "labels" in layer_names
    if scenario.skip_napari_labels is None:
        if not labels_loaded:
            viewer.close()
            raise AssertionError(
                f"{scenario.name}: expected a 'labels' layer in napari, "
                f"got {layer_names}"
            )
    elif labels_loaded:
        print(
            f"  [napari] !! UNSKIP: {scenario.name} now surfaces labels in "
            f"napari — remove skip_napari_labels (was: {scenario.skip_napari_labels})"
        )
    else:
        print(f"  [napari] xfail (skipped): {scenario.skip_napari_labels}")

    if keep_open:
        print("  [napari] close the viewer window to continue...")
        napari.run()
    else:
        viewer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="Open each store in a napari viewer after writing.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="With --napari, block on napari.run() between scenarios.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write stores (default: a fresh tempdir).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete the output dir on exit.",
    )
    args = parser.parse_args()

    base = args.output_dir or Path(tempfile.mkdtemp(prefix="faro-writer-smoke-"))
    print(f"output root: {base}")

    names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    try:
        for name in names:
            scenario = SCENARIOS[name]
            scenario_dir = base / name
            zarr_path = run_scenario(scenario, scenario_dir)
            validate(zarr_path, scenario)
            if args.napari:
                open_in_napari(zarr_path, scenario, keep_open=args.interactive)
    finally:
        if not args.keep and args.output_dir is None:
            shutil.rmtree(base, ignore_errors=True)
            print(f"cleaned up {base}")
        else:
            print(f"keeping {base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
