"""Execute the example notebooks and the templates on the virtual microscope.

Templates ship with working virtual-scope defaults, so they run verbatim
in a real kernel. Only TODO cells listed in ``FILLERS`` (those that end in
``raise NotImplementedError`` because no default makes sense, such as a path
to data on disk) are swapped for a filler keyed by their ``todo:<name>`` tag.
Cells tagged ``gui`` (napari) are dropped, and cells tagged ``parameters``
get small frame counts appended so the whole file stays fast.

Requires the ``test`` extra.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("vmteach")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TEMPLATES = ROOT / "templates"
README = ROOT / "README.md"

SKIP_TAGS = {"gui"}
TODO_PREFIX = "todo:"
TIMEOUT_S = 600

# Fillers for TODO cells that have no runnable default. A filler without a
# matching tag, or a raise-cell without a filler, fails the structure tests.
FILLERS: dict[str, dict[str, str]] = {
    "live_experiment/experiment.ipynb": {},
    "reanalysis/reanalysis.ipynb": {
        # SRC_PATH is injected by the test after it has produced a source run.
        "todo:paths": """
SRC_PATH = os.environ["FARO_TEST_SRC_PATH"]
OUT_PATH = SRC_PATH + "_reanalysis"
""",
    },
}

PARAMS = {
    "live_experiment.ipynb": """
INTERVAL_S = 0.3
N_BASELINE = 2
N_STIM = 4
N_RECOVERY = 2
N_EXTRA = 2
STIM_FRACTION = 0.2
""",
    "live_experiment/experiment.ipynb": """
INTERVAL_S = 0.3
N_BASELINE = 2
N_STIM = 3
N_RECOVERY = 2
""",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tags(cell) -> list[str]:
    return list(cell.get("metadata", {}).get("tags", []))


def _todo_tag(cell) -> str | None:
    for tag in _tags(cell):
        if tag.startswith(TODO_PREFIX):
            return tag
    return None


def _prepare(nb_path: Path, fillers: dict[str, str], params: str | None):
    """Return a copy of the notebook with TODO cells filled and gui cells dropped."""
    nb = nbformat.read(nb_path, as_version=4)
    cells = []
    seen: set[str] = set()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        tags = set(_tags(cell))
        if tags & SKIP_TAGS:
            continue
        todo = _todo_tag(cell)
        if todo is not None and todo in fillers:
            seen.add(todo)
            cell.source = fillers[todo]
        elif "parameters" in tags and params is not None:
            cell.source = cell.source + "\n" + params
        # magics do not survive a plain kernel run without IPython extensions
        cell.source = "\n".join(
            line for line in cell.source.splitlines() if not line.lstrip().startswith("%")
        )
        cells.append(cell)
    missing = set(fillers) - seen
    assert not missing, f"{nb_path.name}: fillers without a TODO cell: {sorted(missing)}"
    nb.cells = cells
    return nb


def _execute(nb, cwd: Path) -> None:
    env_backup = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        client = nbclient.NotebookClient(
            nb,
            timeout=TIMEOUT_S,
            kernel_name="python3",
            resources={"metadata": {"path": str(cwd)}},
        )
        # nbclient's DEBUG records do not survive pytest's log capture
        client.log.setLevel(logging.WARNING)
        client.execute()
    finally:
        if env_backup is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = env_backup


def _readme_anchors() -> set[str]:
    anchors = set()
    for line in README.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if not m:
            continue
        title = re.sub(r"`", "", m.group(2))
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        anchors.add(slug)
    return anchors


def _notebook_readme_links(nb_path: Path) -> set[str]:
    text = nb_path.read_text(encoding="utf-8")
    return set(re.findall(r"README\.md#([\w\-]+)", text))


def _all_notebooks() -> list[Path]:
    return sorted(EXAMPLES.rglob("*.ipynb")) + sorted(TEMPLATES.rglob("*.ipynb"))


# ---------------------------------------------------------------------------
# structure checks (fast, no kernel)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nb_path", _all_notebooks(), ids=lambda p: p.name)
def test_readme_anchors_exist(nb_path: Path):
    anchors = _readme_anchors()
    missing = _notebook_readme_links(nb_path) - anchors
    assert not missing, f"{nb_path.name} links to README sections that do not exist: {sorted(missing)}"


@pytest.mark.parametrize("rel", sorted(FILLERS), ids=lambda r: r.split("/")[0])
def test_template_raise_cells_match_fillers(rel: str):
    """Every cell that raises NotImplementedError has a filler, and vice versa."""
    nb = nbformat.read(TEMPLATES / rel, as_version=4)
    raising = {
        _todo_tag(c)
        for c in nb.cells
        if c.cell_type == "code" and "raise NotImplementedError" in c.source
    }
    assert None not in raising, f"{rel}: a raising cell is missing its todo:<name> tag"
    assert raising == set(FILLERS[rel]), f"{rel}: raise cells {raising} != fillers {set(FILLERS[rel])}"


def test_template_folders_are_complete():
    assert (TEMPLATES / "README.md").exists(), "templates/README.md is missing"
    for folder in TEMPLATES.iterdir():
        if not folder.is_dir():
            continue
        assert (folder / "pyproject.toml").exists(), f"{folder.name}: missing pyproject.toml"
        assert list(folder.glob("*.ipynb")), f"{folder.name}: missing notebook"


# ---------------------------------------------------------------------------
# execution (kernel, virtual microscope)
# ---------------------------------------------------------------------------
@pytest.mark.examples
@pytest.mark.parametrize("name", ["live_experiment.ipynb"])
def test_example_runs(name: str):
    nb_path = EXAMPLES / name
    nb = _prepare(nb_path, fillers={}, params=PARAMS.get(name))
    _execute(nb, cwd=EXAMPLES)


@pytest.mark.examples
def test_live_template_runs():
    rel = "live_experiment/experiment.ipynb"
    nb = _prepare(TEMPLATES / rel, fillers=FILLERS[rel], params=PARAMS.get(rel))
    _execute(nb, cwd=(TEMPLATES / rel).parent)


@pytest.mark.examples
def test_reanalysis_template_runs(tmp_dir):
    """Produce a small source run on the virtual scope, then re-analyse it."""
    src = _make_source_run(Path(tmp_dir) / "source")
    rel = "reanalysis/reanalysis.ipynb"
    nb = _prepare(TEMPLATES / rel, fillers=FILLERS[rel], params=None)
    previous = os.environ.get("FARO_TEST_SRC_PATH")
    os.environ["FARO_TEST_SRC_PATH"] = str(src)
    try:
        _execute(nb, cwd=(TEMPLATES / rel).parent)
    finally:
        if previous is None:
            os.environ.pop("FARO_TEST_SRC_PATH", None)
        else:
            os.environ["FARO_TEST_SRC_PATH"] = previous
    out = Path(str(src) + "_reanalysis")
    assert (out / "exp_data.parquet").exists()


def _make_source_run(path: Path) -> Path:
    from vmteach import load_microscope

    from faro.core.controller import Controller
    from faro.core.data_structures import RTMSequence, SegmentationMethod
    from faro.core.pipeline import ImageProcessingPipeline
    from faro.core.writers import OmeZarrWriter
    from faro.feature_extraction.simple import SimpleFE
    from faro.microscope.simulation import UniMMCoreSimulation
    from faro.segmentation.base import OtsuSegmentator
    from faro.tracking.trackpy import TrackerTrackpy

    core, _ = load_microscope("optogenetic", mode="realtime")
    mic = UniMMCoreSimulation(mmc=core)
    mic.init_scope()
    path = Path(path)
    pipeline = ImageProcessingPipeline(
        storage_path=str(path),
        segmentators=[
            SegmentationMethod("labels", OtsuSegmentator(), use_channel=0, save_tracked=True)
        ],
        feature_extractor=SimpleFE("labels"),
        tracker=TrackerTrackpy(search_range=30),
    )
    events = list(
        RTMSequence(
            time_plan={"interval": 0.3, "loops": 3},
            stage_positions=[{"x": 0.0, "y": 0.0, "z": 0.0}],
            channels=[{"config": "miRFP", "exposure": 50}],
        )
    )
    ctrl = Controller(mic, pipeline, writer=OmeZarrWriter(storage_path=str(path)))
    ctrl.run_experiment(events).wait()
    ctrl.finish_experiment()
    assert (path / "events.json").exists()
    return path
