# Templates

Copy-and-fill notebooks for real work. Each folder is a complete uv project: copy it into your experiments repository, rename it, run `uv sync`, and work through the cells marked **TODO**. The live experiment template runs end to end on the virtual microscope as shipped, so run it once to check your environment, then replace the defaults in the TODO cells; each lists the real-scope alternatives as comments. The re-analysis template only needs its paths cell filled in. Everything else works as it is.

Each folder also has a `custom.py` for the segmentator, feature extractor or stimulator that only this experiment needs. Write them there rather than in notebook cells, so the re-analysis notebook can import the same classes and they can be tested; the pipeline cell shows the import. When a class is needed by a second experiment, move it into faro and pin the commit that has it.

| Folder | Use it for |
|--------|------------|
| `live_experiment/` | A feedback or plain timelapse experiment on a real microscope. Same structure as the [live experiment example](../examples/live_experiment.ipynb), which runs on a virtual microscope and explains every step. |
| `reanalysis/` | Re-processing an experiment already on disk with new segmentation, tracking or features. No microscope needed. Raw images are never rewritten; the output folder gets hard links to the raw data plus fresh labels, tracks and `exp_data.parquet`. See [Re-analysis](../README.md#re-analysis) in the faro README. |

## Start a new experiment

1. Copy the template folder into your experiments repository (in the lab: [faro-experiments](https://github.com/pertzlab/faro-experiments)) and rename it, for example `2026-01-01_erk_pulses`.
2. In `pyproject.toml`, set `name` and pin faro (see below).
3. Open a terminal in the folder and run `uv sync`.
4. Open the notebook, select the `.venv` kernel, run it once as is, then fill in the TODO cells.

`uv sync` creates a `.venv` folder next to the notebook with faro and all its dependencies. Never `pip install` into it by hand; use the commands below so `pyproject.toml` and `uv.lock` stay in sync.

## uv in five commands

| Task | Command |
|------|---------|
| Create or update the environment | `uv sync` |
| Start Jupyter in this environment | `uv run --no-sync --with jupyterlab jupyter lab` |
| Run a script in this environment | `uv run --no-sync python my_script.py` |
| Add a package | `uv add <name>` |
| Re-resolve faro after changing its pin | `uv lock --upgrade-package faro` then `uv sync` |

In VS Code, pick the interpreter `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux, macOS) as the notebook kernel.

## Pinning faro

`pyproject.toml` starts with faro on `branch = "main"`, so every `uv sync` may pull newer code. Before the experiment goes to the scope, pin it to the exact faro commit you tested with:

1. Find the commit: `git -C <path to faro checkout> rev-parse HEAD`.
2. In `pyproject.toml`, replace the `branch = "main"` line with the `rev = "..."` line and paste the commit hash.
3. Run `uv lock --upgrade-package faro` and `uv sync`.
4. Commit `pyproject.toml` and `uv.lock` together.

`uv.lock` records the exact version of every package, not only faro. With it in the repository, `uv sync` rebuilds the same environment on any machine.

## Moving to a newer faro

Change the `rev` hash, run `uv lock --upgrade-package faro` and `uv sync`, rerun the notebook, and fix what changed. Commit the two files once it works. Older experiments keep their own pins and are not affected.

## Working on faro itself

To run an experiment against a local faro checkout, switch the source in `pyproject.toml` to the `path = ...` line (adjust the relative path), then `uv sync`. Code edits in the checkout are picked up immediately. Switch back to a `rev` pin and re-lock before the experiment goes dormant, so it stays reproducible.

## Real microscopes: lab packages

The microscope classes live in lab packages ([faro-pertzlab](https://github.com/pertzlab/faro-pertzlab), [faro-aoki](https://github.com/alandolt/faro-aoki)), not in faro. Add yours to `dependencies` (the commented lines in `pyproject.toml`); it brings faro along. When the experiment's faro source differs from the lab package's (a `rev` pin or a local clone), uncomment `override-dependencies = ["faro"]` too: the lab package pins faro to git main, and without the override uv stops with "conflicting URLs for package `faro`". The override replaces every faro requirement, extras included, so list your faro extras in it (`["faro[cellpose]"]`); otherwise they are dropped silently. See [Lab packages](../README.md#lab-packages) in the faro README.

## When things look stale

`uv sync` again. If faro was updated in place, `uv sync --reinstall-package faro`.
