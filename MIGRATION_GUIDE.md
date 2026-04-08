# Migration Guide: FARO restructuring

This guide covers all breaking changes introduced during the FARO restructuring (formerly rtm-pymmcore). It is aimed at lab members who have existing experiment scripts or notebooks.

---

## 1. How to set up and run experiments (new API)

### Old way (DataFrame-based)

```python
from rtm_pymmcore.microscope.Jungfrau import Jungfrau

mic = Jungfrau()
mic.set_pipeline(pipeline)
mic.run_experiment(df_acquire)  # DataFrame with one row per frame
```

The microscope owned the entire experiment lifecycle: it created the Analyzer, Controller, and event loop internally.

### New way (Controller + RTMEvent)

```python
from faro.microscope.pertzlab.jungfrau import Jungfrau
from faro.core.controller import Controller
from faro.core.data_structures import RTMSequence, Channel, PowerChannel

mic = Jungfrau()
pipeline = ImageProcessingPipeline(storage_path=path, ...)
ctrl = Controller(mic, pipeline)

# Define experiment as RTMSequence (preferred) or list[RTMEvent]
seq = RTMSequence(
    time_plan={"interval": 5, "loops": 100},
    stage_positions=[...],
    channels=(Channel(config="phase-contrast", exposure=50),),
    stim_channels=(PowerChannel(config="CyanStim", exposure=200, power=5),),
    stim_frames={10, 11, 12, 13, 14},
)

ctrl.run_experiment(list(seq))
ctrl.finish_experiment()
```

The microscope no longer knows about experiments — it only provides hardware primitives.

### Bridging old DataFrames with `df_to_events`

If you have an existing `df_acquire` DataFrame, convert it to events:

```python
from faro.core.conversion import df_to_events

events = df_to_events(df_acquire)
ctrl.run_experiment(events)
```

This lets you keep using helper functions like `generate_df_acquire()` and `apply_stim_treatments_to_df_acquire()` while running on the new Controller.

### Multi-phase experiments

The new Controller supports continuing experiments with preserved tracking state:

```python
ctrl.run_experiment(phase1_events)        # baseline
ctrl.continue_experiment(phase2_events)   # stimulation
ctrl.continue_experiment(phase3_events)   # recovery
ctrl.finish_experiment()
```

### Acquisition DataFrame backwards compatibility

The output format is unchanged — the pipeline still writes per-FOV `.parquet` files in the `tracks/` folder. You can also convert events to a DataFrame for inspection:

```python
from faro.core.utils import events_to_dataframe
df = events_to_dataframe(events)
```

---

## 2. What changed for pipeline components

### Stimulation class hierarchy

The old single `Stim` base class with a catch-all signature was replaced by a 3-tier hierarchy:

| Class | What it needs | Example |
|-------|--------------|---------|
| `Stim` | Only metadata | `StimWholeFOV`, `StimLine` |
| `StimWithImage` | Metadata + raw image | `StimColonyPercentage` |
| `StimWithPipeline` | Segmentation labels (+ optional image/tracks) | `CenterCircle`, `StimPercentageOfCell` |

If your custom stimulator only needs metadata (e.g., `img_shape`), inherit from `Stim`. If it needs the raw image, inherit from `StimWithImage`. If it needs segmentation labels, inherit from `StimWithPipeline`.

### Channel dataclass

```python
# Old
Channel(name="phase-contrast", exposure=50, group="Channel", power=5,
        device_name="Spectra", property_name="Cyan_Level")

# New
Channel(config="phase-contrast", exposure=50)           # no power
PowerChannel(config="CyanStim", exposure=200, power=5)  # with power
```

The `name` field was renamed to `config` (matches useq naming). Hardware-specific fields (`device_name`, `property_name`) were removed — the microscope auto-detects them via `resolve_power()`.

### SegmentationMethod

Segmentators are now passed as `SegmentationMethod` dataclass instances, not dicts:

```python
# Old
segmentators = [{"name": "labels", "class": CellposeSegmentator(), "use_channel": 0}]

# New
from faro.core.data_structures import SegmentationMethod
segmentators = [SegmentationMethod("labels", CellposeSegmentator(), use_channel=0)]
```

---

## 3. Where did everything move?

### Experiment notebooks

All experiment notebooks moved from the repository root into `experiments/<name>/`:

| Old location | New location |
|---|---|
| `00_NoStim.ipynb` | `experiments/no_stim/no_stim.ipynb` |
| `01_full_FOV_stimulation_ERK_new_API.ipynb` | `experiments/erk_full_fov_stim/erk_full_fov_stim.ipynb` |
| `02_CellMigration_Mic.ipynb` | `experiments/cell_migration/cell_migration.ipynb` |
| `02_CellMigration.ipynb` | `experiments/cell_migration_test/cell_migration_test.ipynb` |
| `03_LineStimulation.ipynb` | `experiments/line_stimulation/line_stimulation.ipynb` |
| `99_Manual_Re_analysis.ipynb` | `experiments/reanalysis/reanalysis.ipynb` |
| `Legacy_01_full_FOV_stimulation_ERK_w_ramp_...ipynb` | `experiments/legacy_erk_ramp/legacy_erk_ramp.ipynb` |
| `data_analysis_plotting/01_ERK-KTR_data_analysis.ipynb` | `experiments/data_analysis/data_analysis.ipynb` |
| *(new)* | `experiments/demo/demo.ipynb` |
| *(new)* | `experiments/demo_sim_optogenetic/demo_sim_optogenetic.ipynb` |
| *(new)* | `experiments/erk_full_fov_stim/erk_full_fov_stim_updated.ipynb` |

### Python module import paths

| Old import | New import |
|---|---|
| `rtm_pymmcore.controller` | `faro.core.controller` |
| `rtm_pymmcore.data_structures` | `faro.core.data_structures` |
| `rtm_pymmcore.img_processing_pip` | `faro.core.pipeline` + `faro.core.pipeline_post` |
| `rtm_pymmcore.utils` | `faro.core.utils` |
| `rtm_pymmcore.dmd` | `faro.core.dmd` |
| `rtm_pymmcore.microscope.abstract_microscope` | `faro.microscope.base` |
| `rtm_pymmcore.microscope.Jungfrau` | `faro.microscope.pertzlab.jungfrau` |
| `rtm_pymmcore.microscope.Niesen` | `faro.microscope.pertzlab.niesen` |
| `rtm_pymmcore.microscope.Moench` | `faro.microscope.pertzlab.moench` |
| `rtm_pymmcore.microscope.MMDemo` | `faro.microscope.demo` |
| `rtm_pymmcore.stimulation.base_stimulation` | `faro.stimulation.base` |
| `rtm_pymmcore.stimulation.moving_line` | `faro.stimulation.moving_line_20x` |
| `rtm_pymmcore.segmentation.base_segmentation` | `faro.segmentation.base` |
| `rtm_pymmcore.segmentation.imaging_server` | `faro.segmentation.remote` |
| `rtm_pymmcore.segmentation.imaging_server_legacy` | `faro.segmentation.remote_legacy` |
| `rtm_pymmcore.tracking.abstract_tracker` | `faro.tracking.base` |
| `rtm_pymmcore.feature_extraction.abstract_fe` | `faro.feature_extraction.base` |
| `rtm_pymmcore.feature_extraction.simple_fe` | `faro.feature_extraction.simple` |
| `rtm_pymmcore.feature_extraction.optocheck_fe` | `faro.feature_extraction.ref` (or `optocheck` compat shim) |
| `rtm_pymmcore.feature_extraction.abstract_fe_optocheck` | `faro.feature_extraction.base_ref` (or `base_optocheck` compat shim) |

### Deleted classes / functions

| Deleted | Replacement |
|---|---|
| `Fov(index)` class | `FovState()` — created internally by the Analyzer, not by user code |
| `microscope.run_experiment(df)` | `Controller(mic, pipeline).run_experiment(events)` |
| `microscope.set_pipeline(pipeline)` | Pass pipeline directly to `Controller(mic, pipeline)` |

### Backwards-compatible aliases (still work)

| Old name | Alias for |
|---|---|
| `utils.generate_fov_objects(mic, ...)` | `utils.generate_fov_positions(mic, ...)` |
| `utils.generate_fov_objects_from_list(mic, data)` | `utils.generate_fov_positions_from_list(mic, data)` |
| `feature_extraction.optocheck.OptoCheckFE` | `feature_extraction.ref.RefFE` |
| `feature_extraction.base_optocheck.FeatureExtractorOptoCheck` | `feature_extraction.base_ref.FeatureExtractorRef` |

---

## 4. Optocheck → Reference acquisition rename

"Optocheck" (checking optogenetic tool expression) has been generalized to **"ref"** (reference acquisition). This covers any one-time acquisition whose features are broadcast to all timepoints — e.g., high-resolution images, expression checks, or acquisitions that bleach the sample.

### What changed

| Old name | New name |
|---|---|
| `ImgType.IMG_OPTOCHECK` | `ImgType.IMG_REF` |
| `feature_extractor_optocheck` (pipeline param) | `feature_extractor_ref` |
| `FeatureExtractorOptoCheck` (base class) | `FeatureExtractorRef` |
| `OptoCheckFE` (concrete class) | `RefFE` |
| `optocheck_mean_intensity` (DataFrame column) | `ref_mean_intensity` |
| `optocheck/` (storage folder) | `ref/` |
| `optocheck.py`, `base_optocheck.py` (files) | `ref.py`, `base_ref.py` (old files kept as compat shims) |

### New: ref_channels and ref_frames on RTMSequence

Reference acquisitions can now be defined inline on `RTMSequence`, just like stimulation:

```python
seq = RTMSequence(
    time_plan={"interval": 5, "loops": 50},
    stage_positions=positions,
    channels=[{"config": "miRFP", "exposure": 300}],
    ref_channels=(Channel(config="mCitrine", exposure=600),),
    ref_frames={-1},  # last frame only
)
```

Both `stim_frames` and `ref_frames` now support **negative indexing** (`-1` = last frame, `-2` = second-to-last) and `range()` objects (`range(0, 50, 2)` = every other frame).

Alternatively, use phase concatenation for multi-phase experiments:

```python
experiment = RTMSequence(...)
ref_phase = RTMSequence(
    time_plan={"interval": 0, "loops": 1},
    stage_positions=positions,
    channels=[{"config": "mCitrine", "exposure": 600}],
    rtm_metadata={"img_type": ImgType.IMG_REF},
)
events = experiment + ref_phase
```

### Updating your code

**Pipeline:**
```python
# Old
pipeline = ImageProcessingPipeline(..., feature_extractor_optocheck=OptoCheckFE("labels"))

# New
pipeline = ImageProcessingPipeline(..., feature_extractor_ref=RefFE("labels"))
```

**Custom feature extractors:**
```python
# Old
from rtm_pymmcore.feature_extraction.base_optocheck import FeatureExtractorOptoCheck
class MyFE(FeatureExtractorOptoCheck): ...

# New
from faro.feature_extraction.base_ref import FeatureExtractorRef
class MyFE(FeatureExtractorRef): ...
```

The old import paths (`base_optocheck`, `optocheck`) still work as backwards-compat shims.

---


## 5. Pluggable storage backends (Writer)

Image storage is no longer hardcoded. The new `Writer` protocol lets you choose how acquired data is persisted. Three implementations are included:

| Writer | Format | Best for |
|--------|--------|----------|
| `TiffWriter` | Individual `.tiff` files per frame | Backwards compatibility, simple inspection |
| `OmeZarrWriter` | Single OME-Zarr v0.5 store (5D array) | Multi-position experiments, large datasets |
| `OmeZarrWriterPlate` | OME-Zarr plate/well layout | Spatial mosaic viewing in napari |

### Default behaviour (TiffWriter)

If you don't pass a writer, `TiffWriter` is used automatically — existing scripts work unchanged:

```python
ctrl = Controller(mic, pipeline)          # TiffWriter created internally
ctrl.run_experiment(events)
```

Storage layout (same as before):

```
storage_path/
├── raw/          001_00042.tiff
├── labels/       001_00042.tiff
├── stim_mask/    ...
└── tracks/       *.parquet
```

### Using OmeZarrWriter

Pass an `OmeZarrWriter` to the Controller to stream all positions and channels into a single OME-Zarr v0.5 container:

```python
from faro.core.writers import OmeZarrWriter

writer = OmeZarrWriter(
    storage_path=path,
    dtype="uint16",
    store_stim_images=True,   # store stim readouts as extra channels (default: False)
)

ctrl = Controller(mic, pipeline, writer=writer)
ctrl.run_experiment(events)
ctrl.finish_experiment()      # calls writer.close()
```

The Controller calls `writer.init_stream()` automatically before the first frame, deriving positions, channels, and image dimensions from the events and microscope.

**Constructor options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `storage_path` | *(required)* | Root directory for all outputs |
| `dtype` | `"uint16"` | Pixel dtype for raw data |
| `store_stim_images` | `False` | Store stim readouts as additional channels in the zarr array (zeros for non-stim frames). If False, stim images fall back to TIFF. |
| `n_timepoints` | `None` | Expected number of timepoints (`None` = unbounded) |
| `label_dtype` | `"uint16"` | Dtype for label/mask arrays |
| `raw_chunk_t` | `1` | Temporal chunk size for raw data |
| `raw_shard_t` | `None` | Temporal shard size for raw data (`None` = same as chunk) |
| `label_chunk_t` | `1` | Temporal chunk size for labels (1 = random access) |
| `label_shard_t` | `50` | Temporal shard size for labels |
| `overwrite` | `True` | Overwrite existing zarr store |

Storage layout (bf2raw):

```
storage_path/
├── acquisition.ome.zarr/
│   ├── Pos0/
│   │   ├── 0/                 raw + stim readout (t, c, y, x)
│   │   └── labels/
│   │       ├── labels/        segmentation masks
│   │       ├── stim_mask/     stimulation masks
│   │       └── ...
│   ├── Pos1/ ...
│   ├── OME/                   series index
│   └── zarr.json
├── ref/                       TIFF fallback (different channel count)
└── tracks/                    *.parquet (unchanged)
```

### Using OmeZarrWriterPlate

`OmeZarrWriterPlate` inherits all options from `OmeZarrWriter` but arranges each FOV position as a well in a single-row plate. napari-ome-zarr tiles the positions spatially as a mosaic:

```python
from faro.core.writers import OmeZarrWriterPlate

writer = OmeZarrWriterPlate(storage_path=path)
ctrl = Controller(mic, pipeline, writer=writer)
```

Storage layout (plate):

```
storage_path/
├── acquisition.ome.zarr/
│   ├── zarr.json              plate metadata
│   ├── A/
│   │   ├── 1/                 well for Pos0
│   │   │   ├── zarr.json      well metadata
│   │   │   └── 0/
│   │   │       ├── 0/         raw array (t, c, y, x)
│   │   │       └── labels/
│   │   │           ├── labels/
│   │   │           └── stim_mask/
│   │   ├── 2/                 well for Pos1
│   │   └── ...
├── ref/                       TIFF fallback
└── tracks/                    *.parquet
```

### Image routing

Both OME-Zarr writers route images by folder name:

| Folder | Destination |
|--------|-------------|
| `"raw"` | Primary zarr array (streaming) |
| `"stim"` | Extra channel(s) in raw array (if `store_stim_images=True`), else TIFF fallback |
| `"ref"` | Always TIFF fallback (different channel count than raw) |
| Any other name (`"labels"`, `"stim_mask"`, ...) | NGFF label group, created lazily on first write |

### Implementing a custom writer

Any object that satisfies the `Writer` protocol works:

```python
from faro.core.writers import Writer

class MyWriter:
    storage_path: str

    def write(self, img, metadata, folder): ...
    def save_events(self, events): ...
    def close(self): ...
```

---

## 6. Technical changes (internal)

These changes don't affect experiment scripts but are relevant for contributors:

- **Pipeline/pipeline_post deduplication**: 6 shared code blocks extracted into helper functions (`build_frame_dataframe`, `run_tracking`, `extract_and_merge_features`, `dispatch_stim_mask`, `convert_track_dtypes`, `save_segmentation_results`).
- **Pipeline concurrency improvement**: `tracks_queue.put()` now happens before the parquet save, unblocking the next frame earlier. A per-FOV `parquet_lock` prevents concurrent writes.
- **Shutdown race condition fixed**: Worker threads now drain their queues before exiting.
- **Controller no longer reaches into Analyzer internals**: Clean separation of concerns.
- **AbstractMicroscope defines MDA interface**: `run_mda()`, `connect_frame()`, `disconnect_frame()`, `cancel_mda()` — testable without pymmcore.
- **New intermediate class `PyMMCoreMicroscope`**: Bridges AbstractMicroscope → pymmcore-plus. Lab microscopes (Jungfrau, Niesen, Moench) inherit from this.
- **Validation system**: `pipeline.validate_events(events)` checks method signatures and required metadata. `mic.validate_hardware(events)` checks hardware capabilities.
- **Test suite**: 152 tests covering pipeline, tracking, stimulation, event ordering, reference acquisition, negative indexing, crash resilience, burst dispatch, and experiment continuation.
