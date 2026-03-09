# rtm-pymmcore

**Real-time feedback control microscopy.**

rtm-pymmcore acquires images, segments cells, extracts features, tracks them over time, and generates stimulation masks, all while the experiment is running. This enables closed-loop feedback control: stimulation patterns can be computed from the latest segmentation and applied within the same or next timepoint.

## Architecture

```
Pipeline   ◀──▶  Controller   ◀──▶   Microscope
────────         ──────────          ──────────
- segment        - orchestrate       - stage
- track            experiment        - camera
- extract                            - DMD/SLM
  features                           - live cells
- stim mask                           
```
**Microscope**: hardware interface. Any microscope that speaks implements [useq-schema](https://github.com/pymmcore-plus/useq-schema) can be used. Works great with µManger / [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus).

**Pipeline**: modular image processing. Performs segmentation, tracking, feature extraction. Decides if/where to photoactivate the sample.

**Controller**: experiment orchestrator. Queues acquisition events to the microscope, dispatches frames to the pipeline, and coordinates stimulation timing.

## Quickstart

Try **`experiments/demo_sim_optogenetic/`** notebook to run a complete optogenetic feedback experiment on a simulated microscope, no hardware required.

```python
# 1. Set microscope
mic = UniMMCoreSimulation(mmc=mmc)
mic.init_scope()

# 2. Assemble image processing pipeline
pipeline = ImageProcessingPipeline(
    storage_path="/path/to/experiment",
    segmentators=[SegmentationMethod("labels", OtsuSegmentator(), use_channel=0, save_tracked=True)],
    feature_extractor=SimpleFE("labels"),
    tracker=TrackerTrackpy(),
    stimulator=MoveUp(),
)

# 3. Define experiment parameters
events = RTMSequence(
    time_plan={"interval": 5.0, "loops": 20},
    stage_positions=[{"x": 256, "y": 256}],
    channels=[{"config": "BF", "exposure": 50}],
    stim_channels=[{"config": "Cyan", "exposure": 50}],
    stim_frames=range(5, 20),
)

# 4. Run!
ctrl = Controller(mic, pipeline)
ctrl.run_experiment(list(events), stim_mode="current")
```

## Pipeline

The pipeline is modular, each component is independent and can be swapped or set to `None`.

| Component | Purpose | Examples |
|-----------|---------|----------|
| **Segmentation** | Identify cells in images | `OtsuSegmentator`, `SegmentorCellpose`, `SegmentatorStardist`, remote via [imaging-server-kit](https://github.com/imaging-server-kit) |
| **Stimulation** | Generate masks for DMD/SLM | `StimWholeFOV`, `StimPercentageOfCell`, `CenterCircle`, `StimLine` |
| **Feature extraction** | Measure cell properties | `SimpleFE` (position, area), `FE_ErkKtr` (ERK-KTR c/n ratio) |
| **Tracking** | Link cells across frames | `TrackerTrackpy` (via [trackpy](https://github.com/soft-matter/trackpy)) |

```python
pipeline = ImageProcessingPipeline(
    storage_path="/path/to/experiment",
    segmentators=segmentators,  # list of SegmentationMethod
    feature_extractor=fe,
    tracker=tracker,
    stimulator=stimulator,
    feature_extractor_ref=ref_fe,  # optional: for reference acquisition frames
)
```
## Controller

The Controller converts RTMEvents to MDAEvents, queues them through the microscope, and dispatches frames to the pipeline.

### Experiment Definition

Experiments are defined as `RTMSequence` objects — an extension of useq's `MDASequence`. Multiple phases can be concatenated with `+`:

```python
from rtm_pymmcore.core.data_structures import Channel, PowerChannel, RTMSequence

phase_1 = RTMSequence(
    time_plan={"interval": 60.0, "loops": 100},
    stage_positions=fov_positions,
    channels=[{"config": "miRFP", "exposure": 300}],
)

phase_2 = RTMSequence(
    time_plan={"interval": 60.0, "loops": 150},
    stage_positions=fov_positions,
    channels=[{"config": "miRFP", "exposure": 300}],
)

events = phase_1 + phase_2
```

### Stimulation

Stimulation channels are acquired on specific frames, controlled via DMD/SLM. Define them with `stim_channels` and `stim_frames`:

```python
seq = RTMSequence(
    time_plan={"interval": 5.0, "loops": 50},
    stage_positions=fov_positions,
    channels=[{"config": "miRFP", "exposure": 300}],
    stim_channels=(PowerChannel(config="CyanStim", exposure=200, power=10),),
    stim_frames=range(10, 50),
)
```

**Stimulation modes** (set via `ctrl.run_experiment(events, stim_mode=...)`):
* `"current"`: acquire frame, wait for segmentation mask, then stimulate in the same timepoint
* `"previous"`: stimulate using the mask from the previous timepoint, then acquire

### Reference Acquisition

Reference channels are acquired on specific frames for one-time measurements whose features are broadcast to all timepoints — e.g., checking expression of an optogenetic tool, or a high-resolution image that would bleach the sample. Define them with `ref_channels` and `ref_frames`:

```python
seq = RTMSequence(
    time_plan={"interval": 5.0, "loops": 50},
    stage_positions=fov_positions,
    channels=[{"config": "miRFP", "exposure": 300}],
    ref_channels=(Channel(config="mCitrine", exposure=600),),
    ref_frames={-1},  # last frame only
)
```

Alternatively, define the reference as a separate phase:

```python
experiment = RTMSequence(time_plan=..., channels=..., ...)
ref_phase  = RTMSequence(
    time_plan={"interval": 0, "loops": 1},
    stage_positions=fov_positions,
    channels=[{"config": "mCitrine", "exposure": 600}],
    rtm_metadata={"img_type": ImgType.IMG_REF},
)
events = experiment + ref_phase
```

### Frame Specification

Both `stim_frames` and `ref_frames` accept:
* **Sets**: `{0, 5, 10}` — specific frames
* **Ranges**: `range(10, 50)` or `range(0, 50, 2)` — contiguous or strided
* **Negative indices**: `-1` = last frame, `-2` = second-to-last

### Axis Order

`axis_order` controls the nesting of time, position, and channel dimensions (inherited from useq's `MDASequence`). The default is `"tpcz"`:

| `axis_order` | Iteration | Use case |
|---|---|---|
| `"tpcz"` (default) | All positions at t=0, then all at t=1, ... | Maximize temporal resolution per position |
| `"ptcz"` | All timepoints at p=0, then all at p=1, ... | Complete one position before moving to the next |

```python
# Visit all 3 positions at each timepoint before advancing
seq = RTMSequence(
    time_plan={"interval": 5.0, "loops": 50},
    stage_positions=[(0, 0, 0), (100, 100, 0), (200, 200, 0)],
    channels=[{"config": "BF", "exposure": 50}],
    axis_order="tpcz",  # default: (t=0,p=0), (t=0,p=1), (t=0,p=2), (t=1,p=0), ...
)

# Complete all timepoints at each position before moving on
seq = RTMSequence(
    ...,
    axis_order="ptcz",  # (t=0,p=0), (t=1,p=0), ..., (t=49,p=0), (t=0,p=1), ...
)
```

Stimulation and reference channels are assigned per-timepoint, so they work correctly regardless of axis order. For example, `stim_frames={3}` stimulates all positions at t=3, whether they are visited consecutively (`tpcz`) or spread across the run (`ptcz`).

### Running

```python
from rtm_pymmcore.core.controller import Controller

ctrl = Controller(mic, pipeline)
ctrl.run_experiment(events, stim_mode="current")
```

`validate_events()` runs automatically before the experiment starts (disable with `validate=False`). It checks both pipeline compatibility and hardware limits.

### Experiment Continuation

Call `run_experiment()` once, then `continue_experiment()` to append more phases. The Analyzer (and all per-FOV tracking state) is reused, so timesteps, filenames, and particle IDs continue seamlessly.

```python
ctrl = Controller(mic, pipeline)

# Phase 1: baseline — find cells, measure growth rate
phase1 = RTMSequence(time_plan={"interval": 10, "loops": 60}, ...)
ctrl.run_experiment(phase1, validate=False)

# Analyse phase-1 results to decide what to do next
df = pd.read_parquet("tracks/000_latest.parquet")
fast_growers = df.groupby("particle")["area"].apply(lambda x: x.diff().mean())

# Phase 2: stimulate based on analysis
phase2 = RTMSequence(time_plan={"interval": 10, "loops": 120}, ...)
ctrl.continue_experiment(phase2)

# Always call finish_experiment() when done
ctrl.finish_experiment()
```

To add events while an experiment is still running, use `extend_experiment()`:

```python
ctrl.run_experiment(baseline_events, validate=False)  # runs in background thread
ctrl.extend_experiment(extra_events)                   # non-blocking, appends to running acquisition
```

| Method | When to use |
|--------|-------------|
| `run_experiment()` | First acquisition — creates a fresh Analyzer |
| `continue_experiment()` | Subsequent phases — reuses Analyzer, offsets timesteps |
| `extend_experiment()` | Mid-run additions — pushes events into the running loop |
| `finish_experiment()` | Cleanup — shuts down Analyzer, resets state |

## Microscope

The microscope provides the hardware interface. Any microscope that implements the useq-schema MDA protocol can be used, the Controller never depends on pymmcore-plus directly.

### Class Hierarchy

```
AbstractMicroscope                # useq MDA interface
  ├─ PyMMCoreMicroscope           # implements via pymmcore-plus / CMMCorePlus
  │    ├─ MMDemo                  # Micro-Manager demo hardware
  │    ├─ UniMMCoreSimulation     # simulated microscope
  │    ├─ PymmcoreProxyMic        # remote via pymmcore-proxy
  │    └─ pertzlab/
  │         ├─ Jungfrau
  │         ├─ Moench
  │         └─ Niesen
  └─ InscoperMicroscope           # implements via Inscoper SDK (planned)
```

### Interface

| Method | Purpose |
|--------|---------|
| `run_mda(event_iter)` | Start MDA acquisition, returns thread handle |
| `connect_frame(callback)` | Connect frameReady: `callback(img, event)` |
| `disconnect_frame(callback)` | Disconnect frameReady |
| `cancel_mda()` | Cancel running MDA |
| `resolve_group(config_name)` | Return channel group for a config name (optional) |
| `resolve_power(channel)` | Return `(device, property, power)` (optional) |
| `validate_hardware(events)` | Check events against hardware limits (optional) |
| `init_scope()` | Load config, set up hardware |
| `post_experiment()` | Cleanup after experiment |

`PyMMCoreMicroscope` implements the MDA methods via `CMMCorePlus`. Concrete subclasses typically only need `init_scope()`.

## Micro-Manager / pymmcore-plus

The `PyMMCoreMicroscope` branch uses [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) as its hardware layer. Each microscope needs a **Micro-Manager configuration file** with:

* Channel presets for each fluorophore (e.g., `GFP`, `mCherry`, `miRFP`)
* A `System > Startup` preset for initial hardware configuration
* Device properties for cameras, light sources, filter wheels, etc.

For microscopes with controllable light source power, define a `POWER_PROPERTIES` mapping so `PowerChannel` objects resolve to the correct device:

```python
POWER_PROPERTIES = {
    "CyanStim": ("Spectra", "Cyan_Level"),  # config_name → (device, property)
}
```

## Adding Your Own µManager Microscope

Create a new file in `rtm_pymmcore/microscope/` and inherit from `PyMMCoreMicroscope`:

```python
import pymmcore_plus
from rtm_pymmcore.microscope.pymmcore import PyMMCoreMicroscope

class MyScope(PyMMCoreMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    MICROMANAGER_CONFIG = "path/to/config.cfg"
    CHANNEL_GROUP = "Channel"

    def __init__(self):
        super().__init__()
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()
        self.init_scope()

    def init_scope(self):
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setChannelGroup(channelGroup=self.CHANNEL_GROUP)

    def post_experiment(self):
        pass  # optional cleanup
```

For DMD support, set up `self.dmd` in `__init__()`, see `pertzlab/moench.py` for an example.

## Installation

```bash
git clone https://github.com/pertzlab/rtm-pymmcore.git
cd rtm-pymmcore
pip install -r requirements.txt
```

## Contributing

Contributions are welcome. Please submit pull requests or open issues.

## License

MIT License. See `LICENSE` for details.
