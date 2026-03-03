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
)
```
## Controller

The Controller converts RTMEvents to MDAEvents, queues them through the microscope, and dispatches frames to the pipeline.

### Experiment Definition

Experiments are defined as `RTMSequence` objects — an extension of useq's `MDASequence` with stimulation and optocheck support. Multiple phases can be concatenated with `+`:

```python
from rtm_pymmcore.core.data_structures import Channel, PowerChannel, RTMSequence

stim = PowerChannel("CyanStim", exposure=100, group="TTL_ERK", power=10)

phase_1 = RTMSequence(
    time_plan={"interval": 60.0, "loops": 100},
    stage_positions=fov_positions,
    channels=[{"config": "miRFP", "exposure": 300}],
    stim_channels=(stim,),
    stim_frames=range(10, 100),
)

phase_2 = RTMSequence(
    time_plan={"interval": 60.0, "loops": 150},
    stage_positions=fov_positions,
    channels=[{"config": "miRFP", "exposure": 300}],
    stim_channels=(stim,),
    stim_frames=range(30, 120),
)

events = phase_1 + phase_2
```

### Running

```python
from rtm_pymmcore.core.controller import Controller

ctrl = Controller(mic, pipeline)
ctrl.run_experiment(events, stim_mode="current")
```

`validate_events()` runs automatically before the experiment starts (disable with `validate=False`). It checks both pipeline compatibility and hardware limits.

**Stimulation modes:**
* `"current"`: acquire frame, wait for segmentation mask, then stimulate in the same timepoint
* `"previous"`: stimulate using the mask from the previous timepoint, then acquire

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
