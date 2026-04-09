# TODO

Follow-ups from the Moench hardware test sweep (2026-04-09). The two
hardware smoke tests (`tests/hardware/test_line_stimulation.py` and
`tests/hardware/test_cell_migration.py`) now pass cleanly on Moench:

| Test                          | Duration | Zombies | BG errors |
| ----------------------------- | -------- | ------- | --------- |
| `test_line_stimulation_smoke` | 52.2 s   | 0       | 0         |
| `test_cell_migration_smoke`   | 280.2 s  | 0       | 0         |

The items below are issues the hardware run surfaced but that we
haven't fixed yet.

---

## 1. Exclude `Mosaic3` from `waitForDevice` (performance / Moench)

**Symptom.** Every MDA event on Moench logs:

```
[WARN] waitForDevice(Mosaic3) timed out (Wait for device "Mosaic3" timed out after 5000ms), continuing.
```

That's 5 s wasted per event. The cell-migration test has ~12 imaging
events → ~60 s of pure wait cost, and a 24 h experiment pays this tax
on every frame.

**Root cause.** `Mosaic3` (the DMD) has a perpetually-stuck `Busy()`
flag, same pathology as `TIXYDrive`. `MoenchMDAEngine._wait_for_system_excluding_xy`
(`faro/microscope/pertzlab/moench.py:325-376`) already knows how to
skip `TIXYDrive` — `Mosaic3` should be in the same skip list, or the
exclude set should be a class attribute so it's easy to extend.

**Fix sketch.** In `_wait_for_system_excluding_xy`, add
`"Mosaic3"` (or better: a `SKIP_WAIT_DEVICES` class tuple on `Moench`)
to the set of devices the loop skips. The DMD image has already been
committed by `setSLMImage` / `displaySLMImage` by the time we're
waiting, so skipping its `Busy()` poll is safe.

---

## 2. First stim frame silently fires with no mask (data quality)

**Symptom.** On the first stim timestep of `test_cell_migration_smoke`:

```
Warning: Stimulation mask not ready (timeout):
Warning: Stimulation mask unavailable, sending False to SLM.
```

The stim event fires but the SLM gets `False` (no pattern), so **no
actual stimulation happens on that frame** — yet the experiment keeps
going as if it had stimulated. This is the worst kind of silent bug
for a 24 h feedback experiment: you'd believe stim was on from frame
1 and only discover the gap during analysis.

**Status.** The *silent* half is fixed: `Analyzer.get_stim_mask` now
records a `stim_mask` background error on timeout, so hardware tests
assert on it and fail loudly. The stim event still sends `False` to
the SLM (log-and-continue, same as other background failures), but
at least the failure is visible.

**Root cause is UNVERIFIED.** Earlier notes claimed "cellpose GPU
warmup > 80 s" but nobody has actually measured it. Plausible
alternatives:

- **Cellpose-SAM first-inference GPU cold start.** The v4 default
  model is `cpsam`, ~1.2 GB (vs. ~25 MB for older cellpose models).
  First-inference cold start with cuDNN autotuning on a SAM-scale
  graph can be 30–90 s.
- **Weights download on first-ever run.** Cellpose fetches to
  `~/.cellpose/models/` lazily. If that dir is empty, 1.2 GB at
  lab-WiFi speeds is 60+ s *once*, then never again. If this is
  the cause, the "fix" is a one-time pre-download, not a code change.
- **A pipeline task raising an exception that's eaten** before
  `stim_mask_queue.put()` runs. (`_pipeline_task_done` now records
  to `background_errors`, so this case should already be visible —
  if the test reports a `pipeline` error alongside the `stim_mask`
  error, this is the cause.)
- **Deadlock** between the storage worker and the pipeline executor.
- **The first raw imaging frame never making it through**
  `Controller._frame_buffers` (wrong channel count, stale buffer,
  buffer-shape-mismatch path triggered, etc.).

### What to measure on the next Moench run

Don't add more speculation — instrument and collect data. All of
these should be gated behind `analyzer.debug` so normal runs aren't
affected.

1. **`CellposeV4.__init__` wall time.** Time the
   `models.CellposeModel(gpu=gpu)` construction separately from
   first inference. Tells us whether the problem is model loading
   or inference.
2. **`CellposeV4.segment` first N calls.** Keep a per-instance call
   counter and print `[timing] cellpose.segment call=N t=X.Xs` for
   the first 3 calls. If call 1 is 70 s and call 2 is 0.3 s, it's
   first-inference warmup.
3. **Executor-queue lag.** In `_try_submit_pipeline`, print the
   wall-clock gap between `executor.submit(...)` and the first line
   of `pipeline.run` (add a timestamp param, or instrument
   `pipeline.run` directly). Rules out "executor queue backlog".
4. **`stim_mask_queue.put()` timing.** Print when the first stim
   mask actually gets put. The gap between the first
   `_on_frame_ready` and the corresponding `queue.put` is the
   actionable number.
5. **Confirm weights cache state before the run.** `ls -lh ~/.cellpose/models/`
   on the Moench PC before starting the test. If `cpsam` is missing
   or zero-bytes, the first run is a download, not inference.

### Pick a fix based on the measurement

- **If first-inference warmup:** add a `pipeline.prewarm()` hook
  that runs one `segment(np.zeros((256, 256), dtype=np.uint16))`
  inside `Controller.run_experiment` *before* `run_mda`. Consider
  running it on a thread that fans out in parallel with MDA startup
  so it doesn't add serial wall-time.
- **If weights download:** document "pre-download cellpose weights
  before running hardware tests" in the `tests/hardware/conftest.py`
  docstring, or add a session-scope fixture that instantiates
  `CellposeModel(gpu=True)` once to force the download.
- **If executor queue lag or deadlock:** fix the actual bug —
  pre-warming won't help.
- **If a pipeline exception is being eaten:** find the missing
  `_record_background_error` callsite and add it.

### Make the timeout configurable (low priority, independent)

80 s is hardcoded in `Analyzer.get_stim_mask`. Pipelines with slow
first-frame segmenters (cellpose SAM, stardist remote, convpaint)
need more. Add it as a constructor arg on `Analyzer`, or let
`Controller` pass it through from `run_experiment`. Not urgent now
that the background-error recording surfaces the timeout either way.

---

## 3. Cell migration test budget (test infra)

**Current.** `test_cell_migration_smoke` runs in **280 s**. The test
docstring claims "about a minute". User target is **≤ 3 min**.

**What drives the time.**

- `waitForDevice(Mosaic3)` ~60 s (see #1)
- First-frame stim mask timeout ~80 s (see #2)
- Cellpose GPU model load ~20–30 s
- 4 frames × 5 s time plan = 20 s
- Rest: imaging, stage, ref frames, teardown

Fixing #1 + #2 should drop this to ~90–120 s, comfortably inside 3
min. Until then, consider marking the test as `@pytest.mark.slow` so
it isn't blocking on the 3-min budget.

---

## 4. Binning is a cfg group, not a channel property (convention)

This isn't a bug — it's a convention the hardware sweep established
and that future changes should respect.

**Rule.** Camera binning is a **global acquisition-wide setting**.
The zarr writer allocates one `(y, x)` shape per store from
`mmc.getImageHeight() / getImageWidth()` at `init_stream` time, so
any mid-run binning change corrupts every frame downstream.

**Where binning lives on Moench.** `TiMoench.cfg` has a dedicated
`Binning` config group with `1x1` / `2x2` / `4x4` presets. Callers
pick one with:

```python
mic.mmc.setConfig("Binning", "2x2")
```

**Where it must NOT live.** Channel presets in `TTL_ERK` (or any
other light-path group) must not set `Camera,Binning,*`. Previously
`mCitrine` carried `Camera,Binning,2x2` and no other channel reset
it, so one mCitrine capture stuck the camera at 2×2 for the rest of
the run. This caused the cascade of broadcast errors that kicked off
the whole sweep.

**Test fixture default.** `tests/hardware/conftest.py` applies
`Binning,2x2` *before* `set_roi()` — most MM configs reset the ROI
on a binning change, so binning has to happen first. The fixture
then re-asserts the scope's ROI on top.

**Notebooks.** `experiments/22_line_stimulation/line_stimulation.ipynb`
and `experiments/21_cell_migration/cell_migration.ipynb` both call
`mic.mmc.setConfig("Binning", "2x2")` in the ROI cell — update the
argument (or add a new cell earlier) when you want a different
binning for that experiment.

If you add a new Pertzlab microscope cfg, mirror this structure:
keep `Binning` as its own group, and never let a channel preset
touch camera geometry properties (Binning, ROI, PixelSize).

---

## 5. Wishlist / smaller items

- Moench `post_experiment()` currently *starts* the DMD wakeup thread
  rather than stopping it. That's fine between back-to-back
  experiments, but is a confusing name. Consider
  `prepare_for_next_experiment()` or split into two methods.
- `_record_background_error` takes an exception and reads
  `traceback.format_exc()` — which only works if it's called from an
  active `except` block. The storage worker and deferred worker
  already guarantee that, but if we ever record from a non-`except`
  context the traceback field will be stale. Worth a comment.
- The test assertion message prints the full traceback blob inline
  (visible in `assertion output`). That's noisy — consider
  formatting only the exception type + message in the assertion
  string and logging the full tracebacks separately.
