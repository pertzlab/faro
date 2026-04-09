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

**Root cause.** `Analyzer.get_stim_mask`
(`faro/core/controller.py:118`) blocks on `fov_state.stim_mask_queue`
for up to 80 s and then falls through with `stim_mask = None`. The
first-frame cellpose inference on GPU is slower than 80 s because the
model / CUDA context is still warming up, so the mask isn't ready in
time. The fallback silently sends `False` via `_build_stim_slm`
without recording anything to `background_errors`.

**Fix options (pick one or combine):**

- **Pre-warm cellpose.** In `Controller.run_experiment` (or a new
  `pipeline.prewarm()` hook), run one dummy `segment(np.zeros(...))`
  before the MDA thread starts. Eats the first-inference cost once,
  outside the acquisition window.
- **Record as a fatal / background error.** When `get_stim_mask`
  times out, append to `analyzer.background_errors` so it surfaces
  in tests, or escalate to `_fatal_error` like the shape-mismatch
  guard does. At minimum the caller should *know* the frame had no
  stim.
- **Make the timeout configurable.** 80 s is hardcoded; pipelines
  with slow first-frame segmenters (cellpose SAM, stardist remote,
  convpaint) need more, or should opt into "block forever" semantics.

Preference: pre-warm + record-as-background-error. Pre-warm makes
normal runs not hit the timeout at all; recording catches the
edge-case where something still fails.

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
`Binning,2x2` after the ROI is set so the test starts from a known
state regardless of what the previous session left in the device.

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
- `KeepDMDAlive.stop()` has a hardcoded `time.sleep(5)` after joining
  the thread (`faro/microscope/pertzlab/moench.py:69-71`). Check if
  that's still needed now that the thread is a daemon and `shutdown`
  is the canonical teardown path.
- `_record_background_error` takes an exception and reads
  `traceback.format_exc()` — which only works if it's called from an
  active `except` block. The storage worker and deferred worker
  already guarantee that, but if we ever record from a non-`except`
  context the traceback field will be stale. Worth a comment.
- The test assertion message prints the full traceback blob inline
  (visible in `assertion output`). That's noisy — consider
  formatting only the exception type + message in the assertion
  string and logging the full tracebacks separately.
