# TODO — NGFF axis names → napari `dims.axis_labels`

When an OME-Zarr v0.5 store is opened in napari via the
[`napari-ome-zarr`](https://github.com/ome/napari-ome-zarr) plugin, the
dimension sliders are labelled `0, 1, 2, 3` instead of the axis names
declared in the store (e.g. `t, p, y, x`). The fix belongs in the
plugin's reader.

---

## Issue

NGFF v0.5 stores carry per-axis names in `multiscales[0].axes`:

```json
{
  "axes": [
    {"name": "t", "type": "time"},
    {"name": "p", "type": "other"},
    {"name": "c", "type": "channel"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"}
  ]
}
```

`napari-ome-zarr` reads the store, splits the channel axis, and
returns `(data, meta, layer_type)` tuples for each layer. The `meta`
dict contains `scale`, `channel_axis`, `name`, `colormap`, `visible`,
`contrast_limits`, `metadata` — but the per-axis `name` field from the
NGFF metadata is **dropped**, so napari falls back to its default
generic dim labels.

Confirmed by grep of the plugin source: zero references to
`axis_labels`, `axes`, or `dims.`.

---

## Motivation

Without axis names on the sliders the user has to mentally map slider
position back to dimension semantics every time they scrub. With four
sliders this is annoying; with NGFF stores that introduce additional
axes (sub-sequence, phase, well, etc.) it gets worse.

The producer side is already doing the right thing — NGFF axis names
are a required field in the spec and any conformant writer emits them.
The gap is purely on the read/display side. Fixing it once in the
plugin benefits every NGFF v0.5 producer.

---

## Where the gap is

`napari-ome-zarr` is the napari plugin layer; it sits between
`ome-zarr-py` (which knows about NGFF) and napari (which has
`viewer.dims.axis_labels`). The plugin already constructs the layer
metadata dict — extending it to carry an `axis_labels` tuple from the
NGFF axes is a one-line read + a one-line forward, plus napari's
normal plumbing for that field.

This is independent of the long-standing plate-labels upstream gap
(`ome-zarr-py#65`, `napari-ome-zarr#54`) — it's a separate
metadata-forwarding fix and shouldn't be bundled with it.

---

## Plan

1. **Patch `ome/napari-ome-zarr`**
   In `napari_ome_zarr/_reader.py`, where the layer metadata dict is
   built, read the multiscales' `axes` (via the appropriate
   `ome-zarr-py` accessor), pull out the `name` of each axis, and pass
   them through as `axis_labels`:

   ```python
   axes = [a["name"] for a in multiscales_axes]
   meta["axis_labels"] = tuple(axes)
   # napari sets viewer.dims.axis_labels from this when the layer is
   # added.
   ```

   Edge cases:
   - **Plate stores**: napari-ome-zarr already tiles wells along x, so
     the resulting layer's axes reduce to e.g. `('t', 'y', 'x')` —
     match that.
   - **`channel_axis` split**: napari consumes the channel axis when
     splitting, so the forwarded `axis_labels` should exclude `'c'`.

2. **Smoke-test** with stores covering the common axis layouts:
   - single-position, time-lapse → `('t', 'y', 'x')`
   - multi-position, time-lapse → `('t', 'p', 'y', 'x')`
   - plate (tiled wells)        → `('t', 'y', 'x')`

3. **Open a PR** against `ome/napari-ome-zarr`. Reference
   `ome/ome-zarr-py#65` for context only — do not bundle.

---

## Files (upstream PR)

| File | Change |
|---|---|
| `napari_ome_zarr/_reader.py` | Forward NGFF `axes.name` into layer meta as `axis_labels`, dropping `'c'` when `channel_axis` is set |
| `tests/test_reader.py` (or equivalent) | One test per axis-layout permutation (single-pos, multi-pos, plate) asserting the returned `axis_labels` tuple |

Rough size: ~10 LOC + ~30 LOC tests.

---

## Notes

- The `type` field on each axis (`time`, `space`, `channel`, etc.) is
  also currently unused by napari; could be forwarded too, but napari
  doesn't consume it for display so it's a no-op for now.
- Forward-compatible with NGFF / useq-schema extensions that introduce
  additional axis kinds (sub-sequence, phase, well). Those will
  inherit the same display gap, so a generic forwarder fixes them
  for free.
- Workaround for end users on the current released plugin: after
  loading, set the labels manually in napari's console:
  ```python
  viewer.dims.axis_labels = ('t', 'p', 'y', 'x')
  ```
