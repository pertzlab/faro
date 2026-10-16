"""Hardware verification: filter-cube turret changes actually land.

The closed NikonTI adapter (``TIFilterBlock1``) can *silently skip* a cube
change when its internal, callback-maintained position cache already equals
the target ("Already at position; not moving", no exception, no error log).
A missed/mis-filtered position callback desyncs that cache, so a genuinely
needed cube change can vanish and the frame is acquired through the wrong
cube. Unlike the XY/Z stages, the turret has no independent re-read or safety
timeout in the adapter, so ``MoenchCMMCorePlus`` adds one: after each channel
change (``setConfig`` here, or an interactive ``setStateLabel``) it reads the
turret back and, on a mismatch, forces a real move (neighbour -> target,
defeating the "already at position" skip).

This test is the turret analogue of a filter-wheel-change test: it drives the
turret through the *distinct* cubes of the multi-cube channel group (unlike
``TTL_ERK``, every channel of which shares one cube) via the engine's
channel-set path (the same path an MDA uses) and asserts the turret
reports the commanded cube after every change.

Gated by ``--scope`` / ``FARO_SCOPE``.
"""

from __future__ import annotations

import pytest
import useq

DEVICE = "TIFilterBlock1"
PROP = "Label"
# Channel group whose presets select *different* cubes (TTL_ERK keeps one cube
# for every channel, so it can't exercise a turret change).
TURRET_GROUP = "TurquoiseNeonGreenRubymiRFP"
REPEATS = 5


def _cube_for_channel(mmc, group: str, config: str) -> str | None:
    """The TIFilterBlock1 label this channel preset selects, or None."""
    cfg = mmc.getConfigData(group, config)
    for i in range(cfg.size()):
        s = cfg.getSetting(i)
        if s.getDeviceLabel() == DEVICE and s.getPropertyName() == PROP:
            return s.getPropertyValue()
    return None


def _distinct_cube_channels(mmc, group: str) -> list[tuple[str, str]]:
    """[(channel, cube), ...]: one representative channel per distinct cube."""
    if group not in mmc.getAvailableConfigGroups():
        return []
    by_cube: dict[str, str] = {}
    for config in mmc.getAvailableConfigs(group):
        cube = _cube_for_channel(mmc, group, config)
        if cube is not None and cube not in by_cube:
            by_cube[cube] = config
    return [(channel, cube) for cube, channel in by_cube.items()]


@pytest.mark.hardware
def test_filter_turret_changes_land(microscope) -> None:
    """Every commanded cube change must leave the turret on that cube.

    Drives the turret through the distinct cubes ``REPEATS`` times via the
    engine's ``_set_event_channel`` (which applies the verify/force-move fix)
    and asserts the hardware reports the commanded cube after each change.
    Without the verify, a change can intermittently leave the turret on a
    stale cube.
    """
    mmc = microscope.mmc
    engine = microscope.engine

    if DEVICE not in mmc.getLoadedDevices():
        pytest.skip(f"{DEVICE} not loaded on this scope")

    pairs = _distinct_cube_channels(mmc, TURRET_GROUP)
    if len(pairs) < 2:
        pytest.skip(
            f"need >=2 distinct cubes in {TURRET_GROUP!r} to exercise a turret "
            f"change; found {len(pairs)}"
        )

    misses: list[tuple[int, str, str, str]] = []
    for rep in range(REPEATS):
        for channel, expected_cube in pairs:
            event = useq.MDAEvent(channel={"group": TURRET_GROUP, "config": channel})
            engine._set_event_channel(event)  # setConfig + verify/force-move
            try:
                mmc.waitForDevice(DEVICE)
            except RuntimeError:
                pass
            actual = mmc.getStateLabel(DEVICE)
            if actual != expected_cube:
                misses.append((rep, channel, expected_cube, actual))

    n_changes = REPEATS * len(pairs)
    cubes = ", ".join(cube for _, cube in pairs)
    print(f"[turret] cycled {n_changes} changes over cubes: {cubes}")
    assert not misses, (
        f"Filter turret did not land on the commanded cube on "
        f"{len(misses)}/{n_changes} change(s): "
        + "; ".join(
            f"rep{r} {ch}: want {want!r} got {got!r}" for r, ch, want, got in misses[:8]
        )
    )
