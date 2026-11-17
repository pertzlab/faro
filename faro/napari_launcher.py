"""Launch napari-micromanager on a faro microscope class.

``python -m napari_micromanager -c some.cfg`` only loads a Micro-Manager
``.cfg``. Several Pertzlab scopes need Python-side setup that a cfg cannot
express -- the DMD wake/blank loop and hold exposure, the ROI-follows-binning
hook, the laser keepalive, the per-scope MDA engine. This launcher builds the
microscope class first and hands its configured core to
``MainWindow(viewer, mmcore=...)``. The rest mirrors upstream's
``napari_micromanager.__main__``.

Examples
--------
    uv run faro-napari moench
    uv run faro-napari moench --kwarg uncropped=True
    uv run faro-napari mypkg.scopes:MyScope --kwarg port=3

Inside the napari console, ``mic`` and ``mmcore`` are available, so e.g.
``mic.calibrate_dmd(CyanStim)`` can be run interactively.

The widgets in ``PERTZLAB_WIDGETS`` get a button in the plugin's toolbar. They
come from the pertzlab branch of pymmcore-widgets, installed with
``uv sync --extra pertzlab-widgets``.
"""

from __future__ import annotations

import argparse
import ast
import logging
import time
from collections.abc import Sequence
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCOPES = {
    "moench": "faro.microscope.pertzlab.moench:Moench",
    "niesen": "faro.microscope.pertzlab.niesen:Niesen",
    "jungfrau": "faro.microscope.pertzlab.jungfrau:Jungfrau",
    "demo": "faro.microscope.demo:MMDemo",
}

# Widgets from the pertzlab pymmcore-widgets branch that get a button in the
# napari-micromanager toolbar: toolbar label -> (pymmcore_widgets class name,
# MDI6 icon name). Uncomment an entry to add its button, comment it out to
# drop it.
PERTZLAB_WIDGETS = {
    "Stage Map": ("StageMapWidget", "map_marker_multiple"),
    "Continuous Focus": ("ContinuousFocusWidget", "focus_auto"),
}


def parse_kwarg(item: str) -> tuple[str, Any]:
    """Parse a ``key=value`` CLI pair; the value is Python-literal-parsed.

    ``uncropped=False`` (or ``false``) yields ``False``, not a truthy string.
    Values that are not Python literals stay strings, so unquoted Windows
    paths work.
    """
    key, sep, raw = item.partition("=")
    if not sep:
        raise SystemExit(f"--kwarg expects key=value, got {item!r}.")
    keywords = {"true": True, "false": False, "none": None}
    if raw.strip().lower() in keywords:
        return key.strip(), keywords[raw.strip().lower()]
    try:
        return key.strip(), ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return key.strip(), raw


def _cache_snapped_image(core) -> None:
    """Let ``getImage()`` be called more than once per ``snapImage()``.

    napari-micromanager reads the snap buffer in its ``imageSnapped`` slot
    (``CoreViewerLink._image_snapped``), and the slot runs synchronously
    inside ``snapImage()`` on the psygnal backend. ``CMMCorePlus.snap()`` --
    what the Snap button calls -- then reads the buffer a second time. The
    Andor SDK3 and PVCAM adapters hand out each snapped frame only once, so
    the second read raises "Camera image buffer read failed" (the demo
    camera does not, which is why ``faro-napari demo`` never showed it).

    The first read of each camera channel after a snap is kept, and later
    callers get a copy of it until the next snap. While an MDA runs the cache
    is bypassed: the plugin does not read snaps then, and faro's engines keep
    the plain single-read path.
    """
    snap_image = core.snapImage
    get_image = core.getImage
    cache: dict[int, Any] = {}

    def snapImage(*args, **kwargs):
        cache.clear()
        return snap_image(*args, **kwargs)

    def getImage(numChannel=None, *, fix=True):
        if core.mda.is_running():
            return get_image(numChannel, fix=fix)
        # None means the first channel, so getImage() and getImage(0) (e.g.
        # from getTaggedImage) share one read. The raw frame is cached and
        # fixImage, which leaves a fixed image unchanged, is applied per call.
        channel = numChannel or 0
        if channel not in cache:
            cache[channel] = get_image(numChannel, fix=False)
        img = cache[channel].copy()
        return core.fixImage(img) if fix else img

    core.snapImage = snapImage
    core.getImage = getImage


def _stage_map_widget(*, parent, mmcore):
    """Build a ``StageMapWidget`` that shows the MDA widget's positions.

    The map has no position list of its own; it binds to an existing one. The
    MDA dock is opened first so there is a list to bind to. Without it the map
    would start empty until a list is picked from the map's toolbar.

    The plugin docks the returned widget as a tab next to the MDA dock, after
    this factory returns. So the MDA dock is moved below the map on the next
    pass of the event loop, and both are visible at the same time.
    """
    from pymmcore_widgets import StageMapWidget
    from qtpy.QtCore import QTimer

    parent._show_dock_widget("MDA")
    widget = StageMapWidget(parent=parent, mmcore=mmcore)
    widget.bindToFirstPositionTable()
    QTimer.singleShot(0, partial(_dock_mda_below, parent, widget))
    return widget


def _dock_mda_below(parent, widget) -> None:
    """Dock the MDA widget below the dock of ``widget``, not as a tab.

    ``splitDockWidget`` adds the second dock as a tab if the first one is in a
    tab group, and the plugin tabifies every dock it adds. So both docks are
    taken out of their tab groups first.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget

    top = widget.parent()
    while top is not None and not isinstance(top, QDockWidget):
        top = top.parent()
    bottom = parent._dock_widgets.get("MDA")
    if top is None or bottom is None:
        logger.warning("Stage map or MDA dock not found; not docking them together.")
        return

    win = parent.viewer.window._qt_window
    for dock in (top, bottom):
        win.removeDockWidget(dock)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, top)
    win.splitDockWidget(top, bottom, Qt.Orientation.Vertical)
    # removeDockWidget hides a dock.
    top.show()
    bottom.show()


def _continuous_focus_widget(mic, *, parent, mmcore):
    """Build a ``ContinuousFocusWidget`` that holds the scope's PFS lock.

    The widget polls the autofocus device every second from the GUI thread. On
    Moench, a PFS access while the MDA engine reloads the autofocus device
    crashes the process. So when the scope has a ``_pfs_lock``, every read and
    write of the widget takes it.
    """
    from pymmcore_widgets import ContinuousFocusWidget

    lock = getattr(mic, "_pfs_lock", None)
    if lock is None:
        return ContinuousFocusWidget(parent=parent, mmcore=mmcore)

    def locked(func):
        def call(*args):
            with lock:
                return func(*args)

        return call

    return ContinuousFocusWidget(
        parent=parent,
        mmcore=mmcore,
        enabled_source=locked(mmcore.isContinuousFocusEnabled),
        locked_source=locked(mmcore.isContinuousFocusLocked),
        set_enabled=locked(mmcore.enableContinuousFocus),
    )


def _add_pertzlab_widgets(mic) -> None:
    """Add the ``PERTZLAB_WIDGETS`` buttons to the napari-micromanager toolbar.

    The plugin builds its toolbar buttons from its module-level
    ``DOCK_WIDGETS`` when ``MainWindow`` is created, so this must run before
    that. A widget is built the first time its button is clicked, as
    ``factory(parent=window, mmcore=core)``.

    Without the ``pertzlab-widgets`` extra, pymmcore_widgets has none of these
    widgets and the plugin runs as is. A widget missing from a pertzlab build
    is skipped with a warning.
    """
    import pymmcore_widgets
    from fonticon_mdi6 import MDI6
    from napari_micromanager._gui_objects._toolbar import DOCK_WIDGETS

    if not any(
        hasattr(pymmcore_widgets, class_name)
        for class_name, _ in PERTZLAB_WIDGETS.values()
    ):
        logger.info(
            "pymmcore-widgets is not the pertzlab build; no extra toolbar "
            "buttons (`uv sync --extra pertzlab-widgets` adds them)."
        )
        return

    factories = {
        "StageMapWidget": _stage_map_widget,
        "ContinuousFocusWidget": partial(_continuous_focus_widget, mic),
    }
    for label, (class_name, icon_name) in PERTZLAB_WIDGETS.items():
        if not hasattr(pymmcore_widgets, class_name):
            logger.warning(
                "pymmcore_widgets has no %s. Skipping the %r button.",
                class_name,
                label,
            )
            continue
        factory = factories.get(class_name, getattr(pymmcore_widgets, class_name))
        DOCK_WIDGETS[label] = (factory, getattr(MDI6, icon_name))


def _resolve_light_properties(core, wanted) -> list[tuple[str, str, str]]:
    """Map ``(device, name[, label])`` entries to ``(device, property, label)``.

    A name matches a property exactly, else as the prefix of one (for names
    an adapter builds at runtime). Without a label, ``device::property`` is
    shown. Unloaded devices and unmatched names are skipped with a warning.
    """
    loaded = set(core.getLoadedDevices())
    found = []
    for device, name, *rest in wanted:
        if device not in loaded:
            logger.warning("Light widget: device %r is not loaded.", device)
            continue
        props = core.getDevicePropertyNames(device)
        match = (
            name
            if name in props
            else next((p for p in props if p.startswith(name)), None)
        )
        if match is None:
            logger.warning(
                "Light widget: %r has no property %r. Properties: %s",
                device,
                name,
                ", ".join(props),
            )
            continue
        label = rest[0] if rest else f"{device}::{match}"
        found.append((device, match, label))
    return found


def _replace_illumination_widget(mic) -> None:
    """Show only ``mic.LIGHT_PROPERTIES`` in the "Illumination Control" dock.

    napari-micromanager's widget shows every Intensity/Power/Level property
    of the config. Like ``_add_pertzlab_widgets``, this must run before
    ``MainWindow`` is created.
    """
    wanted = getattr(mic, "LIGHT_PROPERTIES", None)
    if not wanted:
        return

    from napari_micromanager._gui_objects._toolbar import DOCK_WIDGETS
    from pymmcore_widgets import PropertiesWidget, PropertyWidget
    from qtpy.QtWidgets import QLabel, QSizePolicy

    class LightWidget(PropertiesWidget):
        """Sliders for a fixed list of light-source properties."""

        def __init__(self, *, parent=None, mmcore=None):
            # PropertiesWidget.__init__ calls rebuild(), which reads this.
            self._wanted = list(wanted)
            super().__init__(parent=parent, mmcore=mmcore)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        def rebuild(self) -> None:
            layout = self.layout()
            while layout.count():
                layout.takeAt(0).widget().deleteLater()
            for row, (dev, prop, label) in enumerate(
                _resolve_light_properties(self._mmc, self._wanted)
            ):
                name = QLabel(label)
                name.setToolTip(f"{dev}::{prop}")
                value = PropertyWidget(dev, prop, mmcore=self._mmc)
                layout.addWidget(name, row, 0)
                layout.addWidget(value, row, 1)

    key = "Illumination Control"
    DOCK_WIDGETS[key] = (LightWidget, DOCK_WIDGETS[key][1])


FARO_ICON = Path(__file__).parent / "resources" / "faro_icon.png"


def _set_faro_app_id() -> None:
    """Give the process its own Windows taskbar group.

    napari's ``get_qapp`` sets napari's app ID, so the taskbar groups the
    window with napari and shows napari's icon. Call after ``get_qapp`` and
    before the first window is created; a window keeps the ID it opened with.
    """
    import sys

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("faro.napari")


def _set_faro_icon(viewer) -> None:
    """Show the faro icon on the napari window and in the taskbar.

    napari sets its own application icon when it creates the QApplication;
    this replaces it once the viewer exists.
    """
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    icon = QIcon(str(FARO_ICON))
    QApplication.instance().setWindowIcon(icon)
    viewer.window._qt_window.setWindowIcon(icon)


def main(args: Sequence[str] | None = None) -> None:
    """Build the microscope, then run napari-micromanager on its core."""
    parser = argparse.ArgumentParser(
        prog="faro-napari",
        description="Run napari-micromanager on a faro microscope class.",
        epilog=f"Registered scopes: {', '.join(SCOPES)}",
    )
    parser.add_argument(
        "scope", help="registered scope name, or 'module.path:ClassName'"
    )
    parser.add_argument(
        "--kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keyword argument for the microscope class (repeatable)",
    )
    parsed_args = parser.parse_args(args)

    target = SCOPES.get(parsed_args.scope, parsed_args.scope)
    if ":" not in target:
        parser.error(f"unknown scope {target!r}")
    module_path, _, class_name = target.partition(":")
    cls = getattr(import_module(module_path), class_name)
    mic = cls(**dict(parse_kwarg(item) for item in parsed_args.kwarg))
    _cache_snapped_image(mic.mmc)

    # Imported only after the microscope is built: pymmcore_widgets sets
    # PYMM_SIGNALS_BACKEND=qt when unset, and faro needs the psygnal backend.
    import napari
    from napari.qt import get_qapp
    from napari_micromanager.main_window import MainWindow

    from faro.core.qt import deliver_gui_slots_on_main_thread

    # psygnal delivers the core's events on the thread that sends them (the
    # MMCore notification thread, the MDA thread). The widgets connect after
    # this call, so their slots run on the GUI thread; faro's callbacks do
    # not change.
    get_qapp()
    _set_faro_app_id()
    if not deliver_gui_slots_on_main_thread(mic.mmc):
        logger.info(
            "Core events (%s) are not routed to the GUI thread.",
            type(mic.mmc.events).__name__,
        )
    _add_pertzlab_widgets(mic)
    _replace_illumination_widget(mic)
    viewer = napari.Viewer()
    win = MainWindow(viewer, mmcore=mic.mmc)
    dw = viewer.window.add_dock_widget(win, name="MicroManager", area="top")
    if hasattr(dw, "_close_btn"):
        dw._close_btn = False
    viewer.update_console({"mic": mic, "mmcore": mic.mmc})
    _set_faro_icon(viewer)
    napari.run()

    # The window is closed, but its widgets are still connected to the core.
    # Block the signals so the teardown does not call into deleted Qt objects.
    mic.mmc.events.block()
    mic.mmc.mda.events.block()
    # The plugin does not cancel an MDA on a core it does not own. Cancel it
    # and wait, so the devices are not unloaded while the engine still uses
    # them.
    mic.mmc.mda.cancel()
    while mic.mmc.mda.is_running():
        time.sleep(0.1)
    # Explicit teardown: the atexit hook of PyMMCoreMicroscope only holds a
    # weakref, and IPython's exit hook frees ``mic`` (the console namespace
    # held the last reference) before that hook runs.
    mic._teardown_hardware()


if __name__ == "__main__":
    main()
