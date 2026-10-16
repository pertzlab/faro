"""Qt status widget for the controller's currently-bound run.

Construct once (optionally with a :class:`faro.core.controller.Controller`)
and re-point it at each new controller with :meth:`set_controller`::

    from faro.widgets import ExperimentStatusWidget
    widget = ExperimentStatusWidget(ctrl)          # or ExperimentStatusWidget()
    viewer.window.add_dock_widget(widget, name="Experiment")
    ...
    widget.set_controller(new_ctrl)                # swap controllers at runtime

Staging a run with ``ctrl.load_experiment(events)`` previews the plan in
the widget (event strip, FOV map, planned duration) and turns the Stop
button into a Start button; pressing Start calls
``ctrl.start_experiment()``. Calling ``ctrl.run_experiment(events)``
directly still works and starts immediately.

Layout (top to bottom):

  - State label  (RUNNING / DONE / CANCELLING / ERROR, color-coded)
  - Legend chips (imaging / stim / ref) -- the chip matching the *current*
    event type is fully opaque; the others are dimmed
  - Event strip  (one cell per RTMEvent, color-coded by type; past cells
    are fully opaque, future cells are dimmed; the current cell has a
    darker border)
  - FOV map      (one dot per unique FOV position, equal-aspect, with a
    grey path drawn in visit order; the dot for the current FOV is
    re-colored in the active event-type's color)
  - Stats panels (three separate shaded frames: timing -- event N/M,
    elapsed, scheduled, lag, remaining; queues -- storage / pipeline /
    deferred depths, the bounded two drawn as fill bars; errors)
  - Pause + Start/Stop buttons

While an experiment is staged (``load_experiment``) the button reads
``Start`` and begins the run; during a run it reads ``Stop``, cancels
the run *and* calls ``finish_experiment()`` (flush buffered frames to
disk, drop the Analyzer) so the next run starts clean; the state banner
shows ``STOPPING...`` for the duration. A cancel from outside faro,
e.g. the stop button of a napari-micromanager MDA widget, cancels the
run like ``handle.cancel()`` would; call ``finish_experiment()`` (or
start the next run) yourself afterwards.

The widget subscribes to ``ctrl.runStarted`` and ``ctrl.experimentLoaded``,
so it automatically re-binds to whichever run is current. Each
``RunHandle.statusChanged`` emission
updates the labels / strip / map; a small QTimer also refreshes the
``elapsed`` / ``remaining`` fields between status updates so the clock
doesn't appear frozen between frames.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Sequence

from qtpy.QtCore import Qt, QTimer, QPointF, QRectF
from qtpy.QtGui import (
    QBrush,
    QColor,
    QFontDatabase,
    QPainter,
    QPalette,
    QPen,
)
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from faro.core.data_structures import ImgType, WaitEvent
from faro.core.run_status import RunHandle, RunStatus

if TYPE_CHECKING:
    from faro.core.controller import Controller


# ─────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────

EVENT_COLORS: dict[str, str] = {
    "imaging": "#2e7d32",
    "stim": "#1565c0",
    "ref": "#ef6c00",
    "wait": "#9e9e9e",
}
DEFAULT_EVENT_COLOR = "#888888"

# Corner radius -- matches napari's own widgets (buttons, layer controls).
_RADIUS_PX = 3

# Translucent neutral overlay for the grouped panels (FOV map / stats).
# A 50%-grey at low alpha lightens a dark theme and darkens a light one,
# so the grouping reads on both without hardcoding a theme color.
_PANEL_BG = "rgba(128, 128, 128, 28)"

# Event strip
_FUTURE_ALPHA = 90
_PAST_ALPHA = 255
_BORDER_PX = 2
_GAP_PX = 1
_MIN_GAP_AT_CELL_W = 3.0
_WAIT_HATCH_MIN_W = 8.0  # min cell width (px) to overlay the wait hatch

# FOV map
_DOT_RADIUS_PX = 5
_PATH_WIDTH_PX = 2
_MAP_PADDING_PX = 24
_MIN_WORLD_EXTENT = 1e-6

# Lag warn threshold (red is recognizable on both light and dark themes)
_LAG_WARN_S = 5.0
_LAG_BAD_COLOR = "#e53935"

# Queue fill bars (storage / pipeline). The bar's chunk is a translucent
# fill drawn *behind* the "N / max" text; depth >= _QUEUE_WARN_FRAC of max
# flips fill + text to red, mirroring the lag warning.
_QUEUE_BAR_HEIGHT = 18
_QUEUE_WARN_FRAC = 0.8
# Neutral mid-grey: at low-ish alpha it lightens a dark theme and darkens
# a light one, so it reads on both without hardcoding a theme color (the
# same trick as _PANEL_BG).
_BAR_FILL = "rgba(128, 128, 128, 120)"
_BAR_FILL_WARN = "rgba(229, 57, 53, 130)"  # _LAG_BAD_COLOR, translucent


# ─────────────────────────────────────────────────────────────────────────
# Helpers (event-list introspection + small formatters)
# ─────────────────────────────────────────────────────────────────────────


def _event_type_token(ev) -> str:
    """Map an RTMEvent to one of {"wait", "ref", "stim", "imaging"} for visualisation.

    Order of precedence matches what the user sees as the *dominant* effect
    for that timepoint: wait > ref > stim > imaging.

    RTMEvent doesn't expose ``stim`` / ``ref`` booleans directly --
    ``events_to_dataframe`` derives them from the channel tuple lengths,
    so we do the same here.
    """
    if isinstance(ev, WaitEvent):
        return "wait"
    if getattr(ev, "ref_channels", ()):
        return "ref"
    if getattr(ev, "stim_channels", ()):
        return "stim"
    # Fallback for plain MDAEvents: peek at metadata['img_type']
    md = getattr(ev, "metadata", None) or {}
    img_type = md.get("img_type")
    if img_type == ImgType.IMG_REF:
        return "ref"
    if img_type == ImgType.IMG_STIM:
        return "stim"
    return "imaging"


def _extract_plan(
    events: Sequence,
) -> tuple[list[str], list[int], list[tuple[float, float]], list[float]]:
    """Walk events once and return (types, fovs, positions_by_fov, scheduled).

    - ``types``: per-event type token
    - ``fovs``:  per-event FOV index
    - ``positions_by_fov``: list indexed by FOV index, holding ``(x, y)``
      of that FOV's stage position (taken from the first event that visits
      each FOV).
    - ``scheduled``: per-event ``min_start_time`` in seconds (0.0 if missing)
    """
    types: list[str] = []
    fovs: list[int] = []
    scheduled: list[float] = []
    seen_fov: dict[int, tuple[float, float]] = {}

    for ev in events:
        types.append(_event_type_token(ev))
        p = ev.index.get("p", 0)
        fovs.append(p)
        if p not in seen_fov:
            seen_fov[p] = (float(ev.x_pos or 0.0), float(ev.y_pos or 0.0))
        mst = getattr(ev, "min_start_time", None)
        scheduled.append(float(mst) if mst is not None else 0.0)

    if seen_fov:
        max_p = max(seen_fov)
        positions = [seen_fov.get(i, (0.0, 0.0)) for i in range(max_p + 1)]
    else:
        positions = []
    return types, fovs, positions, scheduled


def format_duration(seconds: float, *, show_ms: bool = False) -> str:
    """``hh:mm:ss h`` / ``mm:ss min`` / ``s s`` with optional .mmm.

    Leading components are dropped when zero, and the largest displayed
    component is suffixed with its unit ("h" / "min" / "s") so the unit
    stays explicit even after truncation.
    """
    if seconds < 0:
        return "-" + format_duration(-seconds, show_ms=show_ms)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    if show_ms:
        if h:
            return f"{h:d}:{m:02d}:{s:06.3f} h"
        if m:
            return f"{m:d}:{s:06.3f} min"
        return f"{s:.3f} s"
    s_int = int(s)
    if h:
        return f"{h:d}:{m:02d}:{s_int:02d} h"
    if m:
        return f"{m:d}:{s_int:02d} min"
    return f"{s_int:d} s"


def _runs(seq: Sequence[str]):
    """Yield (start, end_exclusive, value) for each contiguous run."""
    if not seq:
        return
    start, cur = 0, seq[0]
    for i in range(1, len(seq)):
        if seq[i] != cur:
            yield start, i, cur
            start, cur = i, seq[i]
    yield start, len(seq), cur


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _chip_style(color_hex: str, *, active: bool) -> str:
    r, g, b = _hex_to_rgb(color_hex)
    if active:
        return (
            f"background-color: rgba({r},{g},{b},255); color: white; "
            f"padding: 2px 8px; border-radius: {_RADIUS_PX}px; font-weight: bold;"
        )
    return (
        f"background-color: rgba({r},{g},{b},60); color: rgba({r},{g},{b},105); "
        f"padding: 2px 8px; border-radius: {_RADIUS_PX}px; font-weight: bold;"
    )


def _bar_style(warn: bool) -> str:
    """Stylesheet for a queue fill bar; *warn* swaps fill + text to red."""
    fill = _BAR_FILL_WARN if warn else _BAR_FILL
    text = f"color: {_LAG_BAD_COLOR}; font-weight: bold; " if warn else ""
    return (
        f"QProgressBar {{ border: none; border-radius: {_RADIUS_PX}px; "
        f"background-color: {_PANEL_BG}; {text}}}"
        f"QProgressBar::chunk {{ background-color: {fill}; "
        f"border-radius: {_RADIUS_PX}px; }}"
    )


def _make_queue_bar(font) -> QProgressBar:
    """A compact QProgressBar used as a fill bar behind 'N / max' text.

    The chunk *is* the background fill; the format string carries the
    numeric read-out. ``bar._warn`` caches the last applied warn state so
    the stylesheet is only re-set when it actually changes.
    """
    bar = QProgressBar()
    bar.setTextVisible(True)
    bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
    bar.setFixedHeight(_QUEUE_BAR_HEIGHT)
    bar.setFont(font)
    bar.setRange(0, 1)
    bar.setValue(0)
    bar.setFormat("-")
    bar._warn = None  # None -> first _set_queue_bar always applies a style
    bar.setStyleSheet(_bar_style(False))
    return bar


def _wrap_panel(form: QFormLayout) -> QFrame:
    """Wrap a stats form in a subtly-shaded, rounded panel frame.

    Echoes napari's boxed layer-controls sections. Each stats section
    gets its own panel, so the layout's normal inter-widget spacing
    reads as a clear gap between distinct areas (a hairline rule did
    not separate them clearly enough).
    """
    panel = QFrame()
    panel.setObjectName("faroPanel")
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addLayout(form)
    panel.setStyleSheet(
        f"QFrame#faroPanel {{ background-color: {_PANEL_BG}; "
        f"border-radius: {_RADIUS_PX}px; }}"
    )
    return panel


# ─────────────────────────────────────────────────────────────────────────
# EventStrip
# ─────────────────────────────────────────────────────────────────────────


class EventStrip(QWidget):
    """Horizontal strip with one cell per event, color-coded by type.

    Past + current cells are fully opaque (acts as a progress bar). Future
    cells are dimmed. Same-type contiguous runs are merged into a single
    fill so 1000s of events still render correctly with consistent alpha
    rather than the over-stacking that per-cell alpha causes at sub-pixel
    cell widths.
    """

    def __init__(self, event_types: Sequence[str], parent: QWidget | None = None):
        super().__init__(parent)
        self._types = list(event_types)
        self._current = -1
        self.setMinimumHeight(20)
        self.setMinimumWidth(120)

    def set_types(self, event_types: Sequence[str]) -> None:
        self._types = list(event_types)
        self._current = -1
        self.update()

    def set_current(self, index: int) -> None:
        if index != self._current:
            self._current = index
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)

        # Empty state: grey rounded placeholder so the timeline region is
        # visible before a run is loaded (replaced by the colored progress
        # bar once events arrive).
        if not self._types:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(128, 128, 128, 28)))
            painter.drawRoundedRect(QRectF(self.rect()), _RADIUS_PX, _RADIUS_PX)
            placeholder = self.palette().color(QPalette.ColorRole.WindowText)
            placeholder.setAlpha(128)
            painter.setPen(QPen(placeholder))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "(no events loaded)",
            )
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        n = len(self._types)
        w = self.width()
        h = self.height()
        gap = _GAP_PX if (w / max(n, 1)) >= _MIN_GAP_AT_CELL_W else 0
        cell_w = (w - max(0, n - 1) * gap) / n
        stride = cell_w + gap

        def x_for(i: int) -> float:
            return i * stride

        def _fill_run(start: int, end: int, t: str, alpha: int) -> None:
            color = QColor(EVENT_COLORS.get(t, DEFAULT_EVENT_COLOR))
            color.setAlpha(alpha)
            # Span to the next run's start so phase boundaries don't leave
            # the trailing inter-cell gap visible (within a run the merged
            # fill already covers those gaps).
            x0 = x_for(start)
            x1 = min(x_for(end), float(w))
            rect = QRectF(x0, 0, x1 - x0, h)
            # Solid base keeps thin cells visible; a bare hatch over a
            # 1-2px wide wait cell renders as near-transparent and reads
            # as a gap. Overlay the hatch only when wide enough to show.
            painter.fillRect(rect, color)
            if t == "wait" and rect.width() >= _WAIT_HATCH_MIN_W:
                hatch = QColor(0, 0, 0, alpha // 3)
                painter.fillRect(rect, QBrush(hatch, Qt.BrushStyle.BDiagPattern))

        # Future / dim layer
        for start, end, t in _runs(self._types):
            _fill_run(start, end, t, _FUTURE_ALPHA)

        # Past + current overlay
        if self._current >= 0:
            past_end = min(self._current + 1, n)
            for start, end, t in _runs(self._types[:past_end]):
                _fill_run(start, end, t, _PAST_ALPHA)

        # Active border
        if 0 <= self._current < n:
            t = self._types[self._current]
            pen = QPen(QColor(EVENT_COLORS.get(t, DEFAULT_EVENT_COLOR)).darker(160))
            pen.setWidth(_BORDER_PX)
            painter.setPen(pen)
            x0 = x_for(self._current)
            cell = min(x_for(self._current + 1), float(w)) - x0
            visible_w = max(cell, 3.0)
            x = min(max(0.0, x0 - (visible_w - cell) / 2), w - visible_w)
            painter.drawRect(QRectF(x + 0.5, 0.5, visible_w - 1, h - 1))


# ─────────────────────────────────────────────────────────────────────────
# FovMap
# ─────────────────────────────────────────────────────────────────────────


class FovMap(QWidget):
    """Equal-aspect map of FOV positions with a visit-order path.

    The active dot is recolored by ``set_current(idx, color=...)`` so it
    can match the current event type's color. Background is transparent so
    napari's theme shows through; the drawing area is constrained to the
    largest centered square so the widget never appears taller than wide.
    """

    def __init__(
        self,
        positions: Sequence[tuple[float, float]],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._positions = list(positions)
        self._current = -1
        self._active_color = EVENT_COLORS["imaging"]
        self.setMinimumWidth(160)
        # The paint code (_world_to_screen) already confines the drawing
        # to the largest centered square in min(width, height), so the
        # *widget* itself need not be square. Make it freely resizable in
        # both directions. The old approach -- setSizePolicy(.., Fixed)
        # plus setFixedHeight(width) in resizeEvent -- pinned the panel's
        # minimum height to its width, so once the dock was undocked and
        # widened it could no longer be shrunk vertically.
        from qtpy.QtWidgets import QSizePolicy

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(80)

    def set_positions(self, positions: Sequence[tuple[float, float]]) -> None:
        self._positions = list(positions)
        self._current = -1
        self.update()

    def set_current(self, index: int, color: str | None = None) -> None:
        changed = index != self._current or (
            color is not None and color != self._active_color
        )
        if color is not None:
            self._active_color = color
        self._current = index
        if changed:
            self.update()

    def _world_to_screen(self) -> tuple[float, float, float]:
        if not self._positions:
            return 1.0, 0.0, 0.0
        xs = [p[0] for p in self._positions]
        ys = [p[1] for p in self._positions]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = max(x_max - x_min, _MIN_WORLD_EXTENT)
        y_span = max(y_max - y_min, _MIN_WORLD_EXTENT)
        # Confine the drawing to the largest centered square -- so even if
        # the widget ends up wider than tall (or vice-versa), the dots stay
        # together in a square region rather than getting spread thin
        # along the longer axis.
        side = min(self.width(), self.height())
        usable = max(1.0, side - 2 * _MAP_PADDING_PX)
        scale_x = usable / x_span if x_span > _MIN_WORLD_EXTENT * 10 else float("inf")
        scale_y = usable / y_span if y_span > _MIN_WORLD_EXTENT * 10 else float("inf")
        scale = min(scale_x, scale_y)
        if scale == float("inf"):
            scale = 1.0
        x_off = self.width() / 2 - scale * (x_min + x_max) / 2
        y_off = self.height() / 2 + scale * (y_min + y_max) / 2  # +y up
        return scale, x_off, y_off

    def _to_screen(self, x: float, y: float) -> QPointF:
        s, xo, yo = self._world_to_screen()
        return QPointF(s * x + xo, yo - s * y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Panel background -- the map paints its own rounded-rect fill (same
        # translucent neutral as the stats panel) rather than sitting in a
        # QFrame, so the FOV-counter text lands exactly at the panel's
        # top-left corner regardless of how the layout sizes the widget.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(128, 128, 128, 28)))
        painter.drawRoundedRect(QRectF(self.rect()), _RADIUS_PX, _RADIUS_PX)

        # FOV counter, pinned to the panel's top-left corner.
        n = len(self._positions)
        if n == 0:
            label = "FOV -/-"
        else:
            cur_txt = str(self._current + 1) if 0 <= self._current < n else "-"
            label = f"FOV {cur_txt}/{n}"
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        painter.setPen(QPen(text_color))
        painter.drawText(QPointF(8, 6 + painter.fontMetrics().ascent()), label)

        if not self._positions:
            return

        # Inactive dots / path use napari's foreground text color at 50%
        # alpha, so the map reads correctly on both light and dark themes.
        inactive = self.palette().color(QPalette.ColorRole.WindowText)
        inactive.setAlpha(128)

        pts = [self._to_screen(*p) for p in self._positions]

        if len(pts) >= 2:
            pen = QPen(inactive)
            pen.setWidth(_PATH_WIDTH_PX)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            for a, b in zip(pts[:-1], pts[1:]):
                painter.drawLine(a, b)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(inactive))
        for pt in pts:
            painter.drawEllipse(pt, _DOT_RADIUS_PX, _DOT_RADIUS_PX)

        if 0 <= self._current < len(pts):
            painter.setBrush(QBrush(QColor(self._active_color)))
            painter.drawEllipse(pts[self._current], _DOT_RADIUS_PX, _DOT_RADIUS_PX)


# ─────────────────────────────────────────────────────────────────────────
# ExperimentStatusWidget
# ─────────────────────────────────────────────────────────────────────────


class ExperimentStatusWidget(QWidget):
    """Read-out + Start/Stop button for the controller's currently-bound run.

    The widget outlives individual controllers: construct it once (with
    or without a controller), dock it, and re-point it at each new
    controller via :meth:`set_controller`.
    """

    def __init__(
        self, controller: "Controller | None" = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._controller: Controller | None = None
        self._handle: RunHandle | None = None
        # True while _on_stop_clicked is inside cancel()+finish_experiment();
        # _refresh keeps the state banner on "STOPPING..." for the duration.
        self._finishing = False
        # True while an experiment is staged via load_experiment but not
        # started -- the Start/Stop button is in "Start" mode.
        self._loaded = False

        # Cached plan derived from the events list (staged or running)
        self._event_types: list[str] = []
        self._event_fovs: list[int] = []
        self._scheduled: list[float] = []
        self._total_duration: float = 0.0

        self._build_ui()
        self._refresh(None)

        if controller is not None:
            self.set_controller(controller)

        # Drain psygnal's queued callbacks via a main-thread QTimer; without
        # this, statusChanged emissions from the worker thread would sit in
        # psygnal's _GLOBAL_QUEUE forever. Idempotent across widgets.
        try:
            from psygnal.qt import start_emitting_from_queue

            start_emitting_from_queue()
        except ImportError:
            pass

        # Tick the elapsed/remaining clocks between statusChanged emissions
        # so the time fields don't visibly freeze between frames.
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(250)

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("Experiment status")

        # ── State chip -- translucent neutral fill (matches the FOV map /
        # stats panels), so it keeps the legend-chip shape without
        # competing with the imaging/stim/ref colors.
        self._state_label = QLabel("idle")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_label.setStyleSheet(
            "font-weight: bold; padding: 2px 8px; "
            f"border-radius: {_RADIUS_PX}px; "
            f"background-color: {_PANEL_BG};"
        )

        # ── Legend chips
        self._legend_chips: dict[str, QLabel] = {}
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.setSpacing(6)
        for label, key in [
            ("imaging", "imaging"),
            ("stim", "stim"),
            ("ref", "ref"),
            ("wait", "wait"),
        ]:
            chip = QLabel(label)
            chip.setStyleSheet(_chip_style(EVENT_COLORS[key], active=False))
            self._legend_chips[key] = chip
            legend_row.addWidget(chip)
        legend_row.addStretch(1)

        # ── Strip + map
        self._strip = EventStrip([])
        self._map = FovMap([])

        # ── Stats form -- inherit napari's font/palette; only the time
        # values get the platform's fixed-width font (column alignment).
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._event_value = QLabel("-/-")
        self._elapsed_value = QLabel("-")
        self._scheduled_value = QLabel("-")
        self._lag_value = QLabel("-")
        self._remaining_value = QLabel("-")
        self._deferred_value = QLabel("-")
        self._errors_value = QLabel("-")
        right_align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for w in (
            self._event_value,
            self._elapsed_value,
            self._scheduled_value,
            self._lag_value,
            self._remaining_value,
            self._deferred_value,
            self._errors_value,
        ):
            w.setAlignment(right_align)
        for w in (
            self._elapsed_value,
            self._scheduled_value,
            self._lag_value,
            self._remaining_value,
            self._deferred_value,
        ):
            w.setFont(mono)

        # storage / pipeline depths render as a fill bar behind "N / max"
        # text -- a QProgressBar whose chunk *is* the background fill.
        # deferred has no bound, so it stays a plain count label.
        self._storage_bar = _make_queue_bar(mono)
        self._pipeline_bar = _make_queue_bar(mono)

        def _stat_form() -> QFormLayout:
            f = QFormLayout()
            f.setContentsMargins(6, 6, 6, 6)
            f.setSpacing(2)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            # Stretch the value column so right-aligned text / the queue
            # bars land at the panel's right edge, not hugging the label.
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            return f

        timing_form = _stat_form()
        timing_form.addRow("event:", self._event_value)
        timing_form.addRow("elapsed:", self._elapsed_value)
        timing_form.addRow("scheduled:", self._scheduled_value)
        timing_form.addRow("lag:", self._lag_value)
        timing_form.addRow("remaining:", self._remaining_value)

        queues_form = _stat_form()
        queues_form.addRow("storage:", self._storage_bar)
        queues_form.addRow("pipeline:", self._pipeline_bar)
        queues_form.addRow("deferred:", self._deferred_value)

        errors_form = _stat_form()
        errors_form.addRow("errors:", self._errors_value)

        # ── Stats panels: three separate shaded frames (timing / queues
        # / errors). Standalone panels with the layout's normal spacing
        # between them -- like the FOV map above -- read as clearly
        # distinct areas. (The FOV map paints its own matching background
        # in paintEvent, so it isn't wrapped here.)
        timing_panel = _wrap_panel(timing_form)
        queues_panel = _wrap_panel(queues_form)
        errors_panel = _wrap_panel(errors_form)

        # ── Pause + Start/Stop buttons. One button plays both roles:
        # "Start" while an experiment is staged (self._loaded), "Stop"
        # once a run is bound.
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._pause_btn.setEnabled(False)
        self._stop_btn = QPushButton("Start")
        self._stop_btn.clicked.connect(self._on_start_stop_clicked)
        self._stop_btn.setEnabled(False)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._pause_btn)
        button_row.addWidget(self._stop_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._state_label)
        layout.addLayout(legend_row)
        layout.addWidget(self._strip)
        # The map is the one elastic element: give it the layout stretch
        # so all leftover vertical space goes to it (the map then draws
        # the largest centered square that fits). No trailing addStretch
        # -- that spacer used to swallow the slack and pin the map at its
        # minimum height.
        layout.addWidget(self._map, 1)
        layout.addWidget(timing_panel)
        layout.addWidget(queues_panel)
        layout.addWidget(errors_panel)
        layout.addLayout(button_row)

    # -- controller / run binding --------------------------------------------

    @property
    def handle(self) -> RunHandle | None:
        """The RunHandle of the run the widget is currently bound to.

        Lets code reach a run that was started from the UI's Start
        button, e.g. ``status_wdg.handle.wait()``. None while no run is
        bound (idle, or an experiment staged but not started); stays set
        after the run finishes, until the next load / controller swap.
        Equivalent to ``controller.current_handle`` on the bound
        controller.
        """
        return self._handle

    @property
    def controller(self) -> "Controller | None":
        """The Controller the widget is bound to (None when detached)."""
        return self._controller

    def set_controller(self, controller: "Controller | None") -> None:
        """Bind the widget to *controller*, replacing any previous binding.

        Lets the widget be constructed once and re-pointed at each new
        controller instead of rebuilding it per experiment. Pass ``None``
        to detach entirely.

        On attach the widget picks up the controller's present state: a
        run in progress (or just finished) is bound immediately, an
        experiment staged via ``load_experiment`` is previewed, and
        otherwise the widget renders idle.
        """
        if controller is self._controller:
            return

        if self._controller is not None:
            for signal, slot in (
                (self._controller.runStarted, self._on_run_started),
                (self._controller.experimentLoaded, self._on_experiment_loaded),
            ):
                try:
                    signal.disconnect(slot)
                except Exception:
                    pass
        self._detach_handle()
        self._loaded = False
        self._controller = controller

        if controller is None:
            self._set_plan([])
            self._refresh(None)
            return

        controller.runStarted.connect(self._on_run_started)
        controller.experimentLoaded.connect(self._on_experiment_loaded)

        handle = getattr(controller, "current_handle", None)
        pending = getattr(controller, "pending_events", None)
        if handle is not None:
            self._on_run_started(handle)
        elif pending:
            self._on_experiment_loaded(pending)
        else:
            self._set_plan([])
            self._refresh(None)

    def _detach_handle(self) -> None:
        """Disconnect + drop the bound RunHandle (if any)."""
        if self._handle is not None:
            try:
                self._handle.statusChanged.disconnect(self._refresh)
            except Exception:
                pass
        self._handle = None

    def _set_plan(self, events: Sequence) -> None:
        """Rebuild the plan-derived widgets (strip, map, totals) from *events*."""
        types, fovs, positions, scheduled = _extract_plan(events)
        self._event_types = types
        self._event_fovs = fovs
        self._scheduled = scheduled
        self._total_duration = scheduled[-1] if scheduled else 0.0
        self._strip.set_types(types)
        self._map.set_positions(positions)

    def _on_experiment_loaded(self, events: Sequence) -> None:
        """Preview a staged (loaded-but-not-started) experiment.

        Shows the full event strip (all cells dimmed / future), the FOV
        positions, the event count, and the planned duration; the
        Start/Stop button flips to ``Start``. Any previous run's handle
        is dropped so its final status doesn't overwrite the preview
        from ``_tick``.
        """
        self._detach_handle()
        self._loaded = True
        self._set_plan(events or [])
        self._render_loaded()

    def _on_run_started(self, handle: RunHandle) -> None:
        """Re-bind to a new run; rebuild the strip + map from its events."""
        self._detach_handle()
        self._loaded = False

        self._handle = handle
        # thread="main" routes worker-thread emits through psygnal's main-
        # thread queue (drained by start_emitting_from_queue's QTimer in
        # __init__). Without it the slot would run synchronously off the
        # worker thread and touch QWidgets / OpenGL from the wrong thread.
        handle.statusChanged.connect(self._refresh, thread="main")

        # Rebuild plan-derived widgets from the events list, if we have one
        self._set_plan(getattr(handle, "events", None) or [])

        self._refresh(handle.status())

    def _on_start_stop_clicked(self) -> None:
        """Dispatch the Start/Stop button by mode."""
        if self._loaded:
            self._on_start_clicked()
        else:
            self._on_stop_clicked()

    def _on_start_clicked(self) -> None:
        """Start the staged experiment (button is in Start mode)."""
        if self._controller is None or not self._loaded:
            return
        # start_experiment -> run_experiment emits runStarted synchronously,
        # so _on_run_started rebinds the widget (and flips the button back
        # to Stop) before this returns. Disable up front against re-entry.
        self._stop_btn.setEnabled(False)
        try:
            self._controller.start_experiment()
        except BaseException as exc:  # noqa: BLE001 - surface, don't crash the slot
            import traceback

            traceback.print_exc()
            self._state_label.setText(f"START FAILED: {type(exc).__name__}")
            self._stop_btn.setEnabled(True)

    def _on_stop_clicked(self) -> None:
        """Cancel the run, then finish the experiment.

        ``cancel()`` aborts the acquisition loop; ``finish_experiment()``
        then flushes buffered frames to disk and disposes of the
        Analyzer, so the next run starts clean instead of leaking the
        old one. The finish drain runs with the Qt loop pumped, so
        napari stays responsive; the state banner reads ``STOPPING...``
        until it returns.
        """
        if self._handle is None or self._controller is None or self._finishing:
            return
        self._finishing = True
        # Disable both buttons up front: finish_experiment() blocks this
        # slot (pumping Qt), so without this a second click would re-enter.
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._state_label.setText("STOPPING...")
        failed = False
        try:
            self._handle.cancel()
            self._controller.finish_experiment()
        except BaseException as exc:  # noqa: BLE001 - surface, don't crash the slot
            import traceback

            traceback.print_exc()
            self._state_label.setText(f"STOP FAILED: {type(exc).__name__}")
            failed = True
        finally:
            self._finishing = False
        if failed:
            # finish_experiment may leave the Analyzer alive (e.g. a drain
            # timeout) -- re-enable Stop so the user can retry.
            self._stop_btn.setEnabled(True)
        else:
            # Teardown done -- reflect the run's final state.
            self._refresh(self._handle.status() if self._handle else None)

    def _on_pause_clicked(self) -> None:
        """Toggle pause / resume on the bound handle."""
        if self._handle is None:
            return
        if self._handle.is_paused():
            self._handle.resume()
        else:
            self._handle.pause()
        # statusChanged may not fire if state didn't transition (pause
        # during "waiting" defers the state flip until the wait ends),
        # so refresh the label locally to keep it in sync.
        self._update_buttons(self._handle.status().state)

    # -- refresh ------------------------------------------------------------

    def _refresh(self, status: RunStatus | None) -> None:
        """Slot connected to ``handle.statusChanged``.

        Also called once with ``None`` at construction time (no handle yet).
        """
        if status is None:
            self._render_idle()
            return

        # ── State banner -- plain bold text, no background fill (a colored
        # banner clashed with the imaging/stim/ref legend colors). While
        # _on_stop_clicked runs, keep it on STOPPING... -- statusChanged
        # emissions from the winding-down worker would otherwise flip it
        # to DONE before the finish drain completes.
        if self._finishing:
            self._state_label.setText("STOPPING...")
        elif status.state == "waiting" and status.wait_remaining_s is not None:
            self._state_label.setText(
                f"WAITING {format_duration(status.wait_remaining_s)}"
            )
        else:
            self._state_label.setText(status.state.upper())

        # ── Strip + map cursor
        cur_idx = self._current_index(status)
        n_total = status.n_events_total or len(self._event_types)

        if 0 <= cur_idx < len(self._event_types):
            t = self._event_types[cur_idx]
            color = EVENT_COLORS.get(t, DEFAULT_EVENT_COLOR)
            self._strip.set_current(cur_idx)
            if t == "wait":  # no FOV
                self._map.set_current(-1)
            else:
                fov_for_event = self._event_fovs[cur_idx]
                self._map.set_current(fov_for_event, color=color)
            self._update_legend(active_type=t)
        else:
            self._strip.set_current(-1)
            self._map.set_current(-1)
            self._update_legend(active_type=None)

        # ── Stats: event index
        self._event_value.setText(f"{cur_idx + 1} / {n_total}" if n_total else "-/-")

        # ── Stats: elapsed, scheduled, lag, remaining
        self._render_time_fields(status, cur_idx)

        # ── Errors
        n_errors = len(status.background_errors)
        if status.fatal_error is not None:
            self._errors_value.setText(
                f"{n_errors} bg + fatal: {type(status.fatal_error).__name__}"
            )
            self._errors_value.setStyleSheet(
                f"color: {_LAG_BAD_COLOR}; font-weight: bold;"
            )
        else:
            self._errors_value.setText(str(n_errors))
            self._errors_value.setStyleSheet("")

        # ── Queue depths
        self._render_queue_fields()

        # ── Buttons -- left alone while finishing (deliberately disabled
        # by _on_stop_clicked; a stale "running" emission must not re-enable).
        if not self._finishing:
            self._update_buttons(status.state)

    def _update_buttons(self, state: str) -> None:
        """Enable/label Pause + Start/Stop according to the run state.

        Only called while a run handle is bound, so the Start/Stop
        button is always in Stop mode here; Start mode is set by
        ``_render_loaded``.
        """
        live = state in ("running", "pausing", "paused", "waiting")
        self._pause_btn.setEnabled(live)
        # Drive the label off pause_event so a click during "waiting"
        # flips to Resume immediately, even though state stays "waiting"
        # until the wait completes.
        paused = self._handle is not None and self._handle.is_paused()
        self._pause_btn.setText("Resume" if paused else "Pause")
        self._stop_btn.setText("Stop")
        self._stop_btn.setEnabled(live)

    def _render_idle(self) -> None:
        self._state_label.setText("idle (no run yet)")
        self._strip.set_current(-1)
        self._map.set_current(-1)
        self._update_legend(active_type=None)
        self._event_value.setText("-/-")
        for w in (
            self._elapsed_value,
            self._scheduled_value,
            self._lag_value,
            self._remaining_value,
        ):
            w.setText("-")
        self._lag_value.setStyleSheet("")
        self._errors_value.setText("-")
        self._errors_value.setStyleSheet("")
        self._render_queue_fields()
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("Pause")
        self._stop_btn.setText("Start")
        self._stop_btn.setEnabled(False)

    def _render_loaded(self) -> None:
        """Static preview of a staged experiment (no live run to poll).

        The strip shows every event at future alpha, the map shows all
        FOV positions, ``remaining`` shows the planned total duration,
        and the Start/Stop button reads ``Start``.
        """
        n = len(self._event_types)
        self._state_label.setText("LOADED (ready to start)")
        self._strip.set_current(-1)
        self._map.set_current(-1)
        self._update_legend(active_type=None)
        self._event_value.setText(f"0 / {n}" if n else "-/-")
        for w in (self._elapsed_value, self._scheduled_value, self._lag_value):
            w.setText("-")
        self._lag_value.setStyleSheet("")
        self._remaining_value.setText(
            format_duration(self._total_duration) if self._total_duration else "-"
        )
        self._errors_value.setText("-")
        self._errors_value.setStyleSheet("")
        self._render_queue_fields()
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("Pause")
        self._stop_btn.setText("Start")
        self._stop_btn.setEnabled(True)

    @staticmethod
    def _current_index(status: RunStatus) -> int:
        """Index of the RTMEvent currently being acquired (-1 if none yet).

        Uses ``n_events_acquired`` -- RTMEvents whose first frame has
        arrived -- which is the *same unit* as ``n_events_total`` and as
        the event strip / FOV map, so progress can never exceed the
        total. Deliberately NOT ``n_frames_received`` (that counts
        per-channel MDAEvents, ~2-3 per RTMEvent, so it overshoots the
        total) and NOT ``n_events_consumed`` (the feed loop runs several
        events ahead of the engine because of the backpressure window,
        which would make the strip jump). Both ``_refresh`` and the
        ``_tick`` QTimer call this so they always agree on "current".
        While ``waiting``, ``n_events_acquired`` doesn't bump (no frame
        arrives for a WaitEvent), so use ``n_events_consumed`` to keep
        the cursor on the wait cell during the countdown.
        """
        if status.state == "waiting" and status.n_events_consumed > 0:
            return status.n_events_consumed - 1
        return status.n_events_acquired - 1 if status.n_events_acquired > 0 else -1

    def _render_time_fields(self, status: RunStatus, cur_idx: int) -> None:
        # Elapsed: prefer (now - started_at); fall back to last_frame_wallclock.
        if status.started_at is not None:
            if status.finished_at is not None:
                elapsed = status.finished_at - status.started_at
            else:
                elapsed = time.monotonic() - status.started_at
        else:
            elapsed = None

        scheduled = (
            self._scheduled[cur_idx] if 0 <= cur_idx < len(self._scheduled) else None
        )

        # lag: prefer the controller's per-frame measurement; fall back to
        # elapsed-vs-scheduled if the per-frame value isn't populated yet.
        lag_s: float | None = None
        if status.lag_ms is not None:
            lag_s = status.lag_ms / 1000.0
        elif elapsed is not None and scheduled is not None:
            lag_s = elapsed - scheduled

        remaining: float | None = None
        if elapsed is not None and self._total_duration:
            remaining = max(0.0, self._total_duration - elapsed)

        self._elapsed_value.setText(
            format_duration(elapsed) if elapsed is not None else "-"
        )
        self._scheduled_value.setText(
            format_duration(scheduled) if scheduled is not None else "-"
        )
        if lag_s is None:
            self._lag_value.setText("-")
            self._lag_value.setStyleSheet("")
        else:
            sign = "+" if lag_s >= 0 else ""
            self._lag_value.setText(f"{sign}{format_duration(lag_s, show_ms=True)}")
            if lag_s > _LAG_WARN_S:
                self._lag_value.setStyleSheet(
                    f"color: {_LAG_BAD_COLOR}; font-weight: bold;"
                )
            else:
                self._lag_value.setStyleSheet("")
        self._remaining_value.setText(
            format_duration(remaining) if remaining is not None else "-"
        )

    def _render_queue_fields(self) -> None:
        """Refresh the storage / pipeline / deferred read-outs.

        Polled from the QTimer so the depths track live between frames --
        in particular the storage bar is seen draining to 0 while
        finish_experiment() runs. Shows empty "-" bars when no experiment
        is active.
        """
        stats = self._controller.queue_stats() if self._controller is not None else None
        if stats is None:
            self._set_queue_bar(self._storage_bar, 0, 0)
            self._set_queue_bar(self._pipeline_bar, 0, 0)
            self._deferred_value.setText("-")
            return
        self._set_queue_bar(self._storage_bar, stats.storage_depth, stats.storage_max)
        self._set_queue_bar(
            self._pipeline_bar, stats.pipeline_inflight, stats.pipeline_max
        )
        self._deferred_value.setText(str(stats.deferred_depth))

    @staticmethod
    def _set_queue_bar(bar: QProgressBar, depth: int, maximum: int) -> None:
        """Drive one queue bar: fill fraction, 'N / max' text, warn color.

        Fill + text turn red once depth reaches _QUEUE_WARN_FRAC of max --
        a near-full storage queue means disk writes can't keep pace and
        the camera buffer is at risk. ``maximum <= 0`` renders an empty
        "-" bar (no active experiment).
        """
        if maximum <= 0:
            bar.setRange(0, 1)
            bar.setValue(0)
            bar.setFormat("-")
            warn = False
        else:
            bar.setRange(0, maximum)
            bar.setValue(min(depth, maximum))
            bar.setFormat(f"{depth} / {maximum}")
            warn = depth >= _QUEUE_WARN_FRAC * maximum
        # Re-style only on a warn-state change -- setStyleSheet every tick
        # would force a needless restyle/repaint.
        if warn != bar._warn:
            bar._warn = warn
            bar.setStyleSheet(_bar_style(warn))

    def _update_legend(self, active_type: str | None) -> None:
        for key, chip in self._legend_chips.items():
            chip.setStyleSheet(
                _chip_style(EVENT_COLORS[key], active=(key == active_type))
            )

    def _tick(self) -> None:
        """QTimer slot: poll the handle and refresh the whole widget.

        ``statusChanged`` is the push path, but its cross-thread queued
        delivery (psygnal worker thread -> Qt main thread) proved
        unreliable in some embeddings -- notably Jupyter notebooks --
        leaving the strip / FOV cursor / counters frozen even though
        ``handle.status()`` is current. Polling the latest snapshot here
        every tick guarantees the whole widget stays live regardless of
        signal delivery; the push connection remains as an optimisation.
        ``_refresh``'s repaints are change-guarded, so a full refresh at
        this cadence is cheap.
        """
        if self._handle is None:
            return
        self._refresh(self._handle.status())
