"""Floating options panel: frameless, semi-transparent, collapses into a live-stats pill."""
import sys

from PyQt6.QtCore import (
    QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QUrl, QVariantAnimation,
    pyqtProperty, pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractButton, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QGraphicsOpacityEffect,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from .capture import available_backends
from .config import BUILTIN_MODELS, DATASET_DIR, MODELS_DIR, Settings
from .engine import FrameResult
from .examples import ExampleStore
from .overlay import exclude_from_capture
from .picker import bgr_to_qimage

IMG_SIZES = [320, 416, 480, 640, 800, 960, 1280, 1600, 1920]
ACCENT = "#3ddc84"
WIDTH = 380
PILL_WIDTH = 330
HEADER_H = 46
ANIM_MS = 260

STYLE = f"""
QWidget {{ color: #e8eaed; font-size: 12px; }}
QLabel[role="section"] {{ color: #8b949e; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
                          padding-top: 10px; }}
QLabel[role="hint"] {{ color: #8b949e; font-size: 11px; }}
QLabel[role="title"] {{ font-size: 13px; font-weight: 700; }}
QLabel[role="sub"] {{ color: #9aa4ad; font-size: 11px; font-family: monospace; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 8px;
    padding: 5px 8px; selection-background-color: {ACCENT}; selection-color: #0b0f0c; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: #1b1e23; border: 1px solid #30363d; border-radius: 6px;
    selection-background-color: rgba(61,220,132,0.25); outline: none; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0; border: none; }}
QPushButton {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px; padding: 6px 12px; }}
QPushButton:hover {{ background: rgba(255,255,255,0.13); }}
QPushButton:pressed {{ background: rgba(255,255,255,0.05); }}
QPushButton[accent="true"] {{ background: {ACCENT}; color: #0b0f0c; border: none; font-weight: 700; }}
QPushButton[accent="true"]:hover {{ background: #5ae69a; }}
QSlider::groove:horizontal {{ height: 4px; background: rgba(255,255,255,0.12); border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QToolTip {{ background: #1b1e23; color: #e8eaed; border: 1px solid #30363d; padding: 4px; }}
QMenu {{ background: #1b1e23; border: 1px solid #30363d; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: rgba(61,220,132,0.25); }}
QListWidget {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }}
QListWidget::item {{ color: #c9d1d9; border-radius: 6px; padding: 2px; }}
QListWidget::item:selected {{ background: rgba(61,220,132,0.22); }}
"""


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class Toggle(QCheckBox):
    """iOS-style switch with an animated knob."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool) -> None:  # noqa: N802 - Qt naming
        super().setChecked(on)
        self._anim.stop()
        self._pos = 1.0 if on else 0.0
        self.update()

    def get_knob(self) -> float:
        return self._pos

    def set_knob(self, v: float) -> None:
        self._pos = v
        self.update()

    knob = pyqtProperty(float, get_knob, set_knob)

    def sizeHint(self) -> QSize:
        return QSize(40 + self.fontMetrics().horizontalAdvance(self.text()) + 10, 24)

    def hitButton(self, pos: QPoint) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(1, (self.height() - 18) / 2, 34, 18)
        off, on = QColor(255, 255, 255, 40), QColor(ACCENT)
        c = QColor.fromRgbF(lerp(off.redF(), on.redF(), self._pos), lerp(off.greenF(), on.greenF(), self._pos),
                            lerp(off.blueF(), on.blueF(), self._pos), lerp(off.alphaF(), on.alphaF(), self._pos))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(track, 9, 9)
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(track.left() + 9 + self._pos * 16, track.center().y()), 7, 7)
        p.setPen(self.palette().color(self.foregroundRole()))
        p.drawText(QRectF(44, 0, self.width() - 44, self.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


class IconButton(QAbstractButton):
    """Small round header button; 'chevron' rotates with the collapse animation."""

    def __init__(self, kind: str, tooltip: str) -> None:
        super().__init__()
        self.kind = kind
        self.angle = 0.0
        self.setFixedSize(28, 28)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, e) -> None:
        self.update()

    def leaveEvent(self, e) -> None:
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.underMouse():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        p.translate(self.width() / 2, self.height() / 2)
        p.setPen(QPen(QColor("#d0d4d9"), 1.8, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin))
        if self.kind == "chevron":
            p.rotate(self.angle)
            path = QPainterPath(QPointF(-5, -2.5))
            path.lineTo(0, 2.5)
            path.lineTo(5, -2.5)
            p.drawPath(path)
        else:  # "more": three dots
            p.setBrush(QColor("#d0d4d9"))
            p.setPen(Qt.PenStyle.NoPen)
            for x in (-6, 0, 6):
                p.drawEllipse(QPointF(x, 0), 1.6, 1.6)


class StatusDot(QWidget):
    COLORS = {"ok": ACCENT, "warn": "#f2c94c", "error": "#ff5f56", "idle": "#6e7681"}

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(12, 12)
        self.state = "idle"

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(self.COLORS[self.state])
        glow = QColor(c)
        glow.setAlpha(70)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(self.rect())
        p.setBrush(c)
        p.drawEllipse(self.rect().adjusted(3, 3, -3, -3))


class Panel(QWidget):
    """Options box. The chevron (or Esc / double-click on the header) collapses it into a pill."""

    targets_changed = pyqtSignal(list)
    model_changed = pyqtSignal(str)
    restart_capture = pyqtSignal()
    forget_screen = pyqtSignal()
    snapshot = pyqtSignal()
    quit_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    pick_screen = pyqtSignal(str, str)       # label, mode
    pick_image = pyqtSignal(str, str, str)   # path, label, mode
    paste_image = pyqtSignal(str, str)       # label, mode
    examples_changed = pyqtSignal()

    def __init__(self, settings: Settings, store: ExampleStore) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Screen Detector")
        self.setStyleSheet(STYLE)
        self.settings = settings
        self.store = store
        self._drag: QPoint | None = None
        self._progress = 0.0 if settings.panel_collapsed else 1.0
        self._anchor: QPoint | None = None  # top-right corner stays put while resizing

        root = QVBoxLayout(self)
        root.setSizeConstraint(QVBoxLayout.SizeConstraint.SetNoConstraint)  # size is driven by the animation
        root.setContentsMargins(14, 0, 14, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        self.clip = QWidget()  # its height animates; the body inside keeps its natural size and is clipped
        self.body = self._build_body()
        self.body.setParent(self.clip)
        self.body_fx = QGraphicsOpacityEffect(self.body)
        self.body.setGraphicsEffect(self.body_fx)
        root.addWidget(self.clip)
        self._body_h = self.body.sizeHint().height()
        self.body.setGeometry(0, 0, WIDTH - 28, self._body_h)

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._apply_progress)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(180)

        QShortcut(QKeySequence("Escape"), self, activated=self.toggle_collapsed)
        self._apply_progress(self._progress)

    # --- construction ---
    def _build_header(self) -> QWidget:
        h = QWidget()
        h.setFixedHeight(HEADER_H)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.dot = StatusDot()
        lay.addWidget(self.dot)
        col = QVBoxLayout()
        col.setSpacing(0)
        title = QLabel("Screen Detector")
        title.setProperty("role", "title")
        self.sub = QLabel("starting ...")
        self.sub.setProperty("role", "sub")
        col.addStretch()
        col.addWidget(title)
        col.addWidget(self.sub)
        col.addStretch()
        lay.addLayout(col, 1)
        self.more = IconButton("more", "Menu")
        self.more.clicked.connect(self._show_menu)
        self.chevron = IconButton("chevron", "Collapse / expand (Esc)")
        self.chevron.clicked.connect(self.toggle_collapsed)
        lay.addWidget(self.more)
        lay.addWidget(self.chevron)
        self.header = h
        return h

    def _section(self, lay: QGridLayout, text: str) -> None:
        lbl = QLabel(text.upper())
        lbl.setProperty("role", "section")
        lay.addWidget(lbl, lay.rowCount(), 0, 1, 2)

    def _row(self, lay: QGridLayout, label: str, w) -> None:
        r = lay.rowCount()
        if label:
            lay.addWidget(QLabel(label), r, 0)
            lay.addWidget(w, r, 1) if isinstance(w, QWidget) else lay.addLayout(w, r, 1)
        else:
            lay.addWidget(w, r, 0, 1, 2) if isinstance(w, QWidget) else lay.addLayout(w, r, 0, 1, 2)

    def _build_body(self) -> QWidget:
        s = self.settings
        body = QWidget()
        g = QGridLayout(body)
        g.setContentsMargins(0, 0, 0, 12)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(7)
        g.setColumnStretch(1, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "hint")
        self._row(g, "", self.status)

        self._section(g, "Detect")
        row = QHBoxLayout()
        self.targets = QLineEdit(s.targets)
        self.targets.setPlaceholderText("cat, dog …  (empty = everything)")
        self.targets.returnPressed.connect(self._apply_targets)
        b = QPushButton("Apply")
        b.setProperty("accent", True)
        b.clicked.connect(self._apply_targets)
        row.addWidget(self.targets, 1)
        row.addWidget(b)
        self._row(g, "Look for", row)
        self.vocab = QLabel("")
        self.vocab.setWordWrap(True)
        self.vocab.setProperty("role", "hint")
        self._row(g, "", self.vocab)

        row = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self._fill_models()
        self.model.activated.connect(lambda _: self._apply_model())
        self.model.lineEdit().returnPressed.connect(self._apply_model)
        b = QPushButton("…")
        b.setFixedWidth(34)
        b.setToolTip("Load custom weights")
        b.clicked.connect(self._browse_model)
        row.addWidget(self.model, 1)
        row.addWidget(b)
        self._row(g, "Model", row)

        row, self.conf_lbl = self._slider(1, 95, round(s.conf * 100), self._conf_changed, f"{s.conf:.2f}")
        self._row(g, "Confidence", row)

        row = QHBoxLayout()
        self.imgsz = QComboBox()
        self.imgsz.addItems([str(v) for v in IMG_SIZES])
        self.imgsz.setCurrentText(str(s.imgsz))
        self.imgsz.setToolTip("Model input resolution. Larger finds smaller/farther objects but is slower.")
        self.imgsz.currentTextChanged.connect(lambda t: setattr(s, "imgsz", int(t)))
        row.addWidget(self.imgsz, 1)
        row.addWidget(QLabel("max fps"))
        row.addWidget(self._spin(1, 240, s.max_fps, lambda v: setattr(s, "max_fps", v)))
        self._row(g, "Input size", row)
        row = QHBoxLayout()
        row.addWidget(self._toggle("FP16", "fp16"))
        self.pause = self._toggle("Pause", "paused")
        row.addWidget(self.pause)
        row.addStretch()
        self._row(g, "", row)

        self._section(g, "Examples · show it what to find")
        row = QHBoxLayout()
        self.ex_label = QLineEdit()
        self.ex_label.setPlaceholderText("label, e.g. chest")
        self.ex_mode = QComboBox()
        self.ex_mode.addItem("Similar (AI)", "similar")
        self.ex_mode.addItem("Exact look", "exact")
        self.ex_mode.setToolTip("Similar: objects that look alike (characters, chests, animals; needs a yoloe model)\n"
                                "Exact look: icons, markers, UI and sprites that always look the same (any model)")
        row.addWidget(self.ex_label, 1)
        row.addWidget(self.ex_mode)
        self._row(g, "New", row)
        row = QHBoxLayout()
        b = QPushButton("Pick from screen")
        b.setProperty("accent", True)
        b.setToolTip("Freeze the screen and drag a box around the thing to find")
        b.clicked.connect(lambda: self.pick_screen.emit(*self._new_example()))
        row.addWidget(b, 1)
        b = QPushButton("Image…")
        b.clicked.connect(self._choose_image)
        row.addWidget(b)
        b = QPushButton("Paste")
        b.setToolTip("Use the image on the clipboard")
        b.clicked.connect(lambda: self.paste_image.emit(*self._new_example()))
        row.addWidget(b)
        self._row(g, "", row)
        self.ex_list = QListWidget()
        self.ex_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.ex_list.setFlow(QListWidget.Flow.LeftToRight)
        self.ex_list.setWrapping(False)
        self.ex_list.setIconSize(QSize(48, 48))
        self.ex_list.setFixedHeight(92)
        self.ex_list.setMovement(QListWidget.Movement.Static)
        self.ex_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ex_list.customContextMenuRequested.connect(self._example_menu)
        self.ex_list.itemChanged.connect(self._example_toggled)
        QShortcut(QKeySequence("Delete"), self.ex_list, activated=self._remove_selected)
        self._row(g, "", self.ex_list)
        row = QHBoxLayout()
        row.addWidget(QLabel("similar ≥"))
        row.addWidget(self._dspin(0.01, 0.95, s.example_conf, "example_conf"))
        row.addWidget(QLabel("exact ≥"))
        row.addWidget(self._dspin(0.3, 0.99, s.exact_thresh, "exact_thresh"))
        row.addStretch()
        self._row(g, "Thresholds", row)
        self.refresh_examples()

        self._section(g, "Overlay")
        row = QHBoxLayout()
        row.addWidget(self._toggle("Labels", "show_labels"))
        row.addWidget(self._toggle("Scores", "show_conf"))
        row.addWidget(self._toggle("HUD", "show_hud"))
        self._row(g, "", row)
        row = QHBoxLayout()
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(46, 26)
        self._paint_color_btn()
        self.color_btn.clicked.connect(self._pick_color)
        row.addWidget(self.color_btn)
        row.addWidget(QLabel("width"))
        row.addWidget(self._spin(1, 8, s.box_thickness, lambda v: setattr(s, "box_thickness", v)))
        row.addStretch()
        self._row(g, "Boxes", row)
        row, self.opacity_lbl = self._slider(30, 100, round(s.panel_opacity * 100), self._opacity_changed,
                                             f"{round(s.panel_opacity * 100)}%")
        self._row(g, "Panel", row)

        self._section(g, "Capture")
        row = QHBoxLayout()
        self.backend = QComboBox()
        self.backend.addItems(available_backends())
        self.backend.setCurrentText(s.backend)
        self.backend.currentTextChanged.connect(lambda t: setattr(s, "backend", t))
        row.addWidget(self.backend, 1)
        row.addWidget(QLabel("monitor"))
        mon = self._spin(0, 8, s.monitor, lambda v: setattr(s, "monitor", v))
        mon.setToolTip("Monitor index (dxgi/mss). On Wayland the screen is chosen in the share dialog.")
        row.addWidget(mon)
        self._row(g, "Backend", row)
        row = QHBoxLayout()
        b = QPushButton("Restart capture")
        b.clicked.connect(self.restart_capture)
        row.addWidget(b)
        if sys.platform.startswith("linux"):
            b = QPushButton("Choose screen…")
            b.setToolTip("Forget the saved Wayland screen-share approval and ask again")
            b.clicked.connect(self.forget_screen)
            row.addWidget(b)
        self._row(g, "", row)

        self._section(g, "Labeler")
        row = QHBoxLayout()
        b = QPushButton("Save snapshot")
        b.setProperty("accent", True)
        b.setToolTip("Save the current frame + detections to dataset/ (YOLO format)")
        b.clicked.connect(self.snapshot)
        row.addWidget(b)
        b = QPushButton("Open folder")
        b.clicked.connect(self._open_dataset)
        row.addWidget(b)
        self._row(g, "", row)
        row = QHBoxLayout()
        auto = QDoubleSpinBox()
        auto.setRange(0, 3600)
        auto.setDecimals(1)
        auto.setSuffix(" s")
        auto.setSpecialValueText("off")
        auto.setValue(s.autosave_interval)
        auto.valueChanged.connect(lambda v: setattr(s, "autosave_interval", v))
        row.addWidget(auto)
        row.addWidget(self._toggle("only hits", "autosave_only_hits"))
        self._row(g, "Auto-save", row)
        self.saved = QLabel("")
        self.saved.setProperty("role", "hint")
        self._row(g, "", self.saved)
        return body

    def _slider(self, lo, hi, val, on_change, text):
        row = QHBoxLayout()
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(val)
        lbl = QLabel(text)
        lbl.setFixedWidth(36)
        sl.valueChanged.connect(on_change)
        row.addWidget(sl, 1)
        row.addWidget(lbl)
        return row, lbl

    def _dspin(self, lo, hi, val, attr) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(0.05)
        sp.setDecimals(2)
        sp.setValue(val)
        sp.setFixedWidth(60)
        sp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sp.valueChanged.connect(lambda v: setattr(self.settings, attr, v))
        return sp

    def _spin(self, lo, hi, val, on_change) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.setFixedWidth(56)
        sp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sp.valueChanged.connect(on_change)
        return sp

    def _toggle(self, text: str, attr: str) -> Toggle:
        t = Toggle(text)
        t.setChecked(getattr(self.settings, attr))
        t.toggled.connect(lambda v: setattr(self.settings, attr, v))
        return t

    def _fill_models(self) -> None:
        names = list(BUILTIN_MODELS)
        names += sorted(p.name for p in MODELS_DIR.glob("*.pt") if p.name not in names)
        if self.settings.model not in names:
            names.append(self.settings.model)
        self.model.addItems(names)
        self.model.setCurrentText(self.settings.model)

    # --- examples ---
    def _new_example(self) -> tuple[str, str]:
        return self.ex_label.text().strip() or "object", self.ex_mode.currentData()

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Example image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.pick_image.emit(path, *self._new_example())

    def refresh_examples(self) -> None:
        self.ex_list.blockSignals(True)
        self.ex_list.clear()
        for ex in self.store.items:
            try:
                pm = QPixmap.fromImage(bgr_to_qimage(ex.patch()))
            except (FileNotFoundError, ValueError):
                continue
            item = QListWidgetItem(QIcon(pm.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation)), ex.label)
            item.setData(Qt.ItemDataRole.UserRole, ex.id)
            item.setToolTip(f"{ex.label} · {'similar (AI)' if ex.mode == 'similar' else 'exact look'}\n"
                            "checkbox: enable · right-click/Delete: remove")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if ex.enabled else Qt.CheckState.Unchecked)
            self.ex_list.addItem(item)
        self.ex_list.blockSignals(False)
        if not self.store.items:
            self.ex_list.setToolTip("No examples yet: pick something on screen, load an image or paste one.")

    def _example_toggled(self, item: QListWidgetItem) -> None:
        self.store.set_enabled(item.data(Qt.ItemDataRole.UserRole), item.checkState() == Qt.CheckState.Checked)
        self.examples_changed.emit()

    def _remove_selected(self) -> None:
        items = self.ex_list.selectedItems()
        for item in items:
            self.store.remove(item.data(Qt.ItemDataRole.UserRole))
        if items:
            self.refresh_examples()
            self.examples_changed.emit()

    def _example_menu(self, pos) -> None:
        item = self.ex_list.itemAt(pos)
        if item is None:
            return
        item.setSelected(True)
        m = QMenu(self)
        m.addAction("Remove", self._remove_selected)
        m.addAction("Remove all with this label", lambda: self._remove_label(item.text()))
        m.exec(self.ex_list.mapToGlobal(pos))

    def _remove_label(self, label: str) -> None:
        for ex in [e for e in self.store.items if e.label == label]:
            self.store.remove(ex.id)
        self.refresh_examples()
        self.examples_changed.emit()

    # --- collapse animation ---
    @property
    def collapsed(self) -> bool:
        return self.settings.panel_collapsed

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self.collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        self.settings.panel_collapsed = collapsed
        if not collapsed:
            self._body_h = self.body.sizeHint().height()
            self.body.setGeometry(0, 0, WIDTH - 28, self._body_h)
        self._anchor = self.geometry().topRight()
        self._anim.stop()
        self._anim.setStartValue(self._progress)
        self._anim.setEndValue(0.0 if collapsed else 1.0)
        self._anim.start()

    def _apply_progress(self, t: float) -> None:
        self._progress = t
        body_h = int(self._body_h * t)
        self.clip.setVisible(body_h > 0)
        self.clip.setFixedHeight(body_h)
        self.body_fx.setOpacity(max(0.0, (t - 0.35) / 0.65))  # content fades after the box starts opening
        self.chevron.angle = lerp(-90, 0, t)
        self.chevron.update()
        w, h = int(lerp(PILL_WIDTH, WIDTH, t)), HEADER_H + body_h
        self.setFixedSize(w, h)
        if self._anchor is not None:
            self.move(self._anchor.x() - w + 1, self._anchor.y())
        self.update()

    def place(self, top_right: QPoint) -> None:
        self._anchor = top_right
        self._apply_progress(self._progress)

    # --- painting / interaction ---
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = lerp(HEADER_H / 2, 14, self._progress)  # pill when collapsed
        p.setPen(QPen(QColor(255, 255, 255, 34), 1))
        p.setBrush(QColor(16, 18, 22, 236))
        p.drawRoundedRect(r, radius, radius)
        if self._progress > 0.02:
            p.setPen(QPen(QColor(255, 255, 255, int(20 * self._progress)), 1))
            p.drawLine(QPointF(14, HEADER_H), QPointF(self.width() - 14, HEADER_H))

    def enterEvent(self, e) -> None:
        self._fade_to(1.0)

    def leaveEvent(self, e) -> None:
        if not self.rect().contains(self.mapFromGlobal(self.cursor().pos())):
            self._fade_to(self.settings.panel_opacity)

    def _fade_to(self, v: float) -> None:
        self._fade.stop()
        self._fade.setEndValue(v)
        self._fade.start()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and e.position().y() <= HEADER_H:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)
            self._anchor = self.geometry().topRight()

    def mouseReleaseEvent(self, e) -> None:
        if self._drag is not None:
            tr = self.geometry().topRight()
            self.settings.panel_pos = [tr.x(), tr.y()]
        self._drag = None

    def mouseDoubleClickEvent(self, e) -> None:
        if e.position().y() <= HEADER_H:
            self.toggle_collapsed()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        exclude_from_capture(self)
        self.setWindowOpacity(self.settings.panel_opacity)

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide_requested.emit()

    def _show_menu(self) -> None:
        m = QMenu(self)
        m.addAction("Pause / resume", lambda: self.pause.setChecked(not self.pause.isChecked()))
        m.addAction("Open dataset folder", self._open_dataset)
        m.addSeparator()
        m.addAction("Hide (tray icon brings it back)", self.hide_requested)
        m.addAction("Quit", self.quit_requested)
        m.exec(self.more.mapToGlobal(QPoint(0, self.more.height() + 4)))

    # --- handlers ---
    def _apply_targets(self) -> None:
        self.settings.targets = self.targets.text()
        self.targets_changed.emit(self.settings.target_list())

    def _apply_model(self) -> None:
        name = self.model.currentText().strip()
        if name and name != self.settings.model:
            self.settings.model = name
            self.model_changed.emit(name)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Model weights", str(MODELS_DIR), "Models (*.pt *.onnx *.engine)")
        if path:
            self.model.setCurrentText(path)
            self._apply_model()

    def _conf_changed(self, v: int) -> None:
        self.settings.conf = v / 100
        self.conf_lbl.setText(f"{v / 100:.2f}")

    def _opacity_changed(self, v: int) -> None:
        self.settings.panel_opacity = v / 100
        self.opacity_lbl.setText(f"{v}%")

    def _paint_color_btn(self) -> None:
        self.color_btn.setStyleSheet(f"background: {self.settings.box_color}; border-radius: 6px;")

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self.settings.box_color), self, "Box colour")
        if c.isValid():
            self.settings.box_color = c.name()
            self._paint_color_btn()

    def _open_dataset(self) -> None:
        DATASET_DIR.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(DATASET_DIR)))

    # --- updates from the controller ---
    def set_status(self, text: str) -> None:
        self.status.setText(text)
        low = text.lower()
        if any(k in low for k in ("error", "failed", "unavailable", "cancelled", "⚠")):
            self.dot.set_state("error" if "⚠" not in text else "warn")
        elif any(k in low for k in ("loading", "downloading", "waiting", "preparing")):
            self.dot.set_state("warn")

    def set_capture_state(self, health: str, running: bool) -> None:
        if not running:
            self.dot.set_state("error")
        else:
            self.dot.set_state({"ok": "ok", "blank": "error", "static": "warn"}.get(health, "warn"))

    def set_saved(self, text: str) -> None:
        self.saved.setText(text)

    def set_vocabulary(self, names: list, open_vocab: bool) -> None:
        if open_vocab:
            self.vocab.setText("Open vocabulary: type any object names.")
            self.vocab.setToolTip("")
        else:
            self.vocab.setText(f"{len(names)} known classes (hover) · yoloe = any name")
            self.vocab.setToolTip(", ".join(names[:400]))

    def update_stats(self, r: FrameResult, capture_fps: float) -> None:
        if self.settings.paused:
            self.sub.setText("paused")
        else:
            self.sub.setText(f"{r.fps:4.0f} fps · {r.infer_ms:4.1f} ms · {len(r.detections)} hits")
