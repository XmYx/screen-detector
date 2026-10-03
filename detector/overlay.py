import ctypes
import sys
import time
import zlib

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from .config import Settings
from .engine import Detection, FrameResult

WDA_EXCLUDEFROMCAPTURE = 0x11
MODES = ["boxes", "corners", "neon", "markers", "spotlight", "heatmap"]
PALETTES = {"turbo": cv2.COLORMAP_TURBO, "inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA,
            "plasma": cv2.COLORMAP_PLASMA, "viridis": cv2.COLORMAP_VIRIDIS, "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT}
LABEL_COLORS = ["#00ff66", "#ffb020", "#3db8ff", "#ff4d8d", "#b77dff", "#ffe14d", "#2ef2d0", "#ff7a45",
                "#9dff3d", "#ff5cf0"]
HEAT_CELL = 6          # heatmap grid cell size in overlay pixels
HEAT_HALF_LIFE = 0.35  # seconds for the heat to fade to half once an object is gone


def exclude_from_capture(widget: QWidget) -> None:
    """Windows 10 2004+: keep our own windows out of the screen grab (no feedback loop)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), WDA_EXCLUDEFROMCAPTURE)
        except (AttributeError, OSError):
            pass


class Overlay(QWidget):
    """Transparent, click-through, always-on-top window that draws the detections."""

    def __init__(self, settings: Settings) -> None:
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                 | Qt.WindowType.WindowDoesNotAcceptFocus)
        if sys.platform.startswith("linux"):
            flags |= Qt.WindowType.X11BypassWindowManagerHint  # above panels, not managed/moved by the WM
        super().__init__(None, flags)
        for attr in (Qt.WidgetAttribute.WA_TranslucentBackground, Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                     Qt.WidgetAttribute.WA_ShowWithoutActivating, Qt.WidgetAttribute.WA_NoSystemBackground):
            self.setAttribute(attr)
        self.settings = settings
        self.result = FrameResult()
        self.capture_fps = 0.0
        self.capture_warning = ""
        self.font = QFont()
        self.font.setPixelSize(13)
        self.font.setBold(True)
        self._heat: np.ndarray | None = None
        self._heat_t = time.monotonic()
        self._toast = ("", 0.0)  # (text, shown until)

    def toast(self, text: str, seconds: float = 3.5) -> None:
        """Short confirmation message at the top of the screen."""
        self._toast = (text, time.monotonic() + seconds)
        self.update()
        QTimer.singleShot(int(seconds * 1000) + 50, self.update)  # clear it even when no results arrive

    def showEvent(self, event) -> None:
        super().showEvent(event)
        exclude_from_capture(self)

    def set_result(self, result: FrameResult) -> None:
        self.result = result
        if self.settings.overlay_mode == "heatmap":
            self._update_heat()
        self.update()

    # --- colours / geometry helpers ---
    def _color(self, d: Detection) -> QColor:
        s = self.settings
        if s.color_by_label:
            return QColor(LABEL_COLORS[zlib.crc32(d.label.encode()) % len(LABEL_COLORS)])
        return QColor(s.example_color if d.source == "example" else s.box_color)

    def _rects(self) -> list[tuple[Detection, QRectF]]:
        r = self.result
        if not (r.frame_w and r.frame_h):
            return []
        sx, sy = self.width() / r.frame_w, self.height() / r.frame_h
        return [(d, QRectF(d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sy)) for d in r.detections]

    def _text(self, d: Detection) -> str:
        if not self.settings.show_conf:
            return d.label
        tags = f" {d.tags}" if d.source == "example" and d.tags else ""
        return f"{d.label} {d.conf:.2f}{tags}"

    def _chip(self, p: QPainter, fm: QFontMetrics, x: float, y: float, text: str, color: QColor,
              round_: bool = False) -> None:
        """Label pill whose bottom edge sits at y (or below y when there is no room above)."""
        tw, th = fm.horizontalAdvance(text) + 10, fm.height() + 3
        top = y - th if y - th >= 0 else y
        box = QRectF(x, top, tw, th)
        for _ in range(8):  # stack labels that would cover each other
            hit = next((b for b in self._placed if b.intersects(box)), None)
            if hit is None:
                break
            box.moveTop(hit.top() - th - 2 if hit.top() - th - 2 >= 0 else hit.bottom() + 2)
        self._placed.append(box)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        radius = th / 2 if round_ else 2
        p.drawRoundedRect(box, radius, radius)
        p.setPen(QColor("black") if color.lightness() > 140 and color.alpha() > 200 else QColor("white"))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    # --- heatmap ---
    def _update_heat(self) -> None:
        """Gaussian blob per detection, height = confidence; old heat fades with a half-life."""
        gw, gh = max(1, self.width() // HEAT_CELL), max(1, self.height() // HEAT_CELL)
        now = time.monotonic()
        if self._heat is None or self._heat.shape != (gh, gw):
            self._heat = np.zeros((gh, gw), np.float32)
        decay = 0.5 ** ((now - self._heat_t) / HEAT_HALF_LIFE)
        self._heat_t = now
        acc = np.zeros_like(self._heat)
        ys, xs = np.arange(gh, dtype=np.float32), np.arange(gw, dtype=np.float32)
        for d, rect in self._rects():
            cx, cy = rect.center().x() / HEAT_CELL, rect.center().y() / HEAT_CELL
            sx, sy = max(1.0, rect.width() / HEAT_CELL / 2.6), max(1.0, rect.height() / HEAT_CELL / 2.6)
            blob = np.outer(np.exp(-0.5 * ((ys - cy) / sy) ** 2), np.exp(-0.5 * ((xs - cx) / sx) ** 2))
            acc = np.maximum(acc, d.conf * blob)
        self._heat = np.maximum(self._heat * decay, acc)

    def _paint_heat(self, p: QPainter) -> None:
        if self._heat is None or self._heat.max() < 0.02:
            return
        heat = np.clip(self._heat, 0, 1)
        cmap = PALETTES.get(self.settings.heat_palette, cv2.COLORMAP_TURBO)
        rgb = cv2.applyColorMap((heat * 255).astype(np.uint8), cmap)[:, :, ::-1]
        alpha = (np.clip(heat * 1.8, 0, 1) ** 0.8 * 210).astype(np.uint8)
        rgba = np.ascontiguousarray(np.dstack([rgb, alpha]))
        h, w = heat.shape
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(0, 0, w * HEAT_CELL, h * HEAT_CELL), img)

    # --- painting ---
    def paintEvent(self, event) -> None:
        s = self.settings
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self.font)
        fm = QFontMetrics(self.font)
        rects = self._rects()
        self._placed: list[QRectF] = []
        mode = s.overlay_mode if s.overlay_mode in MODES else "boxes"
        p.setOpacity(max(0.05, min(1.0, s.overlay_opacity)))
        if mode == "heatmap":
            self._paint_heat(p)
        elif mode == "spotlight" and rects:
            dim = QPainterPath()
            dim.addRect(QRectF(self.rect()))
            holes = QPainterPath()
            for _, r in rects:
                holes.addRoundedRect(r.adjusted(-10, -10, 10, 10), 14, 14)
            p.fillPath(dim.subtracted(holes), QColor(0, 0, 0, 150))
        t = time.monotonic()
        for d, r in rects:
            c = self._color(d)
            th = s.box_thickness
            if mode == "boxes":
                pen = QPen(c, th)
                pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(r)
            elif mode == "corners":
                L = max(8.0, min(r.width(), r.height()) * 0.22)
                p.setPen(QPen(c, th + 1, cap=Qt.PenCapStyle.SquareCap))
                for (x, y), (dx, dy) in zip(((r.left(), r.top()), (r.right(), r.top()), (r.left(), r.bottom()),
                                             (r.right(), r.bottom())), ((1, 1), (-1, 1), (1, -1), (-1, -1))):
                    p.drawLine(QPointF(x, y), QPointF(x + dx * L, y))
                    p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))
                faint = QColor(c)
                faint.setAlpha(40)
                p.setPen(QPen(faint, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(r)
            elif mode == "neon":
                p.setBrush(Qt.BrushStyle.NoBrush)
                for width, alpha in ((14, 18), (9, 35), (5, 70), (2.5, 255)):
                    g = QColor(c)
                    g.setAlpha(alpha)
                    p.setPen(QPen(g, width))
                    p.drawRoundedRect(r, 8, 8)
                p.setPen(QPen(QColor(c).lighter(160), 1))
                p.drawRoundedRect(r, 8, 8)
            elif mode == "markers":
                center = r.center()
                base = max(6.0, min(r.width(), r.height()) * 0.18)
                phase = (t * 1.6 + (zlib.crc32(d.label.encode()) % 100) / 100) % 1.0
                ring = QColor(c)
                ring.setAlpha(int(220 * (1 - phase)))
                p.setPen(QPen(ring, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(center, base * (1 + phase * 1.8), base * (1 + phase * 1.8))
                p.setPen(QPen(c, 2))
                p.drawEllipse(center, base, base)
                p.setBrush(c)
                p.drawEllipse(center, 3, 3)
            elif mode == "spotlight":
                p.setPen(QPen(c, max(1, th - 1)))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(r.adjusted(-10, -10, 10, 10), 14, 14)
            if s.show_labels:
                text = self._text(d)
                if mode == "markers":
                    base = max(6.0, min(r.width(), r.height()) * 0.18)
                    anchor = QPointF(r.center().x() + base * 0.7, r.center().y() - base * 0.7)
                    tip = QPointF(anchor.x() + 18, anchor.y() - 18)
                    p.setPen(QPen(c, 1.5))
                    p.drawLine(anchor, tip)
                    self._chip(p, fm, tip.x(), tip.y() + fm.height() / 2, text, c, round_=True)
                elif mode == "heatmap":
                    self._chip(p, fm, r.center().x() - fm.horizontalAdvance(text) / 2 - 5, r.top(), text,
                               QColor(0, 0, 0, 170), round_=True)
                elif mode == "spotlight":
                    self._chip(p, fm, r.left() - 10, r.top() - 12, text, c, round_=True)
                elif mode == "neon":
                    self._chip(p, fm, r.left() + 6, r.top() - 4, text, c, round_=True)
                else:
                    self._chip(p, fm, r.left() - th / 2, r.top(), text, c)

        p.setOpacity(1.0)
        r = self.result
        if s.show_hud:
            hud = (f"detect {r.fps:5.1f} fps  {r.infer_ms:4.1f} ms   capture {self.capture_fps:5.1f} fps   "
                   f"hits {len(r.detections)}" + ("   [paused]" if s.paused else ""))
            box = QRectF(12, 40, fm.horizontalAdvance(hud) + 16, fm.height() + 8)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor(s.box_color))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, hud)
        text, until = self._toast
        if text and time.monotonic() < until:
            tw = min(self.width() - 40, fm.horizontalAdvance(text) + 32)
            box = QRectF((self.width() - tw) / 2, 70, tw, fm.height() + 16)
            p.setPen(QPen(QColor(61, 220, 132, 200), 1))
            p.setBrush(QColor(16, 18, 22, 225))
            p.drawRoundedRect(box, box.height() / 2, box.height() / 2)
            p.setPen(QColor("#e8eaed"))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, fm.elidedText(text, Qt.TextElideMode.ElideRight, int(tw - 24)))
        if self.capture_warning:
            warn = f"⚠ capture: {self.capture_warning}"
            box = QRectF(12, 40 + fm.height() + 14, fm.horizontalAdvance(warn) + 16, fm.height() + 8)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(160, 0, 0, 200))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor("white"))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, warn)
        p.end()
