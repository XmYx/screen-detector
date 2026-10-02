import ctypes
import sys

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .config import Settings
from .engine import FrameResult

WDA_EXCLUDEFROMCAPTURE = 0x11


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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        exclude_from_capture(self)

    def set_result(self, result: FrameResult) -> None:
        self.result = result
        self.update()

    def paintEvent(self, event) -> None:
        s, r = self.settings, self.result
        p = QPainter(self)
        p.setFont(self.font)
        fm = QFontMetrics(self.font)
        color = QColor(s.box_color)
        text_color = QColor("black") if color.lightness() > 140 else QColor("white")
        if r.frame_w and r.frame_h:
            sx, sy = self.width() / r.frame_w, self.height() / r.frame_h
            ex_color = QColor(s.example_color)
            ex_text = QColor("black") if ex_color.lightness() > 140 else QColor("white")
            for d in r.detections:
                c, tc = (ex_color, ex_text) if d.source != "model" else (color, text_color)
                pen = QPen(c, s.box_thickness)
                pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                rect = QRectF(d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sy)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect)
                if s.show_labels:
                    text = f"{d.label} {d.conf:.2f}" if s.show_conf else d.label
                    tw, th = fm.horizontalAdvance(text) + 8, fm.height() + 2
                    top = rect.top() - th if rect.top() - th >= 0 else rect.top()
                    box = QRectF(rect.left() - s.box_thickness / 2, top, tw, th)
                    p.fillRect(box, c)
                    p.setPen(tc)
                    p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        if s.show_hud:
            hud = (f"detect {r.fps:5.1f} fps  {r.infer_ms:4.1f} ms   capture {self.capture_fps:5.1f} fps   "
                   f"hits {len(r.detections)}" + ("   [paused]" if s.paused else ""))
            box = QRectF(12, 40, fm.horizontalAdvance(hud) + 16, fm.height() + 8)
            p.fillRect(box, QColor(0, 0, 0, 150))
            p.setPen(QColor(color))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, hud)
        if self.capture_warning:
            warn = f"⚠ capture: {self.capture_warning}"
            box = QRectF(12, 40 + fm.height() + 14, fm.horizontalAdvance(warn) + 16, fm.height() + 8)
            p.fillRect(box, QColor(160, 0, 0, 200))
            p.setPen(QColor("white"))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, warn)
        p.end()
