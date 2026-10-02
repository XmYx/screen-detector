"""Box selection over a frozen screen frame or an image, used to create detection examples."""
import numpy as np
from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .overlay import exclude_from_capture


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits().asstring(bpl * h), np.uint8).reshape(h, bpl)[:, : w * 3]
    return np.ascontiguousarray(arr.reshape(h, w, 3)[:, :, ::-1])


class BoxSelector(QWidget):
    """Shows an image; the user drags a box. Emits the box in image pixel coordinates.

    Enter without dragging selects the whole image (handy for files that are already a crop).
    """

    selected = pyqtSignal(list)
    cancelled = pyqtSignal()

    def __init__(self, image: np.ndarray, hint: str, screen_geometry: QRect, fullscreen: bool) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.image = image
        self.qimg = bgr_to_qimage(image)
        self.hint = hint
        self._start: QPointF | None = None
        self._end: QPointF | None = None
        self._mouse = QPointF()
        self._done = False
        if fullscreen:
            self.setGeometry(screen_geometry)
        else:
            ih, iw = image.shape[:2]
            s = min(1.0, screen_geometry.width() * 0.8 / iw, screen_geometry.height() * 0.8 / ih)
            w, h = max(320, int(iw * s)), max(240, int(ih * s))
            self.setGeometry(screen_geometry.center().x() - w // 2, screen_geometry.center().y() - h // 2, w, h)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        exclude_from_capture(self)
        self.activateWindow()
        self.raise_()
        self.setFocus()

    # widget <-> image mapping (image is letterboxed into the widget)
    def _view(self) -> tuple[float, float, float]:
        iw, ih = self.qimg.width(), self.qimg.height()
        s = min(self.width() / iw, self.height() / ih)
        return s, (self.width() - iw * s) / 2, (self.height() - ih * s) / 2

    def _to_image(self, p: QPointF) -> tuple[float, float]:
        s, ox, oy = self._view()
        return (p.x() - ox) / s, (p.y() - oy) / s

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        s, ox, oy = self._view()
        target = QRectF(ox, oy, self.qimg.width() * s, self.qimg.height() * s)
        p.drawImage(target, self.qimg)
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))  # dim everything ...
        if self._start and self._end:
            sel = QRectF(self._start, self._end).normalized()
            src = QRectF((sel.x() - ox) / s, (sel.y() - oy) / s, sel.width() / s, sel.height() / s)
            p.drawImage(sel, self.qimg, src)  # ... except the selection
            p.setPen(QPen(QColor("#3ddc84"), 2))
            p.drawRect(sel)
        else:  # crosshair
            p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(0, self._mouse.y()), QPointF(self.width(), self._mouse.y()))
            p.drawLine(QPointF(self._mouse.x(), 0), QPointF(self._mouse.x(), self.height()))
        f = QFont()
        f.setPixelSize(15)
        f.setBold(True)
        p.setFont(f)
        text = f"{self.hint}   ·   drag a box   ·   Enter = whole image   ·   Esc = cancel"
        box = QRectF(0, 18, self.width(), 34)
        pill = QRectF(box.center().x() - p.fontMetrics().horizontalAdvance(text) / 2 - 16, 18,
                      p.fontMetrics().horizontalAdvance(text) + 32, 34)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(16, 18, 22, 220))
        p.drawRoundedRect(pill, 17, 17)
        p.setPen(QColor("#e8eaed"))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._start = self._end = e.position()
        elif e.button() == Qt.MouseButton.RightButton:
            self._cancel()

    def mouseMoveEvent(self, e) -> None:
        self._mouse = e.position()
        if self._start is not None:
            self._end = e.position()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        (x1, y1), (x2, y2) = self._to_image(self._start), self._to_image(e.position())
        ih, iw = self.image.shape[:2]
        box = [max(0, min(x1, x2)), max(0, min(y1, y2)), min(iw, max(x1, x2)), min(ih, max(y1, y2))]
        if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
            self._finish(box)
        else:
            self._start = self._end = None
            self.update()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self._cancel()
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            ih, iw = self.image.shape[:2]
            self._finish([0, 0, iw, ih])

    def _finish(self, box: list) -> None:
        self._done = True
        self.selected.emit(box)
        self.close()

    def _cancel(self) -> None:
        self._done = True
        self.cancelled.emit()
        self.close()

    def closeEvent(self, e) -> None:
        if not self._done:
            self._done = True
            self.cancelled.emit()
        super().closeEvent(e)
