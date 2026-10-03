import os
import signal
import sys

from .config import MODELS_DIR, Settings


def _configure_platform() -> None:
    if sys.platform.startswith("linux"):
        # The portal handshake runs its own GLib loop on a worker thread; keep Qt off GLib.
        os.environ["QT_NO_GLIB"] = "1"
        # Native Wayland clients may not position themselves or stay on top; XWayland windows can.
        if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def main(args: list[str]) -> int:
    _configure_platform()
    MODELS_DIR.mkdir(exist_ok=True)
    os.chdir(MODELS_DIR)  # ultralytics downloads weights into the cwd

    from PyQt6.QtCore import QObject, QPoint, QTimer
    from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
    from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

    import cv2
    import numpy as np

    from .capture import create_capture, resolve_backend
    from .alerts import SoundAlert
    from .engine import FrameResult, InferenceWorker
    from .examples import ExampleStore
    from .overlay import Overlay
    from .panel import Panel
    from .picker import BoxSelector, qimage_to_bgr

    class Controller(QObject):
        def __init__(self, app: QApplication, settings: Settings) -> None:
            super().__init__()
            self.app, self.settings = app, settings
            self.capture = None
            self._placed = None
            self._last_status = ""
            self.overlay = Overlay(settings)
            self.store = ExampleStore()
            self.panel = Panel(settings, self.store)
            self._selector = None
            self._image_queue: list[str] = []
            self.sound = SoundAlert(settings)
            self.worker = InferenceWorker(settings)

            self.worker.result.connect(self.on_result)
            self.worker.status.connect(self.panel.set_status)
            self.worker.saved.connect(self.panel.set_saved)
            self.worker.model_ready.connect(self.panel.set_vocabulary)
            self.panel.targets_changed.connect(self.worker.request_targets)
            self.panel.model_changed.connect(self.worker.request_model)
            self.panel.restart_capture.connect(self.restart_capture)
            self.panel.forget_screen.connect(self.forget_screen)
            self.panel.snapshot.connect(self.worker.request_snapshot)
            self.panel.quit_requested.connect(app.quit)
            self.panel.pick_screen.connect(self.pick_from_screen)
            self.panel.pick_images.connect(self.pick_from_files)
            self.panel.test_sound.connect(self.sound.play)
            self.worker.learned.connect(lambda msg: self.overlay.toast(msg[:1].upper() + msg[1:]))
            self.panel.paste_image.connect(self.pick_from_clipboard)
            self.panel.examples_changed.connect(self.worker.request_examples)
            self._make_tray()
            self.panel.hide_requested.connect(self.hide_panel)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll)
            self.timer.start(500)

        def _make_tray(self) -> None:
            self.tray = None
            if not QSystemTrayIcon.isSystemTrayAvailable():
                return
            pm = QPixmap(32, 32)
            pm.fill(QColor(0, 0, 0, 0))
            p = QPainter(pm)
            p.setPen(QColor("#00ff66"))
            p.setBrush(QColor(0, 255, 102, 90))
            p.drawEllipse(4, 4, 24, 24)
            p.end()
            menu = QMenu()
            menu.addAction("Show/hide options", self.toggle_panel)
            menu.addAction("Pause/resume", self.toggle_pause)
            menu.addSeparator()
            menu.addAction("Quit", self.app.quit)
            self.tray = QSystemTrayIcon(QIcon(pm))
            self.tray.setToolTip("Screen Detector")
            self.tray.setContextMenu(menu)
            self.tray.activated.connect(
                lambda reason: self.toggle_panel() if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
            self.tray.show()

        def start(self) -> None:
            screen = self.app.primaryScreen().geometry()
            self.overlay.setGeometry(screen)
            self.overlay.show()
            pos = self.settings.panel_pos
            on_screen = pos and any(sc.geometry().contains(QPoint(*pos)) for sc in self.app.screens())
            self.panel.place(QPoint(*pos) if on_screen else QPoint(screen.right() - 24, screen.top() + 48))
            if self.settings.panel_visible or self.tray is None:
                self.panel.show()
            self.worker.start()
            self.restart_capture()

        def toggle_panel(self) -> None:
            if self.panel.isVisible():
                self.panel.hide()
            else:
                self.panel.show()
                self.panel.raise_()

        def hide_panel(self) -> None:
            if self.tray is None:  # nothing could bring it back, so just collapse to the pill
                self.panel.set_collapsed(True)
            else:
                self.panel.hide()

        def toggle_pause(self) -> None:
            self.settings.paused = not self.settings.paused
            self.panel.pause.setChecked(self.settings.paused)

        def restart_capture(self) -> None:
            old, self.capture = self.capture, None
            self.worker.set_capture(None)
            if old:
                old.stop()
            s = self.settings
            try:
                cap = create_capture(s.backend, s.monitor, max(s.max_fps, 30), s.restore_token)
            except Exception as e:  # e.g. PyGObject missing for the pipewire backend
                self.panel.set_status(f"capture backend '{resolve_backend(s.backend)}' unavailable: {e}")
                return
            cap.start()
            self.capture = cap
            self._placed = None
            self.worker.set_capture(cap)

        # --- detect-by-example ---
        def pick_from_screen(self, label: str, mode: str) -> None:
            frame = self.capture.latest() if self.capture else None
            if frame is None:
                self.panel.set_status("⚠ no captured frame yet to pick from")
                return
            panel_was_visible = self.panel.isVisible()
            self.overlay.hide()
            self.panel.hide()

            def restore():
                self.overlay.show()
                if panel_was_visible:
                    self.panel.show()
            self._select(frame.copy(), label, mode, self.overlay.geometry(), True, restore)

        def pick_from_files(self, paths: list, label: str, mode: str) -> None:
            """Several reference images: box each one in turn (Enter = whole image)."""
            self._image_queue = list(paths)
            self._next_image(label, mode)

        def _next_image(self, label: str, mode: str) -> None:
            while self._image_queue:
                path = self._image_queue.pop(0)
                data = np.fromfile(path, np.uint8)  # imread cannot open non-ASCII paths on Windows
                img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
                if img is None:
                    self.panel.set_status(f"⚠ could not read image {path}")
                    continue
                left = len(self._image_queue)
                self._select(img, label, mode, self.panel.screen().geometry(), False,
                             lambda: self._next_image(label, mode),
                             extra=f" · {left} more after this" if left else "")
                return

        def pick_from_clipboard(self, label: str, mode: str) -> None:
            qimg = self.app.clipboard().image()
            if qimg.isNull():
                self.panel.set_status("⚠ the clipboard does not contain an image")
                return
            self._select(qimage_to_bgr(qimg), label, mode, self.panel.screen().geometry(), False, lambda: None)

        def _select(self, img, label, mode, geometry, fullscreen, done, extra: str = "") -> None:
            kind = {"auto": "all methods", "similar": "look-alike", "exact": "exact look"}.get(mode, mode)
            sel = BoxSelector(img, f"Example “{label}” ({kind}){extra}", geometry, fullscreen)
            sel.selected.connect(lambda box: (self._add_example(img, box, label, mode), done()))
            sel.cancelled.connect(done)
            self._selector = sel
            sel.show()

        def _add_example(self, img, box, label: str, mode: str) -> None:
            try:
                self.store.add(img, box, label, mode)
            except ValueError as e:
                self.panel.set_status(f"⚠ {e}")
                return
            self.panel.refresh_examples()
            self.worker.request_examples()
            n = sum(1 for e in self.store.items if e.label == label)
            msg = f"Learning example “{label}” ({n} image{'s' if n > 1 else ''} for this label) …"
            self.panel.set_status(msg)
            self.overlay.toast(msg)

        def forget_screen(self) -> None:
            self.settings.restore_token = ""
            self.restart_capture()

        def on_result(self, r: FrameResult) -> None:
            fps = self.capture.fps if self.capture else 0.0
            self.overlay.capture_fps = fps
            self.overlay.set_result(r)
            self.panel.update_stats(r, fps)
            self.sound.update(r)

        def poll(self) -> None:
            cap = self.capture
            if cap is None:
                return
            status = f"capture [{cap.name}]: {cap.error or cap.status}"
            if cap.frame_size:
                status += f" {cap.frame_size[0]}x{cap.frame_size[1]}"
            if cap.health in ("blank", "static"):
                status += f"\n⚠ {cap.health_hint}"
            self.overlay.capture_warning = cap.health_hint if cap.health == "blank" else ""
            self.panel.set_capture_state(cap.health, cap.running and not cap.error)
            if status != self._last_status:
                self._last_status = status
                self.panel.set_status(status)
            token = getattr(cap, "restore_token", "")
            if token and token != self.settings.restore_token:
                self.settings.restore_token = token
                self.settings.save()
            if cap.region and cap.region != self._placed:
                self._placed = cap.region
                self.overlay.setGeometry(self._screen_for(cap.region).geometry())
            self.overlay.raise_()

        def _screen_for(self, region):
            x, y = region[0], region[1]
            screens = self.app.screens()
            if self.capture.name != "dxgi":  # dxgi does not report a desktop position
                for sc in screens:
                    g, dpr = sc.geometry(), sc.devicePixelRatio()
                    if (g.x(), g.y()) == (x, y) or (round(g.x() * dpr), round(g.y() * dpr)) == (x, y):
                        return sc
            if 0 < self.settings.monitor < len(screens):
                return screens[self.settings.monitor]
            return self.app.primaryScreen()

        def shutdown(self) -> None:
            self.settings.panel_visible = self.panel.isVisible()
            self.timer.stop()
            self.worker.stop()
            if self.capture:
                self.capture.stop()
            self.settings.save()

    settings = Settings.load()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Screen Detector")
    app.setQuitOnLastWindowClosed(False)
    ctl = Controller(app, settings)
    ctl.start()

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    tick = QTimer()
    tick.timeout.connect(lambda: None)  # let Python handle Ctrl+C while Qt runs
    tick.start(200)

    code = app.exec()
    ctl.shutdown()
    return code
