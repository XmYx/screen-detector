import time

import mss
import numpy as np

from .base import BaseCapture


class MSSCapture(BaseCapture):
    """Portable fallback (X11, Windows GDI, macOS). Slower than the native backends."""

    name = "mss"

    def __init__(self, monitor: int, max_fps: int) -> None:
        super().__init__()
        self.monitor = monitor
        self.max_fps = max_fps

    def _run(self) -> None:
        with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
            mons = sct.monitors[1:] or sct.monitors
            mon = mons[min(self.monitor, len(mons) - 1)]
            self.region = (mon["left"], mon["top"], mon["width"], mon["height"])
            self.status = "capturing"
            period = 1.0 / max(1, self.max_fps)
            while self._running:
                t = time.perf_counter()
                img = np.asarray(sct.grab(mon))  # BGRA
                self._publish(np.ascontiguousarray(img[:, :, :3]))
                dt = period - (time.perf_counter() - t)
                if dt > 0:
                    time.sleep(dt)
