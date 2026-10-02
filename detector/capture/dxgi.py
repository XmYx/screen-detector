import time

import dxcam  # Windows only: DXGI Desktop Duplication, 100+ fps with low CPU

from .base import BaseCapture


class DXGICapture(BaseCapture):
    name = "dxgi"

    def __init__(self, monitor: int, max_fps: int) -> None:
        super().__init__()
        self.monitor = monitor
        self.max_fps = max_fps

    def _run(self) -> None:
        cam = dxcam.create(output_idx=self.monitor, output_color="BGR")
        if cam is None:
            raise RuntimeError(f"no DXGI output {self.monitor}")
        try:
            self.region = (0, 0, cam.width, cam.height)
            self.status = "capturing"
            period = 1.0 / max(1, self.max_fps)
            while self._running:
                t = time.perf_counter()
                frame = cam.grab()  # None when the screen did not change
                if frame is not None:
                    self._publish(frame.copy())
                dt = period - (time.perf_counter() - t)
                time.sleep(dt if dt > 0 else 0.001)
        finally:
            cam.release()
            del cam
