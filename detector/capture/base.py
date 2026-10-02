import threading
import time

import numpy as np


class BaseCapture:
    """Grabs frames on its own thread and keeps only the newest one (BGR uint8)."""

    name = "base"

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.status = "starting"
        # Screen area covered by the stream in desktop coordinates (x, y, w, h), if known.
        self.region: tuple[int, int, int, int] | None = None
        self.fps = 0.0
        # Input sanity, refreshed once per second: "ok" | "blank" | "static", plus a human-readable hint.
        self.health = "unknown"
        self.health_hint = ""
        self.frame_size: tuple[int, int] | None = None
        self._last_sample: np.ndarray | None = None
        self._static_since: float | None = None
        self._fps_t = time.perf_counter()
        self._fps_n = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._safe_run, name=f"capture-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _safe_run(self) -> None:
        try:
            self._run()
        except Exception as e:  # surfaced in the UI status line
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
        finally:
            self._running = False
            with self._cond:
                self._cond.notify_all()

    def _run(self) -> None:
        raise NotImplementedError

    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_t, self._fps_n = now, 0
            self._check_health(frame, now)

    def _check_health(self, frame: np.ndarray, now: float) -> None:
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        sample = frame[:: max(1, h // 90), :: max(1, w // 160)].astype(np.int16)
        if sample.std() < 2.0:
            self.health = "blank"
            shade = "black" if sample.mean() < 8 else "a single colour"
            self.health_hint = (f"frames are {shade}: this backend cannot see the screen"
                                + (" (on Wayland use the pipewire backend)" if self.name == "mss" else ""))
        elif self._last_sample is not None and np.array_equal(sample, self._last_sample):
            self._static_since = self._static_since or now
            if now - self._static_since > 5:
                self.health, self.health_hint = "static", "screen content has not changed for 5+ s"
        else:
            self._static_since = None
            self.health, self.health_hint = "ok", ""
        self._last_sample = sample

    def wait_frame(self, last_seq: int, timeout: float = 0.5) -> tuple[int, np.ndarray | None]:
        """Block until a frame newer than last_seq exists; returns (seq, frame) or (last_seq, None)."""
        with self._cond:
            if self._seq == last_seq and self._running:
                self._cond.wait(timeout)
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._frame

    def latest(self) -> np.ndarray | None:
        with self._cond:
            return self._frame

    @property
    def running(self) -> bool:
        return self._running
