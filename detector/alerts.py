"""Sound alert when something new is detected."""
import time
import wave
from collections import deque

import numpy as np
from PyQt6.QtCore import QObject, QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

from .config import ROOT, Settings
from .detection import FrameResult

SOUNDS_DIR = ROOT / "sounds"
PLING = SOUNDS_DIR / "pling.wav"
RECENT_S = 1.5  # a hit only counts as new if more objects are visible than at any time in this window


def builtin_pling() -> str:
    """Generate the default two-tone chime once (no binary asset in the repo)."""
    if not PLING.exists():
        sr = 44100
        t = np.arange(int(sr * 0.45)) / sr
        tone = np.zeros_like(t)
        for start, freq in ((0.0, 1318.5), (0.09, 1975.5)):  # E6 then B6
            tt = np.clip(t - start, 0, None)
            env = np.where(t >= start, np.exp(-tt * 9) * np.minimum(1, tt * 400), 0)
            tone += env * (np.sin(2 * np.pi * freq * tt) + 0.25 * np.sin(4 * np.pi * freq * tt))
        pcm = (tone / np.abs(tone).max() * 0.8 * 32767).astype(np.int16)
        SOUNDS_DIR.mkdir(exist_ok=True)
        with wave.open(str(PLING), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
    return str(PLING)


class SoundAlert(QObject):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self._source = ""
        self._last = 0.0
        self._history: deque[tuple[float, int]] = deque()

    def _count(self, r: FrameResult) -> int:
        wanted = {w.strip().lower() for w in self.settings.sound_labels.split(",") if w.strip()}
        return sum(1 for d in r.detections if not wanted or d.label.lower() in wanted)

    def update(self, r: FrameResult) -> None:
        now = time.monotonic()
        n = self._count(r)
        while self._history and now - self._history[0][0] > RECENT_S:
            self._history.popleft()
        recent = max((c for _, c in self._history), default=0)
        self._history.append((now, n))
        s = self.settings
        if s.sound_enabled and n > recent and now - self._last >= s.sound_cooldown:
            self._last = now
            self.play()

    def play(self) -> None:
        path = self.settings.sound_file or builtin_pling()
        if path != self._source:
            self._source = path
            self.player.setSource(QUrl.fromLocalFile(path))
        self.audio.setVolume(float(self.settings.sound_volume))
        self.player.stop()
        self.player.setPosition(0)
        self.player.play()
