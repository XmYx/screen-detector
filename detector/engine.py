import threading
import time

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from .config import DATASET_DIR, Settings
from .dataset import DatasetWriter
from .detection import (  # noqa: F401  (re-exported for overlay/panel/app)
    OPEN_VOCAB_HINTS, SPECIALISTS, Detection, FrameResult, ensure_weights, is_open_vocab, predict_boxes,
)
from .examples import ExampleStore
from .fusion import ExampleEngine


class Detector:
    """The main model with its "Look for" filter. Words it does not know are handed to the example engine."""

    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model: YOLO | None = None
        self.name = ""
        self.open_vocab = False
        self.class_ids: list[int] | None = None
        self.match_nothing = False

    def load(self, name: str, on_status=lambda msg: None) -> None:
        model = YOLO(ensure_weights(name, on_status))
        self.model, self.name = model, name
        self.open_vocab = is_open_vocab(name)
        self.class_ids, self.match_nothing = None, False

    def vocabulary(self) -> list[str]:
        return [] if self.model is None else list(self.model.names.values())

    def set_targets(self, targets: list[str], have_examples: bool) -> list[str]:
        """Apply the "look for" words; returns the words this model cannot detect.

        With no words, the model reports everything it knows, unless examples are set, in which
        case only the examples are shown.
        """
        self.class_ids, self.match_nothing = None, False
        special = [t for t in targets if t.lower() in SPECIALISTS]  # always handled by their dedicated model
        targets = [t for t in targets if t not in special]
        if not targets:
            self.match_nothing = have_examples or bool(special)
            return special
        if self.open_vocab:
            self.model.set_classes(targets)
            return special
        lookup = {n.lower(): i for i, n in self.model.names.items()}
        ids = [lookup[t.lower()] for t in targets if t.lower() in lookup]
        if ids:
            self.class_ids = ids
        else:
            self.match_nothing = True
        return [t for t in targets if t.lower() not in lookup] + special

    def predict(self, frame: np.ndarray, conf: float, imgsz: int, fp16: bool, tiles: int = 0) -> list[Detection]:
        if self.match_nothing:
            return []
        xyxy, scores, cls = predict_boxes(self.model, frame, conf, imgsz, self.device, fp16,
                                          classes=self.class_ids, tiles=tiles)
        names = self.model.names
        return [Detection(*map(float, b), names[c], float(p)) for b, p, c in zip(xyxy, scores, cls)]


class InferenceWorker(QThread):
    """Pulls the newest captured frame, runs the main model + example engine, emits FrameResult."""

    result = pyqtSignal(object)        # FrameResult
    status = pyqtSignal(str)
    model_ready = pyqtSignal(list, bool)  # vocabulary, open_vocab
    saved = pyqtSignal(str)
    learned = pyqtSignal(str)          # confirmation after (re)learning targets/examples

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self._lock = threading.Lock()
        self._capture = None
        self._pending_model: str | None = settings.model
        self._pending_targets: list[str] | None = settings.target_list()
        self._snapshot = False
        self._running = True
        self.writer = DatasetWriter(DATASET_DIR)

    # --- called from the GUI thread ---
    def set_capture(self, capture) -> None:
        with self._lock:
            self._capture = capture

    def request_model(self, name: str) -> None:
        with self._lock:
            self._pending_model = name
            self._pending_targets = self.settings.target_list()

    def request_targets(self, targets: list[str]) -> None:
        with self._lock:
            self._pending_targets = targets

    def request_examples(self) -> None:
        """Re-learn after examples or matching methods changed."""
        self.request_targets(self.settings.target_list())

    def request_snapshot(self) -> None:
        self._snapshot = True

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    # --- worker thread ---
    def _apply_pending(self, det: Detector, ex: ExampleEngine) -> None:
        with self._lock:
            model, targets = self._pending_model, self._pending_targets
            self._pending_model = self._pending_targets = None
        if model:
            self.status.emit(f"loading {model} ...")
            try:
                det.load(model, self.status.emit)
                det.predict(np.zeros((480, 640, 3), np.uint8), 0.5, self.settings.imgsz, self.settings.fp16)
            except Exception as e:
                self.status.emit(f"model load failed: {e}")
                return
            self.status.emit(f"{model} on {det.device}")
        if targets is None or det.model is None:
            return
        examples = ExampleStore().active()
        try:
            unknown = det.set_targets(targets, bool(examples))
            notes = ex.configure(examples, unknown, self.settings, self.status.emit)
        except Exception as e:
            self.status.emit(f"could not set targets: {e}")
            return
        self.model_ready.emit(det.vocabulary(), det.open_vocab)
        parts = []
        known = [t for t in targets if t not in unknown]
        if known:
            parts.append(f"{', '.join(known)} via {det.name}")
        special = [t for t in unknown if t.lower() in SPECIALISTS]
        open_vocab = [t for t in unknown if t not in special]
        if special:
            parts.append(f"{', '.join(special)} via the face model")
        if open_vocab:
            parts.append(f"{', '.join(open_vocab)} via open-vocabulary YOLOE")
        if examples:
            labels = sorted({e.label for e in examples})
            parts.append(f"{len(examples)} example image(s) for {', '.join(labels)}")
        msg = "looking for " + " · ".join(parts) if parts else f"{det.name}: everything it knows"
        self.status.emit(" · ".join([msg, *notes]))
        self.learned.emit(msg)

    def run(self) -> None:
        det = Detector()
        ex = ExampleEngine(det.device)
        cap, seq = None, 0
        last_auto = time.monotonic()
        fps, n, t_fps = 0.0, 0, time.perf_counter()
        idle_sent = False
        while self._running:
            self._apply_pending(det, ex)
            with self._lock:
                if self._capture is not cap:
                    cap, seq = self._capture, 0
            s = self.settings
            if cap is None or det.model is None or s.paused:
                if not idle_sent:
                    self.result.emit(FrameResult())
                    idle_sent = True
                time.sleep(0.05)
                continue
            seq, frame = cap.wait_frame(seq, 0.25)
            if frame is None:
                continue
            idle_sent = False
            t0 = time.perf_counter()
            try:
                dets = det.predict(frame, s.conf, s.imgsz, s.fp16, tiles=2 if s.small_objects else 0)
                if ex.active:
                    found = ex.detect(frame, s, dets)
                    dets = [d for d in dets if d.label not in ex.example_labels] + found
            except Exception as e:
                self.status.emit(f"inference error: {e}")
                time.sleep(0.5)
                continue
            infer_ms = (time.perf_counter() - t0) * 1000
            n += 1
            if time.perf_counter() - t_fps >= 1.0:
                fps, n, t_fps = n / (time.perf_counter() - t_fps), 0, time.perf_counter()
            h, w = frame.shape[:2]
            self.result.emit(FrameResult(dets, w, h, infer_ms, fps))

            now = time.monotonic()
            auto = s.autosave_interval > 0 and now - last_auto >= s.autosave_interval and (dets or not s.autosave_only_hits)
            if self._snapshot or auto:
                self._snapshot = False
                last_auto = now
                try:
                    path = self.writer.save(frame, dets)
                    self.saved.emit(f"saved {path.name} ({len(dets)} boxes, {self.writer.count()} total)")
                except OSError as e:
                    self.saved.emit(f"save failed: {e}")

            spare = 1.0 / max(1, s.max_fps) - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)
