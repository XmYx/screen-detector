import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset

from .config import DATASET_DIR, MODELS_DIR, Settings
from .dataset import DatasetWriter
from .examples import Example, ExampleStore, TemplateMatcher, visual_embeddings

OPEN_VOCAB_HINTS = ("yoloe", "world")


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    conf: float
    source: str = "model"  # model | similar | exact


@dataclass
class FrameResult:
    detections: list[Detection] = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    infer_ms: float = 0.0
    fps: float = 0.0


def ensure_weights(name: str, on_status=lambda msg: None) -> str:
    """Resolve a model name/path to a local file, downloading official ultralytics weights into models/."""
    p = Path(name)
    if p.is_file():
        return str(p)
    local = MODELS_DIR / p.name
    if local.is_file():
        return str(local)
    if p.name not in GITHUB_ASSETS_NAMES:
        raise FileNotFoundError(f"{name} is not in models/ and is not a downloadable ultralytics model")
    on_status(f"downloading {p.name} ...")
    return str(attempt_download_asset(local))


class Detector:
    """Thin wrapper over an ultralytics model with a target-class filter."""

    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model: YOLO | None = None
        self.name = ""
        self.open_vocab = False
        self.class_ids: list[int] | None = None
        self.match_nothing = False
        self.example_labels: set[str] = set()  # labels coming only from "similar" examples
        self.matcher: TemplateMatcher | None = None

    def load(self, name: str, on_status=lambda msg: None) -> None:
        model = YOLO(ensure_weights(name, on_status))
        self.model, self.name = model, name
        stem = Path(name).stem.lower()
        # "-pf" (prompt-free) models ship a fixed large vocabulary and are filtered like COCO models
        self.open_vocab = any(h in stem for h in OPEN_VOCAB_HINTS) and not stem.endswith("-pf")
        self.class_ids, self.match_nothing = None, False

    def vocabulary(self) -> list[str]:
        return [] if self.model is None else list(self.model.names.values())

    def set_targets(self, targets: list[str], examples: list[Example] = ()) -> list[str]:
        """Apply the "look for" text plus user examples; returns warnings for the status line.

        With examples but no text, only the examples are detected.
        """
        self.class_ids, self.match_nothing, self.example_labels = None, False, set()
        similar = [e for e in examples if e.mode == "similar"]
        exact = [e for e in examples if e.mode == "exact"]
        self.matcher = TemplateMatcher(exact, self.device) if exact else None
        warnings = []
        if self.open_vocab:
            names, embs = [], []
            if targets:
                names += targets
                embs.append(self.model.get_text_pe(targets).float())
            for label, emb in visual_embeddings(self.model, similar, self.device):
                names.append(label)
                embs.append(emb[None])
            self.example_labels = {e.label for e in similar} - set(targets)
            if names:
                pe = torch.cat([e.to(self.device) for e in embs], dim=1)
                self.model.set_classes(names, pe)
            else:
                self.match_nothing = bool(exact)
            return warnings
        if similar:
            warnings.append(f"{len(similar)} 'similar' example(s) need a yoloe model")
        if not targets:
            self.match_nothing = bool(examples)  # examples only: keep the model quiet
            return warnings
        lookup = {n.lower(): i for i, n in self.model.names.items()}
        ids = [lookup[t.lower()] for t in targets if t.lower() in lookup]
        unknown = [t for t in targets if t.lower() not in lookup]
        if unknown:
            warnings.append(f"not in {self.name} vocabulary: {', '.join(unknown)} (try a yoloe model)")
        if ids:
            self.class_ids = ids
        else:
            self.match_nothing = True
        return warnings

    def predict(self, frame: np.ndarray, conf: float, imgsz: int, fp16: bool,
                example_conf: float = 0.15, exact_thresh: float = 0.8) -> list[Detection]:
        out: list[Detection] = []
        if not self.match_nothing:
            lo = min(conf, example_conf) if self.example_labels else conf
            r = self.model.predict(
                frame, conf=lo, imgsz=imgsz, classes=self.class_ids, device=self.device,
                quantize=16 if fp16 and self.device != "cpu" else None, max_det=100, verbose=False,
            )[0]
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for b, p, c in zip(xyxy, confs, cls):
                    label = r.names[c]
                    from_example = label in self.example_labels
                    if p >= (example_conf if from_example else conf):
                        out.append(Detection(*map(float, b), label, float(p), "similar" if from_example else "model"))
        if self.matcher:
            out += [Detection(*m, source="exact") for m in self.matcher.match(frame, exact_thresh)]
        return out


class InferenceWorker(QThread):
    """Pulls the newest captured frame, runs the detector, emits FrameResult."""

    result = pyqtSignal(object)        # FrameResult
    status = pyqtSignal(str)
    model_ready = pyqtSignal(list, bool)  # vocabulary, open_vocab
    saved = pyqtSignal(str)

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
        """Re-learn after examples were added/removed/toggled."""
        self.request_targets(self.settings.target_list())

    def request_snapshot(self) -> None:
        self._snapshot = True

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    # --- worker thread ---
    def _apply_pending(self, det: Detector) -> None:
        with self._lock:
            model, targets = self._pending_model, self._pending_targets
            self._pending_model = self._pending_targets = None
        if model:
            self.status.emit(f"loading {model} ...")
            try:
                det.load(model, self.status.emit)
                if det.open_vocab:  # first set_classes() downloads the text encoder
                    self.status.emit("preparing text encoder ...")
                det.match_nothing = False
                det.predict(np.zeros((480, 640, 3), np.uint8), 0.5, self.settings.imgsz, self.settings.fp16)
            except Exception as e:
                self.status.emit(f"model load failed: {e}")
                return
            self.status.emit(f"{model} on {det.device}")
        if targets is not None and det.model is not None:
            examples = ExampleStore().active()
            if examples:
                self.status.emit(f"learning {len(examples)} example(s) ...")
            try:
                warnings = det.set_targets(targets, examples)
            except Exception as e:
                self.status.emit(f"could not set targets: {e}")
                return
            self.model_ready.emit(det.vocabulary(), det.open_vocab)
            self.status.emit(" · ".join(warnings) if warnings else
                             f"{det.name} ready" + (f" · {len(examples)} example(s)" if examples else ""))

    def run(self) -> None:
        det = Detector()
        cap, seq = None, 0
        last_auto = time.monotonic()
        fps, n, t_fps = 0.0, 0, time.perf_counter()
        idle_sent = False
        while self._running:
            self._apply_pending(det)
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
                dets = det.predict(frame, s.conf, s.imgsz, s.fp16, s.example_conf, s.exact_thresh)
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
