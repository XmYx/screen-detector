"""Shared detection types and model helpers (used by the main detector and the example engine)."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset

from .config import MODELS_DIR

OPEN_VOCAB_HINTS = ("yoloe", "world")

# Dedicated models for words general detectors handle poorly (body parts score ~0.1 in YOLOE).
# word -> (weights, download url, box expansion factor applied to the specialist's boxes)
FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26s-face.pt"  # GPL-3.0
SPECIALISTS = {
    "face": ("yolo26s-face.pt", FACE_URL, 1.0),
    "faces": ("yolo26s-face.pt", FACE_URL, 1.0),
    "human face": ("yolo26s-face.pt", FACE_URL, 1.0),
    "head": ("yolo26s-face.pt", FACE_URL, 1.45),  # face box grown to cover the whole head
    "heads": ("yolo26s-face.pt", FACE_URL, 1.45),
}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    conf: float
    source: str = "model"  # model | text | example
    tags: str = ""         # which methods agreed, e.g. "V·M·✓"


@dataclass
class FrameResult:
    detections: list[Detection] = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    infer_ms: float = 0.0
    fps: float = 0.0


def is_open_vocab(name: str) -> bool:
    stem = Path(name).stem.lower()
    # "-pf" (prompt-free) models ship a fixed large vocabulary and are filtered like COCO models
    return any(h in stem for h in OPEN_VOCAB_HINTS) and not stem.endswith("-pf")


def ensure_weights(name: str, on_status=lambda msg: None, url: str | None = None) -> str:
    """Resolve a model name/path to a local file, downloading it into models/ when needed
    (official ultralytics weights by name, anything else from url)."""
    p = Path(name)
    if p.is_file():
        return str(p)
    local = MODELS_DIR / p.name
    if local.is_file():
        return str(local)
    if url:
        from ultralytics.utils.downloads import safe_download
        on_status(f"downloading {p.name} ...")
        safe_download(url=url, file=local, progress=False)
        return str(local)
    if p.name not in GITHUB_ASSETS_NAMES:
        raise FileNotFoundError(f"{name} is not in models/ and is not a downloadable ultralytics model")
    on_status(f"downloading {p.name} ...")
    return str(attempt_download_asset(local))


def tile_grid(w: int, h: int, n: int, overlap: float = 0.15) -> list[tuple[int, int, int, int]]:
    """n x n overlapping tiles (x1, y1, x2, y2) covering a w x h frame."""
    tw, th = int(w / n * (1 + overlap)), int(h / n * (1 + overlap))
    xs = np.linspace(0, w - tw, n).astype(int)
    ys = np.linspace(0, h - th, n).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def predict_boxes(model, frame: np.ndarray, conf: float, imgsz: int, device: str, fp16: bool,
                  classes=None, tiles: int = 0, agnostic: bool | None = None, max_det: int = 300):
    """Run a YOLO model on the frame (plus n x n zoomed tiles for small objects).

    Returns numpy (xyxy[N,4], conf[N], cls[N]) in frame pixel coordinates.
    """
    images, offsets = [frame], [(0, 0)]
    if tiles >= 2:
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in tile_grid(w, h, tiles):
            images.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
    kw = dict(conf=conf, imgsz=imgsz, classes=classes, device=device, max_det=max_det, verbose=False,
              quantize=16 if fp16 and device != "cpu" else None)
    if agnostic is not None:
        kw["agnostic_nms"] = agnostic
    results = model.predict(images if len(images) > 1 else frame, **kw)
    boxes, scores, cls = [], [], []
    for r, (ox, oy) in zip(results, offsets):
        if r.boxes is None or not len(r.boxes):
            continue
        b = r.boxes.xyxy.clone()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        boxes.append(b)
        scores.append(r.boxes.conf)
        cls.append(r.boxes.cls)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b, s, c = torch.cat(boxes), torch.cat(scores), torch.cat(cls)
    if len(images) > 1:  # merge duplicates between the full frame and overlapping tiles
        keep = batched_nms(b.float(), s.float(), torch.zeros_like(c) if agnostic else c, 0.5)
        b, s, c = b[keep], s[keep], c[keep]
    return b.cpu().numpy(), s.float().cpu().numpy(), c.cpu().numpy().astype(int)
