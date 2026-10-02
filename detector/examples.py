"""Detect-by-example: the user shows the detector what to find.

Each example is an image crop plus a box around the object, stored in examples/:
  examples/examples.json  [{"id", "label", "mode", "box": [x1, y1, x2, y2], "enabled"}]
  examples/<id>.png       the crop (object + surrounding context), box is in its pixel coords

Two matching modes:
  similar  YOLOE visual-prompt embedding, one prompt per example. Generalises across poses/looks,
           works for object-like things (characters, chests, vehicles, animals). Needs a yoloe model.
  exact    GPU normalised cross-correlation (colour, multi-scale). For icons, markers, HUD/UI
           elements and sprites that always look the same. Works with any model.
"""
import json
import secrets
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import ROOT

EXAMPLES_DIR = ROOT / "examples"
MODES = ("similar", "exact")
CONTEXT = 0.6        # extra margin around the box stored with each example (fraction of box size)
MAX_SIDE = 1024      # stored crops are capped to this size
PROMPT_SIDE = 640    # crops are resized to this before computing the embedding


@dataclass
class Example:
    id: str
    label: str
    mode: str
    box: list
    enabled: bool = True

    @property
    def path(self):
        return EXAMPLES_DIR / f"{self.id}.png"

    def image(self) -> np.ndarray:
        img = cv2.imread(str(self.path))
        if img is None:
            raise FileNotFoundError(self.path)
        return img

    def patch(self) -> np.ndarray:
        x1, y1, x2, y2 = (int(round(v)) for v in self.box)
        return self.image()[y1:y2, x1:x2]


class ExampleStore:
    def __init__(self) -> None:
        self.file = EXAMPLES_DIR / "examples.json"
        self.items: list[Example] = []
        try:
            self.items = [Example(**e) for e in json.loads(self.file.read_text())]
        except (OSError, ValueError, TypeError):
            pass
        self.items = [e for e in self.items if e.path.exists()]

    def save(self) -> None:
        EXAMPLES_DIR.mkdir(exist_ok=True)
        self.file.write_text(json.dumps([asdict(e) for e in self.items], indent=2))

    def add(self, img: np.ndarray, box, label: str, mode: str) -> Example:
        """Store the box region of img (BGR) with some context around it."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            raise ValueError("selection is too small")
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - bw * CONTEXT)), int(max(0, y1 - bh * CONTEXT))
        cx2, cy2 = int(min(w, x2 + bw * CONTEXT)), int(min(h, y2 + bh * CONTEXT))
        crop = img[cy1:cy2, cx1:cx2]
        s = min(1.0, MAX_SIDE / max(crop.shape[:2]))
        if s < 1.0:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ex = Example(secrets.token_hex(4), label.strip() or "object", mode,
                     [(x1 - cx1) * s, (y1 - cy1) * s, (x2 - cx1) * s, (y2 - cy1) * s])
        EXAMPLES_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(ex.path), crop)
        self.items.append(ex)
        self.save()
        return ex

    def remove(self, ex_id: str) -> None:
        for e in [e for e in self.items if e.id == ex_id]:
            self.items.remove(e)
            e.path.unlink(missing_ok=True)
        self.save()

    def set_enabled(self, ex_id: str, enabled: bool) -> None:
        for e in self.items:
            if e.id == ex_id:
                e.enabled = enabled
        self.save()

    def labels(self) -> list[str]:
        return sorted({e.label for e in self.items})

    def active(self, mode: str | None = None) -> list[Example]:
        return [e for e in self.items if e.enabled and (mode is None or e.mode == mode)]


def visual_embeddings(yoloe, examples: list[Example], device: str) -> list[tuple[str, torch.Tensor]]:
    """One normalised YOLOE visual-prompt embedding (1, D) per example, paired with its label.

    Examples are kept as separate prompts rather than averaged: varied examples of one label
    (front/back view, day/night) blur into a weak mean, while separate prompts each match their look.
    """
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    out = []
    for ex in examples:
        img = ex.image()
        s = PROMPT_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        box = np.array([[v * s for v in ex.box]], dtype=np.float32)
        pred = YOLOEVPSegPredictor(overrides=dict(task="segment", mode="predict", save=False, verbose=False,
                                                  batch=1, imgsz=PROMPT_SIDE, device=device))
        pred.set_prompts({"bboxes": box, "cls": np.array([0])})
        yoloe.model.model[-1].nc = 1
        pred.setup_model(model=yoloe.model, verbose=False)
        out.append((ex.label, F.normalize(pred.get_vpe(img).reshape(1, -1).float(), dim=-1)))
    return out


class TemplateMatcher:
    """Colour normalised cross-correlation on the GPU, at a few scales, merged with NMS."""

    WORK_WIDTH = 1920            # frames are downscaled to this width before matching
    SCALES = (0.8, 1.0, 1.25)

    def __init__(self, examples: list[Example], device: str) -> None:
        self.device = device
        self.templates: list[tuple[str, np.ndarray]] = []
        for ex in examples:
            patch = ex.patch()
            if patch.size and min(patch.shape[:2]) >= 4:
                self.templates.append((ex.label, patch))

    def __bool__(self) -> bool:
        return bool(self.templates)

    def match(self, frame: np.ndarray, thresh: float) -> list[tuple[float, float, float, float, str, float]]:
        h, w = frame.shape[:2]
        f = min(1.0, self.WORK_WIDTH / w)
        small = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else frame
        img = torch.from_numpy(small).to(self.device).permute(2, 0, 1)[None].float() / 255  # 1,3,H,W
        # running sums along x of [I, I^2], shared by every template size (box sums via differences)
        # (float64: long float32 running sums lose the precision the variance needs)
        self._cx = F.pad(torch.cat([img, img * img], 1).double().cumsum(3), (1, 0))
        out = []
        for label, patch in self.templates:
            boxes, scores = [], []
            for sc in self.SCALES:
                # patches come from screen captures at ~native size; scale with the frame
                t = cv2.resize(patch, None, fx=f * sc, fy=f * sc, interpolation=cv2.INTER_AREA)
                th, tw = t.shape[:2]
                if th < 4 or tw < 4 or th >= img.shape[2] or tw >= img.shape[3]:
                    continue
                score = self._ncc(img, t)
                ys, xs = torch.nonzero(score >= thresh, as_tuple=True)
                if len(ys) > 2000:  # flat template; keep the best responses only
                    top = torch.topk(score[ys, xs], 2000).indices
                    ys, xs = ys[top], xs[top]
                for y, x, v in zip(ys.tolist(), xs.tolist(), score[ys, xs].tolist()):
                    boxes.append([x / f, y / f, tw / f, th / f])
                    scores.append(v)
            if boxes:
                for i in np.array(cv2.dnn.NMSBoxes(boxes, scores, thresh, 0.3)).flatten():
                    x, y, bw, bh = boxes[i]
                    out.append((x, y, x + bw, y + bh, label, scores[i]))
        return out

    def _ncc(self, img: torch.Tensor, tpl: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(tpl).to(self.device).permute(2, 0, 1)[None].float() / 255
        t = t - t.mean(dim=(2, 3), keepdim=True)
        t_norm = t.pow(2).sum().sqrt().clamp_min(1e-6)
        n = t.shape[2] * t.shape[3]
        num = F.conv2d(img, t)[0, 0]
        th, tw = t.shape[2], t.shape[3]
        rows = self._cx[..., tw:] - self._cx[..., :-tw]
        cy = F.pad(rows.cumsum(2), (0, 0, 1, 0))
        box = (cy[..., th:, :] - cy[..., :-th, :])[0]  # 6,H',W' window sums of I and I^2
        s1, s2 = box[:3], box[3:]
        # floor = per-pixel std of ~0.02: flat screen regions cannot match a textured template
        var = (s2 - s1 * s1 / n).sum(0).float().clamp_min(3 * n * 0.02 ** 2)
        return (num / (var.sqrt() * t_norm)).clamp(-1, 1)
