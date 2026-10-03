"""Detect-by-example: the user shows the detector what to find.

Each example is an image crop plus a box around the object, stored in examples/:
  examples/examples.json  [{"id", "label", "mode", "box": [x1, y1, x2, y2], "enabled"}]
  examples/<id>.png       the crop (object + surrounding context), box is in its pixel coords

Any number of examples can share a label; every one of them is used. Matching methods
(combined per label by fusion.ExampleEngine):
  visual prompt  YOLOE embedding of each example: finds look-alike objects, any pose/size
  label as text  YOLOE text prompt from the label name ("chest", "car")
  template       GPU colour normalised cross-correlation over a 0.3x-1.6x scale pyramid: icons,
                 markers, HUD/UI and sprites that look the same every time, also when smaller
  verifier       DINOv2 similarity between each candidate and the examples (boost or veto)
Example modes pick the methods: auto = all, similar = no template, exact = template only.
"""
import json
import math
import secrets
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import ROOT

EXAMPLES_DIR = ROOT / "examples"
MODES = ("auto", "similar", "exact")
CONTEXT = 0.6        # extra margin around the box stored with each example (fraction of box size)
MAX_SIDE = 1024      # stored crops are capped to this size
PROMPT_SIDE = 640    # crops are resized to this before computing the YOLOE embedding


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

    def uses(self, method: str) -> bool:
        if self.mode == "exact":
            return method in ("template", "verify")
        if self.mode == "similar":
            return method != "template"
        return True


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

    def active(self) -> list[Example]:
        return [e for e in self.items if e.enabled]


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
    """Colour normalised cross-correlation on the GPU over a wide scale range.

    Each template/scale pair is matched on the image-pyramid level where the template's short side
    is ~TARGET_SIDE px, so large scales stay cheap and small (far away) copies are still found.
    """

    WORK_WIDTH = 1920                                   # frames are downscaled to this width first
    SCALES = tuple(float(s) for s in np.geomspace(0.3, 1.6, 9))
    TARGET_SIDE = 20
    MIN_SIDE = 6
    LEVELS = 4

    def __init__(self, examples: list[Example], device: str) -> None:
        self.device = device
        self._tfft: dict[tuple, torch.Tensor] = {}  # cached template spectra per (template, scale, level size)
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
        base = torch.from_numpy(small).to(self.device).permute(2, 0, 1)[None].float() / 255  # 1,3,H,W
        levels: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        out = []
        for ti, (label, patch) in enumerate(self.templates):
            boxes, scores = [], []
            ph, pw = patch.shape[:2]
            for sc in self.SCALES:
                th0, tw0 = ph * f * sc, pw * f * sc
                if min(th0, tw0) < self.MIN_SIDE:
                    continue
                k = int(min(self.LEVELS - 1, max(0, math.floor(math.log2(min(th0, tw0) / self.TARGET_SIDE)))))
                if k not in levels:
                    img = F.avg_pool2d(base, 2 ** k) if k else base
                    levels[k] = (img, torch.fft.rfft2(img))  # one spectrum per level, shared by all templates
                img, spec = levels[k]
                d = 2 ** k
                th, tw = max(3, round(th0 / d)), max(3, round(tw0 / d))
                if th >= img.shape[2] or tw >= img.shape[3]:
                    continue
                key = (ti, sc, img.shape[2], img.shape[3])
                score = self._ncc(img, spec, key, patch, tw, th)
                peak = F.max_pool2d(score[None, None], 3, 1, 1)[0, 0]
                ys, xs = torch.nonzero((score >= thresh) & (score >= peak), as_tuple=True)
                if len(ys) > 500:
                    top = torch.topk(score[ys, xs], 500).indices
                    ys, xs = ys[top], xs[top]
                for y, x, v in zip(ys.tolist(), xs.tolist(), score[ys, xs].tolist()):
                    boxes.append([x * d / f, y * d / f, tw * d / f, th * d / f])
                    scores.append(v)
            if boxes:
                for i in np.array(cv2.dnn.NMSBoxes(boxes, scores, thresh, 0.3)).flatten():
                    x, y, bw, bh = boxes[i]
                    out.append((x, y, x + bw, y + bh, label, scores[i]))
        return out

    def _ncc(self, img: torch.Tensor, spec: torch.Tensor, key: tuple, patch: np.ndarray,
             tw: int, th: int) -> torch.Tensor:
        H, W = img.shape[2], img.shape[3]
        if key not in self._tfft:
            t = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(t).to(self.device).permute(2, 0, 1)[None].float() / 255
            t = t - t.mean(dim=(2, 3), keepdim=True)
            # correlation = convolution with the flipped template; done in the frequency domain
            self._tfft[key] = (torch.fft.rfft2(t.flip(2, 3), s=(H, W)), t.pow(2).sum().sqrt().clamp_min(1e-6))
        tspec, t_norm = self._tfft[key]
        n = th * tw
        num = torch.fft.irfft2((spec * tspec).sum(1), s=(H, W))[0, th - 1:, tw - 1:]
        # window means of I and I^2 (float32 is precise here: small-window averages, not long running sums)
        # separable: a row pass then a column pass costs th+tw per pixel instead of th*tw
        m = F.avg_pool2d(F.avg_pool2d(torch.cat([img, img * img], 1), (1, tw), stride=1), (th, 1), stride=1)[0]
        var = n * (m[3:] - m[:3] * m[:3]).sum(0)
        # floor = per-pixel std of ~0.02: flat screen regions cannot match a textured template
        var = var.clamp_min(3 * n * 0.02 ** 2)
        return (num / (var.sqrt() * t_norm)).clamp(-1, 1)


class Verifier:
    """DINOv2 image-similarity check: how much does a candidate crop look like the label's examples?"""

    MODEL = "facebook/dinov2-small"  # downloaded to the Hugging Face cache on first use
    SIDE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, device: str) -> None:
        from transformers import AutoModel

        self.device = device
        self.half = device != "cpu"
        model = AutoModel.from_pretrained(self.MODEL).to(device).eval()
        self.model = model.half() if self.half else model
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)
        self.refs: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def embed_boxes(self, image: np.ndarray, boxes, mirror: bool = False) -> torch.Tensor:
        """Embed square crops centred on each box (same framing for examples and candidates, so the
        aspect ratio is kept); cropping and resizing run on the GPU in one batch."""
        from torchvision.ops import roi_align

        if not len(boxes):
            return torch.empty(0, 384, device=self.device)
        if image is not getattr(self, "_img_src", None):  # one upload per frame
            self._img_src = image
            self._img = torch.from_numpy(image).to(self.device).permute(2, 0, 1)[None].float()[:, [2, 1, 0]] / 255
        img = self._img
        rois = []
        for x1, y1, x2, y2 in boxes:
            cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.08
            rois.append([0, cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
        x = roi_align(img, torch.tensor(rois, device=self.device, dtype=torch.float32),
                      (self.SIDE, self.SIDE), sampling_ratio=2, aligned=True)
        if mirror:
            x = torch.cat([x, x.flip(3)])
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x.half() if self.half else x).pooler_output.float()
        return F.normalize(out, dim=-1)

    def set_references(self, examples: list[Example]) -> None:
        per_label: dict[str, list[torch.Tensor]] = {}
        for ex in examples:
            per_label.setdefault(ex.label, []).append(self.embed_boxes(ex.image(), [ex.box], mirror=True))
        self.refs = {label: torch.cat(v) for label, v in per_label.items()}

    def best_label(self, frame: np.ndarray, boxes) -> list[tuple[str | None, float]]:
        """For unlabeled boxes: the most similar label and its similarity."""
        if not len(boxes) or not self.refs:
            return [(None, 0.0)] * len(boxes)
        emb = self.embed_boxes(frame, boxes)
        labels = list(self.refs)
        sims = torch.stack([(self.refs[l] @ emb.T).max(0).values for l in labels])  # L, N
        best = sims.max(0)
        return [(labels[i], float(v)) for i, v in zip(best.indices.tolist(), best.values.tolist())]

    def similarity(self, frame: np.ndarray, items: list[tuple[str, list[float]]]) -> list[float]:
        """Best cosine similarity of each (label, box) crop to that label's examples."""
        keep = [i for i, (label, _) in enumerate(items) if label in self.refs]
        sims = [0.0] * len(items)
        if keep:
            emb = self.embed_boxes(frame, [items[i][1] for i in keep])
            for row, i in enumerate(keep):
                sims[i] = float((self.refs[items[i][0]] @ emb[row]).max())
        return sims
