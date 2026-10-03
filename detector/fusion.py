"""Runs several detection methods at once and fuses them into one score per object.

Sources per label (any subset can fire for the same object):
  V  YOLOE visual prompt from each example image
  T  YOLOE text prompt (example label, or a "Look for" word the main model does not know)
  M  multi-scale template match
  Y  the main model, when the label is one of its classes
  F  a specialist model for words general detectors miss (face, head)
  ✓  DINOv2 similarity of the candidate to the label's example images (boost / veto)
  ≈  similarity search: broad proposals (vehicle, person, item, ...) that look like an example

Candidates of one label that overlap are clustered; every source's score is calibrated to a
probability and combined with a noisy-OR, so agreement between methods raises confidence and
a single weak hint stays low. A candidate that looks clearly unlike the examples is vetoed.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .detection import SPECIALISTS, Detection, ensure_weights, predict_boxes
from .examples import Example, TemplateMatcher, Verifier, visual_embeddings

EXAMPLE_YOLOE = "yoloe-26s-seg.pt"
LOW_CONF = 0.06          # YOLOE candidates are collected low and judged by fusion
TEMPLATE_THRESH = 0.62
GENERIC_LABELS = {"object", "thing", "item", "stuff", "target", "it"}
TAG = {"vp": "V", "text": "T", "tpl": "M", "model": "Y", "spec": "F"}
# broad categories whose boxes are offered to the similarity search ("prop" source)
PROPOSALS = ["object", "vehicle", "person", "animal", "item", "container", "building", "weapon", "plant", "sign"]
MAX_PROPOSALS = 40


def calibrate(source: str, s: float) -> float:
    """Raw method score -> rough probability that the object is really there (single-source cap < 1)."""
    if source == "tpl":
        return float(np.clip((s - 0.55) / 0.35, 0, 1)) * 0.85
    if source == "model":
        return min(1.0, s) * 0.9
    return float(np.clip(s / 0.45, 0, 1)) * 0.8  # YOLOE visual/text prompt confidences run low


def verify_prob(sim: float) -> float:
    # DINOv2 CLS similarity: unrelated ~0.1-0.3, same kind of object ~0.35-0.6, same object 0.7+
    return float(np.clip((sim - 0.25) / 0.45, 0, 1)) * 0.9


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Cluster:
    label: str
    boxes: list = field(default_factory=list)    # (box, p)
    sources: dict = field(default_factory=dict)  # source -> best p

    def add(self, box: np.ndarray, source: str, p: float) -> None:
        self.boxes.append((box, p))
        self.sources[source] = max(p, self.sources.get(source, 0.0))

    @property
    def best(self) -> np.ndarray:
        return max(self.boxes, key=lambda bp: bp[1])[0]

    @property
    def box(self) -> np.ndarray:
        w = np.array([p for _, p in self.boxes]) + 1e-6
        return (np.stack([b for b, _ in self.boxes]) * w[:, None]).sum(0) / w.sum()


class ExampleEngine:
    """Detects user examples and open-vocabulary words with every enabled method, then fuses."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.yoloe = None
        self.sources: list[str] = []          # per YOLOE class index: "vp" | "text"
        self.matcher: TemplateMatcher | None = None
        self.verifier: Verifier | None = None
        self.example_labels: set[str] = set()
        self.text_terms: set[str] = set()      # plain words (no examples): plain confidence threshold
        self.debug: list[tuple] = []           # last frame's clusters: (label, box, sources, similarity, fused)
        self.specialists: dict[str, tuple] = {}  # weights -> (YOLO model, [(label, expand)])
        self.similar_search = False
        self.prop_yoloe = None                 # separate YOLOE for broad proposals (one class per box,
                                               # so broad words would otherwise outrank the examples)

    @property
    def active(self) -> bool:
        return bool(self.sources or self.matcher or self.specialists)

    def configure(self, examples: list[Example], text_terms: list[str], s, on_status) -> list[str]:
        """(Re)build all matchers. text_terms are "Look for" words the main model cannot detect."""
        notes = []
        self.example_labels = {e.label for e in examples}
        # words with a dedicated model (face, head) go there; the rest become YOLOE text prompts
        wanted = list(dict.fromkeys(list(text_terms) + sorted(self.example_labels)))
        spec_words = [w for w in wanted if w.lower() in SPECIALISTS]
        loaded = {}
        for w in spec_words:
            weights, url, expand = SPECIALISTS[w.lower()]
            if weights not in loaded:
                from ultralytics import YOLO
                model = self.specialists.get(weights, (None,))[0] or YOLO(ensure_weights(weights, on_status, url))
                loaded[weights] = (model, [])
            loaded[weights][1].append((w, expand))
        self.specialists = loaded
        self.text_terms = set(text_terms) - self.example_labels - set(spec_words)
        vp_examples = [e for e in examples if e.uses("vp")] if s.ex_vp else []
        text = list(self.text_terms)
        if s.ex_text:
            text += sorted({e.label for e in examples if e.uses("text") and e.label.lower() not in GENERIC_LABELS
                            and e.label not in spec_words})
        self.similar_search = bool(examples) and s.ex_similar and s.ex_verify
        if self.similar_search and self.prop_yoloe is None:
            from ultralytics import YOLOE
            self.prop_yoloe = YOLOE(ensure_weights(EXAMPLE_YOLOE, on_status))
            self.prop_yoloe.set_classes(PROPOSALS)
        names, embs, sources = [], [], []
        if vp_examples or text:
            if self.yoloe is None:
                from ultralytics import YOLOE
                on_status(f"loading {EXAMPLE_YOLOE} for examples / open-vocabulary words ...")
                self.yoloe = YOLOE(ensure_weights(EXAMPLE_YOLOE, on_status))
            if vp_examples:
                on_status(f"learning {len(vp_examples)} example image(s) ...")
            for label, emb in visual_embeddings(self.yoloe, vp_examples, self.device):
                names.append(label)
                embs.append(emb[None])
                sources.append("vp")
            if text:
                embs.append(self.yoloe.get_text_pe(text).float())
                names += text
                sources += ["text"] * len(text)
            self.yoloe.set_classes(names, torch.cat([e.to(self.device) for e in embs], dim=1))
        self.sources = sources
        tpl_examples = [e for e in examples if e.uses("template")] if s.ex_template else []
        self.matcher = TemplateMatcher(tpl_examples, self.device) if tpl_examples else None
        if s.ex_verify and examples:
            if self.verifier is None:
                on_status("loading DINOv2 verifier ...")
                try:
                    self.verifier = Verifier(self.device)
                except Exception as e:  # optional: everything else still works without it
                    notes.append(f"verifier unavailable: {e}")
            if self.verifier is not None:
                self.verifier.set_references(examples)
        elif self.verifier is not None:
            self.verifier.refs = {}
        return notes

    def detect(self, frame: np.ndarray, s, model_dets: list[Detection]) -> list[Detection]:
        out: list[Detection] = []
        cands: list[tuple[str, np.ndarray, str, float]] = []  # label, box, source, p
        proposals: list[tuple[np.ndarray, float]] = []
        tiles = 2 if s.small_objects else 0
        if self.sources:
            xyxy, conf, cls = predict_boxes(self.yoloe, frame, LOW_CONF, s.imgsz, self.device, s.fp16,
                                            tiles=tiles, agnostic=False)
            for b, c, k in zip(xyxy, conf, cls):
                label, src = self.yoloe.names[k], self.sources[k]
                if label in self.text_terms:
                    if c >= s.conf:
                        out.append(Detection(*map(float, b), label, float(c), "text", "T"))
                else:
                    cands.append((label, b, src, calibrate(src, float(c))))
        if self.similar_search:
            xyxy, conf, _ = predict_boxes(self.prop_yoloe, frame, LOW_CONF, s.imgsz, self.device, s.fp16,
                                          tiles=tiles, agnostic=True)
            proposals = list(zip(xyxy, conf.tolist()))
        for model, words in self.specialists.values():
            xyxy, conf, _ = predict_boxes(model, frame, min(s.conf, 0.25), s.imgsz, self.device, s.fp16, tiles=tiles)
            for label, expand in words:
                for b, c in zip(xyxy, conf):
                    b = self._expand(b, expand, frame.shape)
                    if label in self.example_labels:
                        cands.append((label, b, "spec", calibrate("model", float(c))))
                    elif c >= s.conf:
                        out.append(Detection(*map(float, b), label, float(c), "text", "F"))
        if self.matcher:
            for x1, y1, x2, y2, label, v in self.matcher.match(frame, TEMPLATE_THRESH):
                cands.append((label, np.array([x1, y1, x2, y2]), "tpl", calibrate("tpl", v)))
        for d in model_dets:
            if d.label in self.example_labels:
                cands.append((d.label, np.array([d.x1, d.y1, d.x2, d.y2]), "model", calibrate("model", d.conf)))
        if not cands and not proposals:
            return out

        clusters: list[Cluster] = []
        for label, box, src, p in sorted(cands, key=lambda c: -c[3]):
            for cl in clusters:
                if cl.label == label and iou(cl.best, box) >= 0.4:
                    cl.add(box, src, p)
                    break
            else:
                cl = Cluster(label)
                cl.add(box, src, p)
                clusters.append(cl)

        self.debug = []
        verify = self.verifier is not None and bool(self.verifier.refs)
        sims = self.verifier.similarity(frame, [(c.label, c.box.tolist()) for c in clusters]) if verify else []
        for i, cl in enumerate(clusters):
            probs = list(cl.sources.values())
            tags = [TAG[k] for k in ("vp", "text", "tpl", "model") if k in cl.sources]
            if verify:
                probs.append(verify_prob(sims[i]))
                if sims[i] >= 0.6:
                    tags.append("✓")
            fused = 1.0 - float(np.prod([1.0 - p for p in probs]))
            if verify and sims[i] < 0.3:  # looks clearly unlike every example
                fused *= 0.35
            self.debug.append((cl.label, cl.box.astype(int).tolist(), dict(cl.sources), sims[i] if verify else None, fused))
            if fused >= s.match_thresh:
                out.append(Detection(*map(float, cl.box), cl.label, fused, "example", "·".join(tags)))
        if self.similar_search and proposals:
            out += self._similar(frame, s, proposals, clusters)
        return out

    def _similar(self, frame, s, proposals, clusters) -> list[Detection]:
        """Broad-category boxes that look enough like one of the examples (other variants, models, colours)."""
        taken = [cl.best for cl in clusters]
        fresh: list[np.ndarray] = []
        for b, _ in sorted(proposals, key=lambda bc: -bc[1]):
            if all(iou(b, t) < 0.4 for t in taken + fresh):
                fresh.append(b)
            if len(fresh) >= MAX_PROPOSALS:
                break
        out = []
        for b, (label, sim) in zip(fresh, self.verifier.best_label(frame, [x.tolist() for x in fresh])):
            fused = verify_prob(sim) if label is not None and sim >= s.similar_thresh else 0.0
            self.debug.append((label, b.astype(int).tolist(), {"prop": 0.0}, sim, fused))
            if fused >= s.match_thresh:
                out.append(Detection(*map(float, b), label, fused, "example", "≈"))
        return out

    @staticmethod
    def _expand(b: np.ndarray, f: float, shape) -> np.ndarray:
        if f == 1.0:
            return b
        cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f, (b[3] - b[1]) * f
        cy -= (b[3] - b[1]) * (f - 1) * 0.25  # grow mostly upwards (hair / top of the head)
        return np.array([max(0, cx - w / 2), max(0, cy - h / 2), min(shape[1], cx + w / 2), min(shape[0], cy + h / 2)])
