"""Labeler: saves captured frames with the current detections as a YOLO dataset.

dataset/
  images/20261002_181500_123.jpg
  labels/20261002_181500_123.txt   # "<class_id> <cx> <cy> <w> <h>" normalised 0..1
  classes.txt                      # class names, line number = class id (stable across runs)
  data.yaml                        # ready for: yolo train data=dataset/data.yaml model=yolo26s.pt

Labels are model predictions, so review/correct them (CVAT, Label Studio, ... all
read YOLO format) before training a custom model on them.
"""
from datetime import datetime
from pathlib import Path

import cv2


class DatasetWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.images = root / "images"
        self.labels = root / "labels"
        self.classes_file = root / "classes.txt"
        self.classes: list[str] = []
        if self.classes_file.exists():
            self.classes = [c for c in self.classes_file.read_text().splitlines() if c.strip()]

    def count(self) -> int:
        return len(list(self.images.glob("*.jpg"))) if self.images.exists() else 0

    def _class_id(self, name: str) -> int:
        if name not in self.classes:
            self.classes.append(name)
        return self.classes.index(name)

    def save(self, frame, detections) -> Path:
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_path = self.images / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = frame.shape[:2]
        n_classes = len(self.classes)
        lines = []
        for d in detections:
            cx, cy = (d.x1 + d.x2) / 2 / w, (d.y1 + d.y2) / 2 / h
            bw, bh = (d.x2 - d.x1) / w, (d.y2 - d.y1) / h
            lines.append(f"{self._class_id(d.label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (self.labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if len(self.classes) != n_classes or not (self.root / "data.yaml").exists():
            self._write_meta()
        return img_path

    def _write_meta(self) -> None:
        self.classes_file.write_text("\n".join(self.classes) + "\n")
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(self.classes))
        (self.root / "data.yaml").write_text(
            f"path: {self.root.as_posix()}\ntrain: images\nval: images\nnames:\n{names}\n"
        )
