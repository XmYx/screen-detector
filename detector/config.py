import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
MODELS_DIR = ROOT / "models"
DATASET_DIR = ROOT / "dataset"

# Offered in the model picker; any extra *.pt dropped into models/ is listed too.
# Weights are downloaded by ultralytics on first use.
BUILTIN_MODELS = [
    "yolo26n.pt",           # COCO-80, fastest
    "yolo26s.pt",           # COCO-80
    "yolo26m.pt",           # COCO-80, default: best accuracy/speed on 4K screens (~10 ms @1280, RTX 4090)
    "yolo26l.pt",           # COCO-80, most accurate
    "yoloe-26s-seg.pt",     # open vocabulary: type anything in "Look for"
    "yoloe-26m-seg.pt",
    "yoloe-26l-seg.pt",
    "yoloe-26l-seg-pf.pt",  # prompt-free: tags thousands of object types with no "Look for" list
]


@dataclass
class Settings:
    model: str = "yolo26m.pt"
    targets: str = "cat"          # comma separated; empty = every class the model knows
    conf: float = 0.35
    imgsz: int = 1280            # screens are 2-4K; 640 shrinks them too much
    max_fps: int = 60
    fp16: bool = True
    paused: bool = False
    # detect-by-example (examples/)
    example_conf: float = 0.15    # "similar" visual-prompt scores run lower than text-prompt ones
    exact_thresh: float = 0.80    # correlation needed for an "exact look" template match
    example_color: str = "#ffb020"
    # overlay
    show_labels: bool = True
    show_conf: bool = True
    show_hud: bool = True
    box_color: str = "#00ff66"
    box_thickness: int = 2
    # capture
    backend: str = "auto"         # auto | pipewire | dxgi | mss
    monitor: int = 0
    restore_token: str = ""       # Wayland portal: skip the "share screen" dialog next time
    # labeler
    autosave_interval: float = 0.0  # seconds between automatic snapshots, 0 = off
    autosave_only_hits: bool = True
    # ui
    panel_visible: bool = True
    panel_collapsed: bool = False
    panel_opacity: float = 0.9    # resting opacity; the panel turns opaque while hovered
    panel_pos: list | None = None  # [x, y] of the panel's top-right corner

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    def target_list(self) -> list[str]:
        return [t.strip() for t in self.targets.split(",") if t.strip()]
