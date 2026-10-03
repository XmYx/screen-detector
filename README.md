# Screen Detector

Live screen-capture object detector with a click-through overlay, a floating options
panel, detect-by-example (show it a picture of what to find) and a YOLO-format labeler.
Runs on Windows and Linux (X11 and Wayland). Built on Ultralytics YOLO26 / YOLOE and PyTorch.

## Run

Download the archive for your OS from [Releases](../../releases) (or clone), then:

```
Linux:    ./run.sh          (or: python3 run.py)
Windows:  run.bat           (or: py run.py)
```

Requires Python 3.10+. An NVIDIA GPU is recommended (CPU works, slowly).

First start creates `.venv/`, installs PyTorch (CUDA build if `nvidia-smi` is found;
set `DETECTOR_CPU=1` to force CPU) and `requirements.txt`, then launches. Uses `uv` when
available, plain `pip` otherwise. Editing `requirements.txt` triggers a re-sync on the
next start; `--reinstall` forces one, `--setup-only` prepares the venv without launching.

Linux/Wayland additionally needs the distro packages
`python3-gi python3-dbus gir1.2-gstreamer-1.0 gstreamer1.0-pipewire` (preinstalled on Ubuntu).

## Use

- The floating panel (top right, drag by its header) is semi-transparent and turns opaque on hover.
  The chevron, Esc or a double-click on the header collapses it into a pill that keeps showing
  fps / latency / hits. The ⋯ menu or tray icon hides it completely. The status dot is green when the
  capture is healthy, amber while waiting or loading, and red when frames are blank or capture failed.
- **Look for**: any comma-separated words, Enter to apply. Words the selected model knows (COCO
  `yolo26*`: person, car, cat, … hover the hint for the list) run on it; any other word is sent
  automatically to an open-vocabulary YOLOE model, so "chest, sword, barrel" work with any model.
  `face` and `head` use a dedicated face model (general detectors barely see body parts).
- **Model**: official weights are downloaded into `models/` automatically on first use (the configured
  one is pre-fetched during setup). Default `yolo26m` at input 1280 was the most accurate on real 4K
  screen captures (~10 ms on an RTX 4090). `yoloe-26*-seg-pf` tags thousands of object types with no list.
- **Input size**: bigger = smaller/farther objects found, slower. Screens are 2-4K, so keep 1280+.
  **Small objects** additionally scans zoomed 2x2 tiles (finds far away / tiny objects, ~2x slower).
- **Examples · show it what to find**: type a label, then *Pick from screen* (freezes the screen,
  drag a box), *Images…* (one or more pictures; drag a box or press Enter for the whole image) or
  *Paste*. Add **2-5 examples per label** from different angles, sizes and lighting: every example
  is used, and one picture of a car's side will not match its rear. Untick an example to disable it.
  All enabled methods run at the same time and their results are fused per object:

  | Tag | Method | Good for |
  |-----|--------|----------|
  | V | YOLOE visual prompt from each example | look-alike objects in any pose / size |
  | T | the label as a text prompt | generic things ("car", "chest") |
  | M | GPU template matching, 0.3x-1.6x scales | icons, markers, HUD, sprites, smaller copies |
  | Y / F | the main model / face model, when the label is one of their classes | |
  | ✓ | DINOv2 check that the candidate looks like the examples (boosts, or vetoes clear mismatches) | |
  | ≈ | similarity search: other objects (vehicles, people, items, …) that look like an example | variants, other colours/models |

  Agreement between methods raises the score (shown as e.g. `chest 0.93 V·M·✓`); *Match ≥* sets the
  fused score needed, *Similar ≥* how alike a similarity-search find must be. Each method can be
  switched off. With examples but no "Look for" words, only the examples are shown. Example modes:
  *Auto* = all methods, *Similar* = no template matching, *Exact look* = template matching only.
- **Overlay**: styles *Boxes, Corners, Neon, Markers* (pulsing targets), *Spotlight* (dims everything
  else) and *Heatmap* (detection probability as heat, with selectable palettes and a smooth fade),
  overall opacity, colour per label, labels/scores/HUD toggles.
- **Alerts**: *Sound* plays a pling when something new is detected (not on every frame: only when more
  objects are visible than in the last 1.5 s). Pick any WAV/MP3/OGG file (right-click the file button
  for the built-in sound), set volume and cooldown, and optionally limit it to some labels.
- **Labeler**: *Save snapshot* or *Auto-save every N s* writes `dataset/images`, `dataset/labels`
  (YOLO txt), `classes.txt` and `data.yaml`. Review the auto-labels, then train a custom model:
  `.venv/bin/yolo train data=dataset/data.yaml model=yolo26s.pt epochs=100` and drop the resulting
  `best.pt` into `models/` (it appears in the model list).

## Capture backends

| Platform        | Backend    | Notes |
|-----------------|------------|-------|
| Windows         | `dxgi`     | DXGI Desktop Duplication via dxcam. Overlay/panel are excluded from capture (Win10 2004+). |
| Linux Wayland   | `pipewire` | xdg-desktop-portal ScreenCast. Approve the share dialog once; the choice is remembered ("Choose screen…" resets it). |
| Linux X11 / any | `mss`      | Fallback. On Wayland it only sees XWayland windows (native windows are black). |

Every backend's frames are checked once per second: all-black or single-colour frames raise a red
warning in the panel and on the overlay HUD (e.g. `mss` on Wayland), and a frozen picture is flagged.

On Linux the overlay runs through XWayland (`QT_QPA_PLATFORM=xcb`) because native Wayland
clients cannot stay on top or be click-through. Linux cannot exclude the overlay from the
capture, so drawn boxes appear in the frames the model sees (outlines rarely trigger detections).

## Layout

```
run.py               stdlib bootstrapper (venv + deps), then detector.app.main
detector/app.py      controller: wires capture -> inference worker -> overlay/panel
detector/capture/    pipewire.py (Wayland), dxgi.py (Windows), mss_backend.py (fallback)
detector/engine.py   ultralytics model wrapper + inference QThread
detector/detection.py shared detection types, model download, tiled inference
detector/fusion.py   runs all methods at once and fuses them per object (+ similarity search)
detector/examples.py detect-by-example: storage, YOLOE visual prompts, FFT template matcher, DINOv2 verifier
detector/alerts.py   sound alert
detector/picker.py   box selection over a frozen frame / image
detector/overlay.py  transparent click-through box renderer
detector/panel.py    floating options panel (collapses into a stats pill)
detector/dataset.py  YOLO dataset writer
config.json          saved settings (created on exit)
models/              downloaded / custom weights
examples/            your detection examples

## License

AGPL-3.0, matching the Ultralytics dependency. The optional face model (downloaded on first use of
"face"/"head") is [akanametov/yolo-face](https://github.com/akanametov/yolo-face), GPL-3.0.
```
