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
- **Look for**: comma-separated class names, Enter to apply. COCO models (`yolo26*`) know 80
  classes (hover the hint to list them). `yoloe-*` models are open-vocabulary: type anything
  ("treasure chest, health pack, person").
- **Model**: official weights are downloaded into `models/` automatically on first use (the configured
  one is pre-fetched during setup). Default `yolo26m` at input 1280 was the most accurate on real 4K
  screen captures (~10 ms on an RTX 4090). `yoloe-26*-seg-pf` tags thousands of object types with no list.
- **Input size**: bigger = smaller/farther objects found, slower. Screens are 2-4K, so keep 1280+.
- **Examples · show it what to find**: type a label, pick a mode, then *Pick from screen* (freezes
  the screen, drag a box), *Image…* (any picture, drag a box or press Enter for the whole image) or
  *Paste* (clipboard). Example hits are drawn in a separate colour. Add several examples per label
  (different angles, lighting) for better recall; untick an example to disable it.
  - *Similar (AI)*: YOLOE visual prompts find objects that look alike (characters, chests,
    vehicles, animals). Needs a `yoloe-*` model; combines with the "Look for" text.
  - *Exact look*: colour template matching on the GPU at several scales, for icons, minimap
    markers, HUD/UI elements and sprites that always look the same. Works with any model.
  - With examples but no "Look for" text, only the examples are detected.
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
detector/examples.py detect-by-example: storage, YOLOE visual prompts, GPU template matcher
detector/picker.py   box selection over a frozen frame / image
detector/overlay.py  transparent click-through box renderer
detector/panel.py    floating options panel (collapses into a stats pill)
detector/dataset.py  YOLO dataset writer
config.json          saved settings (created on exit)
models/              downloaded / custom weights
examples/            your detection examples

## License

AGPL-3.0, matching the Ultralytics dependency.
```
