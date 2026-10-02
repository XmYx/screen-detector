#!/usr/bin/env python3
"""Self-bootstrapping launcher (stdlib only).

Creates ./.venv on first run, installs PyTorch (CUDA build when an NVIDIA GPU
is present) plus requirements.txt, then relaunches the app inside the venv.
Dependencies are re-synced automatically whenever requirements.txt changes.

    python run.py              # start the detector
    python run.py --reinstall  # force a dependency re-sync
    python run.py --setup-only # just prepare the venv
"""
import hashlib
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS = ROOT / "requirements.txt"
STAMP = VENV / ".deps-stamp"
IS_WIN = os.name == "nt"
# Linux: reuse the distro's PyGObject/GStreamer bindings (needed for Wayland PipeWire capture).
SYSTEM_SITE = sys.platform.startswith("linux")
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"  # covers RTX 20xx..50xx


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def has_nvidia() -> bool:
    if os.environ.get("DETECTOR_CPU") == "1":
        return False
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        return subprocess.run([smi, "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def deps_signature(cuda: bool) -> str:
    h = hashlib.sha256(REQS.read_bytes())
    h.update(f"cuda={cuda};py={sys.version_info[:2]};sys={SYSTEM_SITE}".encode())
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def bootstrap(force: bool) -> None:
    py = venv_python()
    uv = shutil.which("uv")
    if not py.exists():
        print(f"[setup] creating virtual environment in {VENV}")
        venv.EnvBuilder(with_pip=not uv, system_site_packages=SYSTEM_SITE).create(VENV)
    cuda = has_nvidia()
    sig = deps_signature(cuda)
    if not force and STAMP.exists() and STAMP.read_text().strip() == sig:
        return
    print(f"[setup] installing dependencies (torch: {'CUDA' if cuda else 'CPU'}) - first run can take a few minutes")
    if uv:
        base = [uv, "pip", "install", "--python", str(py)]
    else:
        base = [str(py), "-m", "pip", "install", "--disable-pip-version-check"]
        run(base + ["-q", "--upgrade", "pip"])
    torch_cmd = base + ["torch", "torchvision"]
    if cuda:
        torch_cmd += ["--index-url", CUDA_INDEX]
    run(torch_cmd)
    run(base + ["-r", str(REQS)])
    STAMP.write_text(sig)
    prefetch_model(py)
    print("[setup] done")


def prefetch_model(py: Path) -> None:
    """Download the configured detection model now so the first launch starts instantly."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from detector.config import Settings\n"
        "from detector.engine import ensure_weights\n"
        "m = Settings.load().model; print('[setup] model', m, '->', ensure_weights(m))"
    )
    if subprocess.call([str(py), "-c", code, str(ROOT)], cwd=ROOT) != 0:
        print("[setup] model prefetch failed; it will be retried when the app starts")


def main() -> int:
    args = sys.argv[1:]
    force = "--reinstall" in args
    setup_only = "--setup-only" in args
    args = [a for a in args if a not in ("--reinstall", "--setup-only")]
    if sys.version_info < (3, 10):
        print("Python 3.10+ is required")
        return 1
    if not in_venv():
        try:
            bootstrap(force)
        except subprocess.CalledProcessError as e:
            print(f"[setup] dependency install failed ({e}). Fix the error above and rerun, or use --reinstall.")
            return 1
        if setup_only:
            return 0
        cmd = [str(venv_python()), str(Path(__file__).resolve()), *args]
        if IS_WIN:  # execv on Windows does not replace the console process cleanly
            return subprocess.call(cmd)
        os.execv(cmd[0], cmd)
    if setup_only:
        return 0
    sys.path.insert(0, str(ROOT))
    from detector.app import main as app_main
    return app_main(args)


if __name__ == "__main__":
    sys.exit(main())
