import os
import sys

from .base import BaseCapture


def available_backends() -> list[str]:
    if sys.platform == "win32":
        return ["auto", "dxgi", "mss"]
    if sys.platform.startswith("linux"):
        return ["auto", "pipewire", "mss"]
    return ["auto", "mss"]


def resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    if sys.platform == "win32":
        return "dxgi"
    if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
        return "pipewire"  # X11 grabbing returns black for native Wayland windows
    return "mss"


def create_capture(backend: str, monitor: int, max_fps: int, restore_token: str = "") -> BaseCapture:
    backend = resolve_backend(backend)
    if backend == "dxgi":
        try:
            from .dxgi import DXGICapture
            return DXGICapture(monitor, max_fps)
        except ImportError:
            backend = "mss"
    if backend == "pipewire":
        from .pipewire import PipeWireCapture
        return PipeWireCapture(restore_token)
    from .mss_backend import MSSCapture
    return MSSCapture(monitor, max_fps)
