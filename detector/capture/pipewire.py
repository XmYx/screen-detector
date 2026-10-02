"""Wayland screen capture via xdg-desktop-portal ScreenCast + PipeWire + GStreamer.

Needs the distro packages python3-gi, python3-dbus and gstreamer1.0-pipewire
(the venv is created with --system-site-packages so they are visible).
On first start the desktop shows a "share screen" dialog; the returned
restore token is saved so later runs start without asking.
"""
import secrets

import numpy as np

import dbus
import gi
from dbus.mainloop.glib import DBusGMainLoop

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from .base import BaseCapture  # noqa: E402

PORTAL = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST = "org.freedesktop.portal.ScreenCast"
REQUEST = "org.freedesktop.portal.Request"

SOURCE_MONITOR = 1
CURSOR_HIDDEN = 1
PERSIST_UNTIL_REVOKED = 2


class PortalError(RuntimeError):
    pass


class ScreenCastPortal:
    def __init__(self) -> None:
        DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SessionBus()
        self.sender = self.bus.get_unique_name()[1:].replace(".", "_")
        self.iface = dbus.Interface(self.bus.get_object(PORTAL, PORTAL_PATH), SCREENCAST)
        self.session = None

    def _call(self, method, *args, options: dict, timeout: int = 180) -> dict:
        """Invoke a portal method and wait for its Request.Response signal."""
        token = "det_" + secrets.token_hex(6)
        options = dict(options, handle_token=token)
        path = f"{PORTAL_PATH}/request/{self.sender}/{token}"
        loop = GLib.MainLoop()
        result: dict = {}

        def on_response(code, results):
            result["code"], result["results"] = int(code), results
            loop.quit()

        def on_timeout():
            result.setdefault("code", -1)
            loop.quit()
            return False

        match = self.bus.add_signal_receiver(on_response, "Response", REQUEST, PORTAL, path)
        try:
            method(*args, dbus.Dictionary(options, signature="sv"))
            tid = GLib.timeout_add_seconds(timeout, on_timeout)
            loop.run()
            if "results" in result:
                GLib.source_remove(tid)
        finally:
            match.remove()
        code = result.get("code", -1)
        if code != 0:
            raise PortalError({1: "screen sharing was cancelled", -1: "portal timed out"}.get(code, f"portal error {code}"))
        return result["results"]

    def open(self, restore_token: str = "") -> tuple[int, int, dict, str]:
        """Returns (pipewire_fd, node_id, stream_props, new_restore_token)."""
        res = self._call(self.iface.CreateSession,
                         options={"session_handle_token": "det_s" + secrets.token_hex(4)})
        self.session = str(res["session_handle"])
        opts = {
            "types": dbus.UInt32(SOURCE_MONITOR),
            "multiple": False,
            "cursor_mode": dbus.UInt32(CURSOR_HIDDEN),
            "persist_mode": dbus.UInt32(PERSIST_UNTIL_REVOKED),
        }
        if restore_token:
            opts["restore_token"] = restore_token
        self._call(self.iface.SelectSources, self.session, options=opts)
        res = self._call(self.iface.Start, self.session, "", options={})
        streams = res.get("streams") or []
        if not streams:
            raise PortalError("no stream selected")
        node_id, props = streams[0]
        fd = self.iface.OpenPipeWireRemote(self.session, dbus.Dictionary({}, signature="sv")).take()
        return fd, int(node_id), dict(props), str(res.get("restore_token", ""))

    def close(self) -> None:
        if self.session:
            try:
                self.bus.get_object(PORTAL, self.session).Close(dbus_interface="org.freedesktop.portal.Session")
            except dbus.DBusException:
                pass
            self.session = None


class PipeWireCapture(BaseCapture):
    name = "pipewire"

    def __init__(self, restore_token: str = "") -> None:
        super().__init__()
        self.restore_token = restore_token

    def _run(self) -> None:
        Gst.init(None)
        portal = ScreenCastPortal()
        self.status = "waiting for screen-share approval"
        pipeline = None
        try:
            fd, node, props, token = portal.open(self.restore_token)
            if token:
                self.restore_token = token
            if "position" in props and "size" in props:
                (x, y), (w, h) = props["position"], props["size"]
                self.region = (int(x), int(y), int(w), int(h))
            pipeline = Gst.parse_launch(
                f"pipewiresrc fd={fd} path={node} do-timestamp=true keepalive-time=500 always-copy=true "
                "! videoconvert n-threads=4 ! video/x-raw,format=BGRx "
                "! appsink name=sink max-buffers=1 drop=true sync=false"
            )
            sink = pipeline.get_by_name("sink")
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("could not start the GStreamer pipeline")
            self.status = "capturing"
            bus = pipeline.get_bus()
            while self._running:
                msg = bus.pop_filtered(Gst.MessageType.ERROR)
                if msg:
                    err, _ = msg.parse_error()
                    raise RuntimeError(f"GStreamer: {err.message}")
                sample = sink.emit("try-pull-sample", 200 * Gst.MSECOND)
                if sample is None:
                    continue
                self._publish(self._to_array(sample))
        finally:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
            portal.close()

    @staticmethod
    def _to_array(sample) -> np.ndarray:
        st = sample.get_caps().get_structure(0)  # no GstVideo typelib needed
        w, h = st.get_value("width"), st.get_value("height")
        buf = sample.get_buffer()
        stride = buf.get_size() // h
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("could not map buffer")
        try:
            raw = np.frombuffer(mapinfo.data, dtype=np.uint8, count=stride * h).reshape(h, stride)
            return np.ascontiguousarray(raw[:, : w * 4].reshape(h, w, 4)[:, :, :3])
        finally:
            buf.unmap(mapinfo)
