from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import select
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any

import cv2
import numpy as np

from twopoint_project.contrl.target_center_servo import (
    AimUpdate,
    PointPrediction,
)
from twopoint_project.vision.inferencer import TargetCorners

DEFAULT_MONITOR_OUTPUT_DIR = Path("outputs")
WEBRTC_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>center_then_flash WebRTC</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #10131a;
      color: #f8fafc;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      background: #07090d;
    }
    video {
      width: 100%;
      height: 100vh;
      object-fit: contain;
      background: #000;
    }
    aside {
      padding: 16px;
      border-left: 1px solid #29313d;
      background: #151a23;
      overflow: auto;
    }
    h1 {
      margin: 0 0 16px;
      font-size: 18px;
    }
    dl {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px 10px;
      margin: 0;
      font-size: 13px;
    }
    dt { color: #94a3b8; }
    dd { margin: 0; overflow-wrap: anywhere; }
    pre {
      margin: 16px 0 0;
      padding: 12px;
      border: 1px solid #29313d;
      border-radius: 6px;
      background: #0b1018;
      color: #dbeafe;
      font-size: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 860px) {
      body { display: block; }
      video { height: auto; aspect-ratio: 4 / 3; }
      aside { border-left: 0; border-top: 1px solid #29313d; }
    }
  </style>
</head>
<body>
  <video id="video" autoplay playsinline muted></video>
  <aside>
    <h1>center_then_flash</h1>
    <dl>
      <dt>连接</dt><dd id="connection">idle</dd>
      <dt>ICE</dt><dd id="ice">idle</dd>
      <dt>backend</dt><dd id="backend">-</dd>
      <dt>providers</dt><dd id="providers">-</dd>
      <dt>frame</dt><dd id="frame">-</dd>
      <dt>target</dt><dd id="target">-</dd>
      <dt>reason</dt><dd id="reason">-</dd>
    </dl>
    <pre id="status">{}</pre>
  </aside>
  <script>
    const video = document.getElementById("video");
    const connection = document.getElementById("connection");
    const ice = document.getElementById("ice");
    const backend = document.getElementById("backend");
    const providers = document.getElementById("providers");
    const frame = document.getElementById("frame");
    const target = document.getElementById("target");
    const reason = document.getElementById("reason");
    const status = document.getElementById("status");
    let pc = null;

    function waitForIceGatheringComplete(peer) {
      if (peer.iceGatheringState === "complete") {
        return Promise.resolve();
      }
      return new Promise((resolve) => {
        function checkState() {
          if (peer.iceGatheringState === "complete") {
            peer.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }
        }
        peer.addEventListener("icegatheringstatechange", checkState);
      });
    }

    function renderStatus(message) {
      backend.textContent = message.backend || "-";
      providers.textContent = (message.providers || []).join(", ") || "-";
      frame.textContent = message.frame_id ?? "-";
      reason.textContent = message.reason || (message.valid ? "ok" : "-");
      if (message.target) {
        target.textContent = `${message.target.x.toFixed(3)}, ${message.target.y.toFixed(3)} / ${message.target.confidence.toFixed(2)}`;
      } else {
        target.textContent = "-";
      }
      status.textContent = JSON.stringify(message, null, 2);
    }

    async function start() {
      pc = new RTCPeerConnection({ iceServers: [] });
      const channel = pc.createDataChannel("status");
      channel.onmessage = (event) => renderStatus(JSON.parse(event.data));
      window.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && channel.readyState === "open") {
          channel.send(JSON.stringify({ type: "stop" }));
        }
      });
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.ontrack = (event) => { video.srcObject = event.streams[0]; };
      pc.onconnectionstatechange = () => { connection.textContent = pc.connectionState; };
      pc.oniceconnectionstatechange = () => { ice.textContent = pc.iceConnectionState; };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGatheringComplete(pc);
      const response = await fetch("/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pc.localDescription),
      });
      await pc.setRemoteDescription(await response.json());
    }

    window.addEventListener("beforeunload", () => {
      if (pc) {
        pc.close();
      }
    });
    start().catch((error) => {
      connection.textContent = "failed";
      status.textContent = String(error.stack || error);
    });
  </script>
</body>
</html>
"""


def default_monitor_output_path(mode: str = "center_then_flash") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_MONITOR_OUTPUT_DIR / f"{mode}_{stamp}.mp4"


def raw_monitor_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}")


def normalized_to_pixel(point: PointPrediction, width: int, height: int) -> tuple[int, int]:
    x = int(round(point["x"] * max(width - 1, 1)))
    y = int(round(point["y"] * max(height - 1, 1)))
    return x, y


def draw_text(image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_aim_frame(
    frame_bgr: Any,
    points: list[PointPrediction],
    update: AimUpdate,
    target_corners_normalized: TargetCorners = (),
    raw_target_center: PointPrediction | None = None,
) -> Any:
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    colors = {
        "target_center": (0, 220, 0),
        "laser_point": (0, 0, 255),
    }
    confidence_lines: list[tuple[str, tuple[int, int, int]]] = []
    valid_laser_pixel: tuple[int, int] | None = None
    valid_target_pixel: tuple[int, int] | None = None
    for point in points:
        label = str(point["label"])
        color = colors.get(label, (0, 200, 255))
        x, y = normalized_to_pixel(point, width, height)
        valid = label == "target_center" or point["confidence"] > 0.0
        confidence_lines.append((f"{label}: conf={point['confidence']:.2f}", color))
        radius = 4 if valid else 2
        outline_radius = 5 if valid else 4
        thickness = -1 if valid else 1
        cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x, y), outline_radius, (255, 255, 255), 1, cv2.LINE_AA)
        if label == "target_center" and valid:
            valid_target_pixel = (x, y)
        elif label == "laser_point" and valid:
            valid_laser_pixel = (x, y)

    if raw_target_center is not None:
        raw_x, raw_y = normalized_to_pixel(raw_target_center, width, height)
        cv2.circle(image, (raw_x, raw_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

    for corner_x, corner_y in target_corners_normalized:
        corner = (
            int(round(corner_x * max(width - 1, 1))),
            int(round(corner_y * max(height - 1, 1))),
        )
        cv2.circle(image, corner, 3, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(image, corner, 4, (255, 255, 255), 1, cv2.LINE_AA)

    if valid_target_pixel is not None:
        arrow_start = valid_laser_pixel or center
        cv2.arrowedLine(
            image,
            arrow_start,
            valid_target_pixel,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.12,
        )

    for index, (text, color) in enumerate(confidence_lines):
        draw_text(image, text, (14, 54 + index * 22), color)

    if update.step is not None:
        draw_text(
            image,
            "err=({:+.4f},{:+.4f}) step=({:+.3f},{:+.3f}) settled={}".format(
                update.step.error.x,
                update.step.error.y,
                update.step.x_delta_deg,
                update.step.y_delta_deg,
                update.settled,
            ),
            (14, 26),
            (0, 255, 255),
        )
    else:
        draw_text(image, f"invalid: {update.reason}", (14, 26), (0, 0, 255))
    return image


def dataclass_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value


def captured_age_seconds(captured: Any, now_monotonic: float | None = None) -> float:
    now = time.monotonic() if now_monotonic is None else now_monotonic
    captured_ns = getattr(captured, "captured_at_monotonic_ns", None)
    if captured_ns is not None:
        return max(now - int(captured_ns) / 1_000_000_000.0, 0.0)
    timestamp = getattr(captured, "timestamp", None)
    if timestamp is not None:
        return max(now - float(timestamp), 0.0)
    return 0.0


def local_urls(host: str, port: int) -> list[str]:
    if host not in {"0.0.0.0", "::"}:
        return [f"http://{host}:{port}/"]

    hosts = {"127.0.0.1"}
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            address = item[4][0]
            if not address.startswith("127."):
                hosts.add(address)
    except OSError:
        pass
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=1.0)
        for address in output.split():
            if "." in address and not address.startswith("127."):
                hosts.add(address)
    except (OSError, subprocess.SubprocessError):
        pass
    return [f"http://{address}:{port}/" for address in sorted(hosts)]


class VideoRecorder:
    def __init__(self, output_path: Path, fps: float, label: str) -> None:
        self.output_path = output_path
        self.fps = fps
        self.label = label
        self.writer: cv2.VideoWriter | None = None
        self.frame_count = 0

    def write(self, frame_bgr: Any) -> None:
        if self.writer is None:
            height, width = frame_bgr.shape[:2]
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.output_path}")
            print(f"monitor: recording {self.label} video to {self.output_path}")

        self.writer.write(frame_bgr)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class LatestAnnotatedFrame:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame_bgr: Any | None = None
        self._status: dict[str, Any] = {"type": "vision_status", "reason": "waiting_for_frame"}

    def publish(self, frame_bgr: Any, status: dict[str, Any]) -> None:
        with self._condition:
            self._frame_bgr = frame_bgr.copy()
            self._status = dict(status)
            self._condition.notify_all()

    def snapshot(self) -> tuple[Any, dict[str, Any]]:
        with self._condition:
            if self._frame_bgr is None:
                return self._waiting_frame(), dict(self._status)
            return self._frame_bgr.copy(), dict(self._status)

    @staticmethod
    def _waiting_frame() -> Any:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(image, "waiting for center_then_flash frame", (28, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image


class WebRtcStatusBroadcaster:
    def __init__(self) -> None:
        self._channels: set[Any] = set()

    def add(self, channel: Any) -> None:
        self._channels.add(channel)

    def discard(self, channel: Any) -> None:
        self._channels.discard(channel)

    def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False)
        for channel in tuple(self._channels):
            if channel.readyState == "open":
                channel.send(payload)
            elif channel.readyState in {"closing", "closed"}:
                self._channels.discard(channel)


class CenterWebRtcServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        frame_buffer: LatestAnnotatedFrame,
        stop_event: threading.Event,
        task_name: str,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.task_name = task_name
        self.broadcaster = WebRtcStatusBroadcaster()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: Any | None = None
        self._pcs: set[Any] = set()
        self._started = threading.Event()
        self._startup_error: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_thread, name="center-webrtc-server", daemon=True)
        self._thread.start()
        self._started.wait(timeout=5.0)
        if self._startup_error is not None:
            raise RuntimeError("failed to start center WebRTC server") from self._startup_error

    def stop(self) -> None:
        if self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop)
        try:
            future.result(timeout=5.0)
        except Exception as exc:
            print(f"monitor: WebRTC cleanup failed: {exc}")
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_async())
            self._started.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _start_async(self) -> None:
        from aiohttp import web
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from aiortc.rtcconfiguration import RTCConfiguration
        from av import VideoFrame

        frame_buffer = self.frame_buffer
        broadcaster = self.broadcaster
        pcs = self._pcs

        class LatestFrameTrack(VideoStreamTrack):
            kind = "video"

            async def recv(self) -> Any:
                pts, time_base = await self.next_timestamp()
                frame_bgr, status = frame_buffer.snapshot()
                broadcaster.send(status)
                frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                frame.pts = pts
                frame.time_base = time_base
                return frame

        async def index(_: Any) -> Any:
            page = WEBRTC_INDEX_HTML.replace("center_then_flash", self.task_name)
            return web.Response(text=page, content_type="text/html")

        async def health(_: Any) -> Any:
            _, status = frame_buffer.snapshot()
            return web.json_response({"ok": True, "latest": status})

        async def offer(request: Any) -> Any:
            params = await request.json()
            pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
            pcs.add(pc)

            @pc.on("datachannel")
            def on_datachannel(channel: Any) -> None:
                if channel.label != "status":
                    return
                broadcaster.add(channel)

                @channel.on("close")
                def on_close() -> None:
                    broadcaster.discard(channel)

                @channel.on("message")
                def on_message(message: str) -> None:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        return
                    if isinstance(data, dict) and data.get("type") == "stop":
                        self.stop_event.set()

            @pc.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                if pc.connectionState in {"failed", "closed"}:
                    await pc.close()
                    pcs.discard(pc)

            pc.addTrack(LatestFrameTrack())
            await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_get("/health", health)
        app.router.add_post("/offer", offer)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print("monitor: WebRTC URLs:")
        for url in local_urls(self.host, self.port):
            print(f"  {url}")

    async def _cleanup(self) -> None:
        coros = [pc.close() for pc in self._pcs]
        if coros:
            await asyncio.gather(*coros)
        self._pcs.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


class CenterRunMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        output_path: Path,
        fps: float,
        save_raw_video: bool,
        webrtc_host: str,
        webrtc_port: int,
        backend: str,
        providers: list[str],
        task_name: str = "center_then_flash",
    ) -> None:
        self.enabled = enabled
        self.output_path = output_path
        self.raw_output_path = raw_monitor_output_path(output_path)
        self.fps = fps
        self.save_raw_video = save_raw_video
        self.webrtc_host = webrtc_host
        self.webrtc_port = webrtc_port
        self.backend = backend
        self.providers = providers
        self.task_name = task_name
        self._active = False
        self.frame_buffer = LatestAnnotatedFrame()
        self.stop_event = threading.Event()
        self.recorder = VideoRecorder(output_path, fps, "annotated")
        self.raw_recorder = VideoRecorder(self.raw_output_path, fps, "raw") if save_raw_video else None
        self.server = CenterWebRtcServer(
            host=webrtc_host,
            port=webrtc_port,
            frame_buffer=self.frame_buffer,
            stop_event=self.stop_event,
            task_name=task_name,
        )

    def __enter__(self) -> CenterRunMonitor:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def start(self) -> None:
        if not self.enabled or self._active:
            return
        self.server.start()
        self._active = True

    def close(self) -> None:
        if not self._active:
            return
        self.server.stop()
        self.recorder.close()
        print(f"monitor: saved {self.recorder.frame_count} frame(s) to {self.output_path}")
        if self.raw_recorder is not None:
            self.raw_recorder.close()
            print(f"monitor: saved {self.raw_recorder.frame_count} raw frame(s) to {self.raw_output_path}")
        self._active = False

    def on_frame(
        self,
        captured: Any,
        points: list[PointPrediction],
        update: AimUpdate,
        target_corners_normalized: TargetCorners = (),
        raw_target_center: PointPrediction | None = None,
    ) -> None:
        if not self.enabled:
            return

        annotated = draw_aim_frame(
            captured.frame_bgr,
            points,
            update,
            target_corners_normalized,
            raw_target_center,
        )
        status = {
            "type": "vision_status",
            "task": self.task_name,
            "backend": self.backend,
            "providers": self.providers,
            "frame_id": captured.frame_id,
            "timestamp": captured.timestamp,
            "stream_generation": getattr(captured, "stream_generation", 0),
            "pts_ns": getattr(captured, "pts_ns", None),
            "age_ms": round(captured_age_seconds(captured) * 1000.0, 1),
            "points": points,
            "raw_target_center": raw_target_center,
            "target_corners_normalized": target_corners_normalized,
            "valid": update.valid,
            "moved": update.moved,
            "settled": update.settled,
            "reason": update.reason,
            "target": dataclass_to_dict(update.target),
            "step": dataclass_to_dict(update.step),
        }
        self.recorder.write(annotated)
        if self.raw_recorder is not None:
            self.raw_recorder.write(captured.frame_bgr)
        self.frame_buffer.publish(annotated, status)

    def stop_requested(self) -> bool:
        return self.stop_event.is_set()


class EscKeyStopper:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> EscKeyStopper:
        if self.enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("track: press Esc to stop safely")
        else:
            print("track: stdin is not a TTY; press Ctrl+C to stop safely")
        return self

    def __exit__(self, *args: object) -> None:
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def should_stop(self) -> bool:
        if not self.enabled:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return False
        return sys.stdin.read(1) == "\x1b"
