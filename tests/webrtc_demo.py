#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.append(str(REPO_ROOT))
    sys.path.append(str(REPO_ROOT / "src"))

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcconfiguration import RTCConfiguration
from av import VideoFrame

from tests.draw import NoopGimbal, draw_aim_frame
from dotenv import load_dotenv as load_project_dotenv
from twopoint_project.config import env_float, env_int, env_str
from twopoint_project.contrl.target_center_servo import AimUpdate, TargetCenterServo
from twopoint_project.tasks.resources import open_camera_capture
from twopoint_project.vision.inferencer import DEFAULT_IMG_SIZE, build_vision_inferencer
from twopoint_project.vision.pipeline import VisionProducer


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>twopoint WebRTC demo</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111318;
      color: #f1f5f9;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
    }
    main {
      min-width: 0;
      display: grid;
      place-items: center;
      background: #080a0f;
    }
    video {
      width: 100%;
      height: 100vh;
      object-fit: contain;
      background: #000;
    }
    aside {
      border-left: 1px solid #2a303a;
      padding: 16px;
      background: #151922;
      overflow: auto;
    }
    h1 {
      margin: 0 0 16px;
      font-size: 18px;
      font-weight: 650;
    }
    dl {
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 8px 12px;
      margin: 0 0 16px;
      font-size: 13px;
    }
    dt {
      color: #94a3b8;
    }
    dd {
      margin: 0;
      overflow-wrap: anywhere;
    }
    button {
      appearance: none;
      width: 100%;
      min-height: 38px;
      border: 1px solid #475569;
      border-radius: 6px;
      background: #e2e8f0;
      color: #0f172a;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary {
      margin-top: 8px;
      background: #1f2937;
      color: #f8fafc;
    }
    pre {
      margin: 16px 0 0;
      padding: 12px;
      border: 1px solid #2a303a;
      border-radius: 6px;
      background: #0b0f16;
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 860px) {
      body {
        display: block;
      }
      video {
        height: auto;
        aspect-ratio: 4 / 3;
      }
      aside {
        border-left: 0;
        border-top: 1px solid #2a303a;
      }
    }
  </style>
</head>
<body>
  <main>
    <video id="video" autoplay playsinline muted></video>
  </main>
  <aside>
    <h1>twopoint WebRTC</h1>
    <dl>
      <dt>连接</dt><dd id="connection">idle</dd>
      <dt>ICE</dt><dd id="ice">idle</dd>
      <dt>backend</dt><dd id="backend">-</dd>
      <dt>providers</dt><dd id="providers">-</dd>
      <dt>frame</dt><dd id="frame">-</dd>
      <dt>target</dt><dd id="target">-</dd>
      <dt>reason</dt><dd id="reason">-</dd>
    </dl>
    <button id="start">Start</button>
    <button id="stop" class="secondary">Stop</button>
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
      stop();
      pc = new RTCPeerConnection({ iceServers: [] });
      const channel = pc.createDataChannel("status");
      channel.onmessage = (event) => renderStatus(JSON.parse(event.data));
      pc.addTransceiver("video", { direction: "recvonly" });

      pc.ontrack = (event) => {
        video.srcObject = event.streams[0];
      };
      pc.onconnectionstatechange = () => {
        connection.textContent = pc.connectionState;
      };
      pc.oniceconnectionstatechange = () => {
        ice.textContent = pc.iceConnectionState;
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGatheringComplete(pc);

      const response = await fetch("/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pc.localDescription),
      });
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
    }

    function stop() {
      if (pc) {
        pc.close();
        pc = null;
      }
      connection.textContent = "closed";
      ice.textContent = "closed";
    }

    document.getElementById("start").addEventListener("click", start);
    document.getElementById("stop").addEventListener("click", stop);
    window.addEventListener("beforeunload", stop);
    start().catch((error) => {
      connection.textContent = "failed";
      status.textContent = String(error.stack || error);
    });
  </script>
</body>
</html>
"""


def load_demo_dotenv() -> Path | None:
    env_file = Path(env_str("WEBRTC_ENV_FILE", str(REPO_ROOT / ".env"))).expanduser()
    if not env_file.is_absolute():
        env_file = REPO_ROOT / env_file
    if not env_file.exists():
        return None

    try:
        load_project_dotenv(env_file)
    except TypeError:
        load_project_dotenv()
    return env_file


def parse_args() -> argparse.Namespace:
    loaded_env_file = load_demo_dotenv()
    parser = argparse.ArgumentParser(description="Serve annotated twopoint vision frames over WebRTC.")
    parser.set_defaults(loaded_env_file=str(loaded_env_file) if loaded_env_file else None)
    parser.add_argument("--host", default=env_str("WEBRTC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=env_int("WEBRTC_PORT", 8080))
    parser.add_argument("--camera", type=int, default=env_int("TWOPOINT_CAMERA_INDEX", 0))
    parser.add_argument("--camera-width", type=int, default=env_int("TWOPOINT_CAMERA_WIDTH", 640))
    parser.add_argument("--camera-height", type=int, default=env_int("TWOPOINT_CAMERA_HEIGHT", 480))
    parser.add_argument("--camera-fps", type=int, default=env_int("TWOPOINT_CAMERA_FPS", 30))
    parser.add_argument(
        "--vision-backend",
        choices=("onnx", "traditional"),
        default=env_str("TWOPOINT_VISION_BACKEND", "onnx").strip(),
    )
    parser.add_argument("--onnx", default=env_str("TWOPOINT_ONNX_PATH", "model-bin/best.onnx").strip())
    parser.add_argument("--img-size", type=int, default=env_int("TWOPOINT_IMG_SIZE", DEFAULT_IMG_SIZE))
    parser.add_argument("--center-x", type=float, default=env_float("CENTER_TARGET_X", 0.5))
    parser.add_argument("--center-y", type=float, default=env_float("CENTER_TARGET_Y", 0.5))
    parser.add_argument("--x-gain-deg", type=float, default=env_float("CENTER_X_GAIN_DEG", 8.0))
    parser.add_argument("--y-gain-deg", type=float, default=env_float("CENTER_Y_GAIN_DEG", -8.0))
    parser.add_argument("--max-step-deg", type=float, default=env_float("CENTER_MAX_STEP_DEG", 1.0))
    parser.add_argument("--deadband", type=float, default=env_float("CENTER_DEADBAND", 0.006))
    args = parser.parse_args()
    return args


def dataclass_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value


class StatusBroadcaster:
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


class DemoResources:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.capture = open_camera_capture(
            camera_index=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
        self.inferencer = build_vision_inferencer(
            backend=args.vision_backend,
            onnx_path=args.onnx,
            img_size=args.img_size,
        )
        self.vision = VisionProducer(capture=self.capture, inferencer=self.inferencer)
        self.servo = TargetCenterServo(
            center_x=args.center_x,
            center_y=args.center_y,
            x_gain_deg=args.x_gain_deg,
            y_gain_deg=args.y_gain_deg,
            max_step_deg=args.max_step_deg,
            deadband=args.deadband,
        )
        self.noop_gimbal = NoopGimbal()
        self.broadcaster = StatusBroadcaster()
        self.camera_error: str | None = None

    def start(self) -> None:
        try:
            self.capture.__enter__()
            self.vision.start()
            self.camera_error = None
        except Exception as exc:
            self.camera_error = str(exc)
            self.capture.__exit__(None, None, None)
            print(f"webrtc: camera_error={self.camera_error}")

    def stop(self) -> None:
        self.vision.stop()
        self.capture.__exit__(None, None, None)

    def read_annotated_frame(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self.camera_error is not None:
            raise RuntimeError(self.camera_error)

        vision_frame = self.vision.read_latest(timeout=1.0)
        update = self.servo.update(self.noop_gimbal, vision_frame.points)
        annotated = draw_aim_frame(
            vision_frame.captured.frame_bgr,
            vision_frame.points,
            update,
            vision_frame.target_corners_normalized,
            vision_frame.raw_target_center,
        )
        return annotated, self.status_for(vision_frame, update)

    def status_for(self, vision_frame: Any, update: AimUpdate) -> dict[str, Any]:
        captured = vision_frame.captured
        return {
            "type": "vision_status",
            "env_file": self.args.loaded_env_file,
            "backend": self.args.vision_backend,
            "providers": self.inferencer.providers,
            "camera": self.args.camera,
            "camera_error": self.camera_error,
            "frame_id": captured.frame_id,
            "timestamp": captured.timestamp,
            "age_ms": round((time.monotonic() - captured.timestamp) * 1000.0, 1),
            "points": vision_frame.points,
            "valid": update.valid,
            "moved": update.moved,
            "settled": update.settled,
            "reason": update.reason,
            "target": dataclass_to_dict(update.target),
            "step": dataclass_to_dict(update.step),
        }


class AnnotatedVisionTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, resources: DemoResources) -> None:
        super().__init__()
        self.resources = resources

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        loop = asyncio.get_running_loop()
        try:
            annotated, status = await loop.run_in_executor(None, self.resources.read_annotated_frame)
        except Exception as exc:
            annotated = self.error_frame(str(exc))
            status = {
                "type": "vision_status",
                "backend": self.resources.args.vision_backend,
                "providers": self.resources.inferencer.providers,
                "camera": self.resources.args.camera,
                "camera_error": str(exc),
                "valid": False,
                "reason": str(exc),
            }

        self.resources.broadcaster.send(status)
        frame = VideoFrame.from_ndarray(annotated, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    @staticmethod
    def error_frame(message: str) -> np.ndarray:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(image, "vision frame unavailable", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, message[:72], (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        return image


async def index(_: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def health(request: web.Request) -> web.Response:
    resources: DemoResources = request.app["resources"]
    return web.json_response(
        {
            "ok": True,
            "env_file": resources.args.loaded_env_file,
            "backend": resources.args.vision_backend,
            "providers": resources.inferencer.providers,
            "camera": resources.args.camera,
            "camera_ok": resources.camera_error is None,
            "camera_error": resources.camera_error,
        }
    )


async def offer(request: web.Request) -> web.Response:
    params = await request.json()
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    request.app["pcs"].add(pc)

    @pc.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        if channel.label != "status":
            return
        resources: DemoResources = request.app["resources"]
        resources.broadcaster.add(channel)

        @channel.on("close")
        def on_close() -> None:
            resources.broadcaster.discard(channel)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            request.app["pcs"].discard(pc)

    resources: DemoResources = request.app["resources"]
    pc.addTrack(AnnotatedVisionTrack(resources))

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def on_startup(app: web.Application) -> None:
    resources: DemoResources = app["resources"]
    resources.start()
    print(f"webrtc: env_file={resources.args.loaded_env_file or 'not found'}")
    print(f"webrtc: backend={resources.args.vision_backend}")
    print(f"webrtc: providers={resources.inferencer.providers}")


async def on_shutdown(app: web.Application) -> None:
    coros = [pc.close() for pc in app["pcs"]]
    if coros:
        await asyncio.gather(*coros)
    app["pcs"].clear()
    app["resources"].stop()


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


def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application()
    app["resources"] = DemoResources(args)
    app["pcs"] = set()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_post("/offer", offer)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    args = parse_args()
    print("webrtc: open one of these URLs from a LAN browser:")
    for url in local_urls(args.host, args.port):
        print(f"  {url}")
    web.run_app(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
