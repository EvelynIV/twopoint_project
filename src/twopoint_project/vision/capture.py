from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30
DEFAULT_CAMERA_BY_ID_DIR = Path("/dev/v4l/by-id")
DEFAULT_SAMPLE_TIMEOUT_SECONDS = 1.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 5.0
DEFAULT_RECONNECT_DELAY_SECONDS = 0.5


@dataclass(frozen=True)
class CapturedFrame:
    camera_id: str
    stream_generation: int
    frame_id: int
    captured_at_monotonic_ns: int
    pts_ns: int | None
    duration_ns: int | None
    width: int
    height: int
    pixel_format: str
    frame_bgr: np.ndarray
    sample_interval_ns: int | None
    copy_duration_ns: int

    @property
    def timestamp(self) -> float:
        """Compatibility timestamp in the monotonic clock domain."""
        return self.captured_at_monotonic_ns / 1_000_000_000.0

    @property
    def identity(self) -> tuple[int, int]:
        return self.stream_generation, self.frame_id


class LatestCapturedFrame:
    """Single-consumer, non-blocking overwrite slot for captured frames."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._version = 0
        self._last_read_version = 0
        self._closed = False
        self.overwrite_count = 0

    def publish(self, frame: CapturedFrame) -> None:
        with self._condition:
            if self._closed:
                return
            if self._version > self._last_read_version:
                self.overwrite_count += 1
            self._latest = frame
            self._version += 1
            self._condition.notify_all()

    def read_after(
        self,
        identity: tuple[int, int] | None,
        timeout: float | None = None,
    ) -> CapturedFrame | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._latest is not None and self._latest.identity != identity:
                    self._last_read_version = self._version
                    return self._latest
                if self._closed:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def clear(self) -> None:
        with self._condition:
            self._latest = None
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def resolve_camera_device(
    device: str | Path | None = None,
    *,
    by_id_dir: Path = DEFAULT_CAMERA_BY_ID_DIR,
) -> str:
    if device is not None:
        candidate = Path(device)
        if not candidate.exists():
            raise FileNotFoundError(f"camera device does not exist: {candidate}")
        return str(candidate)

    candidates = sorted(path for path in by_id_dir.glob("*-video-index0") if path.exists())
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise FileNotFoundError(f"no stable camera device found under {by_id_dir}")
    joined = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"multiple stable camera devices found; cannot select automatically: {joined}")


def build_gstreamer_pipeline(device: str, width: int, height: int, fps: int) -> str:
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("camera width, height, and fps must be greater than 0")
    return " ".join(
        [
            "v4l2src",
            f"device={json.dumps(device)}",
            "do-timestamp=true",
            "!",
            f"image/jpeg,width={width},height={height},framerate={fps}/1",
            "!",
            "queue",
            "max-size-buffers=1",
            "max-size-bytes=0",
            "max-size-time=0",
            "leaky=downstream",
            "!",
            "jpegdec",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=BGR",
            "!",
            "appsink",
            "name=frame_sink",
            "max-buffers=1",
            "drop=true",
            "sync=false",
            "emit-signals=false",
            "enable-last-sample=false",
            "wait-on-eos=false",
        ]
    )


def _clock_time(value: int, clock_time_none: int) -> int | None:
    return None if value == clock_time_none else int(value)


class CameraCapture:
    """Continuously captures the latest decoded BGR frame through GStreamer."""

    def __init__(
        self,
        *,
        width: int = DEFAULT_CAMERA_WIDTH,
        height: int = DEFAULT_CAMERA_HEIGHT,
        fps: int = DEFAULT_CAMERA_FPS,
        device: str | Path | None = None,
        sample_timeout: float = DEFAULT_SAMPLE_TIMEOUT_SECONDS,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        reconnect_delay: float = DEFAULT_RECONNECT_DELAY_SECONDS,
        camera_index: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("camera width, height, and fps must be greater than 0")
        if sample_timeout <= 0 or startup_timeout <= 0 or reconnect_delay < 0:
            raise ValueError("camera timeout values must be positive and reconnect delay non-negative")
        if device is not None and camera_index is not None:
            raise ValueError("set either device or camera_index, not both")
        if camera_index is not None:
            device = Path(f"/dev/video{camera_index}")

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device = device
        self.sample_timeout = float(sample_timeout)
        self.startup_timeout = float(startup_timeout)
        self.reconnect_delay = float(reconnect_delay)
        self.latest = LatestCapturedFrame()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader_identity: tuple[int, int] | None = None
        self._last_error: BaseException | None = None
        self._stream_generation = 0
        self.negotiated_caps: str | None = None

    def __enter__(self) -> CameraCapture:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def open(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.latest = LatestCapturedFrame()
        self._reader_identity = None
        self._last_error = None
        self._stop_event.clear()
        self._first_frame_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="twopoint-camera-capture",
            daemon=True,
        )
        self._thread.start()
        if not self._first_frame_event.wait(self.startup_timeout):
            error = self._last_error
            self.release()
            if error is not None:
                raise RuntimeError("camera did not produce a frame before startup timeout") from error
            raise TimeoutError("camera did not produce a frame before startup timeout")

    def read_frame(self, timeout: float | None = None) -> CapturedFrame:
        if self._thread is None or not self._thread.is_alive():
            self.open()
        frame = self.latest.read_after(self._reader_identity, timeout)
        if frame is None:
            if self._stop_event.is_set():
                raise RuntimeError("camera capture stopped") from self._last_error
            raise TimeoutError("timed out waiting for a new camera frame")
        self._reader_identity = frame.identity
        return frame

    def release(self) -> None:
        self._stop_event.set()
        self.latest.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(self.sample_timeout + 1.0, 2.0))
        self._thread = None

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                resolved_device = resolve_camera_device(self.device)
                self._run_pipeline(resolved_device)
            except BaseException as exc:
                self._last_error = exc
                self.latest.clear()
                if self._stop_event.is_set():
                    break
                print(f"camera: stream error: {exc}; reconnecting")
                self._stop_event.wait(self.reconnect_delay)
        self.latest.close()

    def _run_pipeline(self, device: str) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        pipeline_text = build_gstreamer_pipeline(device, self.width, self.height, self.fps)
        pipeline = Gst.parse_launch(pipeline_text)
        sink = pipeline.get_by_name("frame_sink")
        if sink is None:
            raise RuntimeError("GStreamer pipeline does not contain frame_sink")
        bus = pipeline.get_bus()
        state_result = pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer camera pipeline failed to enter PLAYING")

        generation = self._stream_generation + 1
        frame_id = 0
        timeout_count = 0
        announced = False
        previous_sample_at: int | None = None
        try:
            while not self._stop_event.is_set():
                message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message is not None:
                    if message.type == Gst.MessageType.ERROR:
                        error, debug = message.parse_error()
                        raise RuntimeError(f"GStreamer camera error: {error}; {debug}")
                    raise RuntimeError("GStreamer camera stream reached EOS")

                sample = sink.emit("try-pull-sample", int(self.sample_timeout * Gst.SECOND))
                if sample is None:
                    timeout_count += 1
                    if timeout_count >= 3:
                        raise TimeoutError("timed out waiting for three consecutive camera samples")
                    continue
                timeout_count = 0
                captured_at = time.monotonic_ns()
                caps = sample.get_caps()
                structure = caps.get_structure(0)
                width = int(structure.get_value("width"))
                height = int(structure.get_value("height"))
                pixel_format = str(structure.get_value("format"))
                fraction_ok, fps_numerator, fps_denominator = structure.get_fraction("framerate")
                negotiated_fps = (
                    fps_numerator / fps_denominator
                    if fraction_ok and fps_denominator != 0
                    else 0.0
                )
                if (
                    (width, height, pixel_format) != (self.width, self.height, "BGR")
                    or negotiated_fps != self.fps
                ):
                    raise RuntimeError(
                        "unexpected negotiated camera caps: "
                        f"{width}x{height} {pixel_format} @{negotiated_fps:g}; expected "
                        f"{self.width}x{self.height} BGR @{self.fps}"
                    )

                buffer = sample.get_buffer()
                mapped_ok, mapped = buffer.map(Gst.MapFlags.READ)
                if not mapped_ok:
                    raise RuntimeError("failed to map GstBuffer from appsink")
                copy_started = time.monotonic_ns()
                try:
                    row_bytes = width * 3
                    if mapped.size % height != 0:
                        raise RuntimeError(
                            f"unexpected BGR buffer size {mapped.size} for {width}x{height}"
                        )
                    stride = mapped.size // height
                    if stride < row_bytes:
                        raise RuntimeError(
                            f"GStreamer BGR stride {stride} is smaller than row size {row_bytes}"
                        )
                    rows = np.frombuffer(mapped.data, dtype=np.uint8).reshape(height, stride)
                    frame_bgr = rows[:, :row_bytes].reshape(height, width, 3).copy()
                finally:
                    buffer.unmap(mapped)
                copy_duration = time.monotonic_ns() - copy_started

                frame_id += 1
                frame = CapturedFrame(
                    camera_id=device,
                    stream_generation=generation,
                    frame_id=frame_id,
                    captured_at_monotonic_ns=captured_at,
                    pts_ns=_clock_time(buffer.pts, Gst.CLOCK_TIME_NONE),
                    duration_ns=_clock_time(buffer.duration, Gst.CLOCK_TIME_NONE),
                    width=width,
                    height=height,
                    pixel_format=pixel_format,
                    frame_bgr=frame_bgr,
                    sample_interval_ns=(
                        None if previous_sample_at is None else captured_at - previous_sample_at
                    ),
                    copy_duration_ns=copy_duration,
                )
                previous_sample_at = captured_at
                if not announced:
                    self._stream_generation = generation
                    self.negotiated_caps = structure.to_string()
                    print(
                        f"camera: generation={generation} device={device} "
                        f"caps={self.negotiated_caps}"
                    )
                    announced = True
                self.latest.publish(frame)
                self._first_frame_event.set()
        finally:
            pipeline.set_state(Gst.State.NULL)
