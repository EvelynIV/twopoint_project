from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from twopoint_project.config import (
    BehaviorConfig,
    CameraConfig,
    CenterConfig,
    CenterFlashTrackConfig,
    CenterThenFlashConfig,
    F32CConfig,
    LaserConfig,
    RecordingConfig,
    RuntimeConfig,
    VisionConfig,
    WebRtcConfig,
)
from twopoint_project.f32c.gimbal import GimbalAngles
from twopoint_project.vision.inferencer import VisionInferenceDetails
from twopoint_project.vision3.laser_area_mapping import predict_laser_point_from_normalized_area


class FakeCapture:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.frame = SimpleNamespace(frame_id=1, frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8))
        self.closed = False

    def __enter__(self) -> FakeCapture:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def read_frame(self, timeout: float | None = None) -> object:
        return self.frame


class FakeGimbal:
    def __init__(self) -> None:
        self.initialized = False
        self.disabled = False
        self.moves: list[tuple[float, float]] = []
        self.x_angle = 0.0
        self.y_angle = 0.0

    def __enter__(self) -> FakeGimbal:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def initialize(self) -> None:
        self.initialized = True

    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        self.moves.append((x_delta_deg, y_delta_deg))

    def move_to(self, x_angle_deg: float, y_angle_deg: float) -> None:
        self.x_angle = x_angle_deg
        self.y_angle = y_angle_deg
        self.moves.append((x_angle_deg, y_angle_deg))

    def read_angles(self, timeout: float = 0.01) -> GimbalAngles:
        return GimbalAngles(self.x_angle, self.y_angle, time.monotonic_ns())

    def commanded_angles(self) -> GimbalAngles:
        return GimbalAngles(
            self.x_angle,
            self.y_angle,
            time.monotonic_ns(),
            feedback_valid=False,
        )

    def sync_commanded_angles(self, angles: GimbalAngles) -> None:
        self.x_angle = angles.x_deg
        self.y_angle = angles.y_deg

    def disable(self) -> None:
        self.disabled = True


class FakeLaser:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.events: list[str] = []

    def __enter__(self) -> FakeLaser:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def on(self) -> None:
        self.events.append("on")

    def off(self) -> None:
        self.events.append("off")


class FakeInferencer:
    @property
    def providers(self) -> list[str]:
        return ["fake"]

    def predict(self, frame_bgr: np.ndarray) -> list[dict[str, float | str]]:
        return [{"label": "target_center", "x": 0.5, "y": 0.5, "confidence": 1.0}]


class FakeDistanceInferencer(FakeInferencer):
    def predict_with_details(self, frame_bgr: np.ndarray) -> VisionInferenceDetails:
        height, width = frame_bgr.shape[:2]
        area = 0.02
        laser = predict_laser_point_from_normalized_area(
            area,
            frame_width=width,
            frame_height=height,
        )
        assert laser is not None
        return VisionInferenceDetails(
            points=[
                {
                    "label": "target_center",
                    "x": laser.x,
                    "y": laser.y,
                    "confidence": 1.0,
                }
            ],
            target_area_normalized=area,
            target_distance_cm=142.0,
        )


class FakeVisionFrames:
    def __init__(self, frames: list[object]) -> None:
        self.frames = list(frames)

    def read_nowait_latest(self) -> object | None:
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakeVideoRecorder:
    instances: list[FakeVideoRecorder] = []

    def __init__(self, output_path: Path, fps: float, label: str) -> None:
        self.output_path = output_path
        self.fps = fps
        self.label = label
        self.frames: list[object] = []
        FakeVideoRecorder.instances.append(self)

    def write(self, frame_bgr: object) -> None:
        self.frames.append(frame_bgr)

    def close(self) -> None:
        pass


def make_task_config() -> CenterThenFlashConfig:
    return CenterThenFlashConfig(
        mode="center_then_flash",
        f32c=F32CConfig(
            port="/dev/null",
            baudrate=115200,
            x_id=1,
            y_id=2,
            speed_rpm=100,
            startup_delay=0,
            command_interval=0,
            enable_settle_delay=0,
            debug_frames=False,
        ),
        center=CenterConfig(
            target_x=0.5,
            target_y=0.5,
            x_gain_deg=8.0,
            y_gain_deg=-8.0,
            max_step_deg=1.0,
            deadband=0.006,
            loop_hz=0,
            timeout=2.0,
            stable_frames=1,
        ),
        laser=LaserConfig(hold_seconds=0),
        behavior=BehaviorConfig(fire_after_timeout=True, exit_after_fire=True),
        recording=RecordingConfig(),
    )


def make_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        config_path=Path("unused.json"),
        camera=CameraConfig(width=1280, height=720, fps=30),
        vision=VisionConfig(
            backend="traditional",
            onnx_path="model-bin/runs/twopoint/best.onnx",
            img_size=640,
        ),
        webrtc=WebRtcConfig(enabled=False, host="0.0.0.0", port=8080),
    )


def make_track_config() -> CenterFlashTrackConfig:
    return CenterFlashTrackConfig(
        mode="center_flash_track",
        f32c=F32CConfig(
            port="/dev/null",
            baudrate=115200,
            x_id=1,
            y_id=2,
            speed_rpm=100,
            startup_delay=0,
            command_interval=0,
            enable_settle_delay=0,
            debug_frames=False,
        ),
        center=CenterConfig(
            target_x=0.5,
            target_y=0.5,
            x_gain_deg=8.0,
            y_gain_deg=-8.0,
            max_step_deg=1.0,
            deadband=0.006,
            loop_hz=0,
            timeout=2.0,
            stable_frames=1,
        ),
        laser=LaserConfig(hold_seconds=0),
        behavior=BehaviorConfig(fire_after_timeout=True, exit_after_fire=False),
        recording=RecordingConfig(),
    )


def make_constant_laser_track_config() -> CenterFlashTrackConfig:
    task_config = make_track_config()
    return CenterFlashTrackConfig(
        mode=task_config.mode,
        f32c=task_config.f32c,
        center=task_config.center,
        laser=LaserConfig(hold_seconds=0, on_during_run=True),
        behavior=task_config.behavior,
        recording=task_config.recording,
    )


class StopAfterCalls:
    def __init__(self, calls: int) -> None:
        self.calls = 0
        self.limit = calls

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.limit


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

