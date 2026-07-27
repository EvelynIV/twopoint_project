from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from twopoint_project.contrl.target_center_servo import FeedForwardConfig, PIDAxisGains
from twopoint_project.contrl.laser_alignment_servo import TrackClosedLoopConfig
from twopoint_project.f32c.gimbal import (
    DEFAULT_BAUDRATE,
    DEFAULT_COMMAND_INTERVAL,
    DEFAULT_ENABLE_SETTLE_DELAY,
    DEFAULT_SERIAL_PORT,
    DEFAULT_SPEED_RPM,
    DEFAULT_STARTUP_DELAY,
    DEFAULT_X_ID,
    DEFAULT_Y_ID,
)
from twopoint_project.vision.capture import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
)
from twopoint_project.vision.inferencer import (
    DEFAULT_IMG_SIZE,
    DEFAULT_NPU_LIBRARY_PATH,
    DEFAULT_NPU_MODEL_PATH,
    DEFAULT_NPU_NMS_THRESHOLD,
    DEFAULT_NPU_BOX_CONFIDENCE_THRESHOLD,
    DEFAULT_ONNX_PATH,
    DEFAULT_VISION_BACKEND,
)
from twopoint_project.vision.target_filter import TargetCenterFilterConfig


DEFAULT_CONFIG_PATH = Path("configs/tasks/center_then_flash.json")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class VisionConfig:
    backend: str = DEFAULT_VISION_BACKEND
    onnx_path: str = DEFAULT_ONNX_PATH
    npu_model_path: str = DEFAULT_NPU_MODEL_PATH
    npu_library_path: str = DEFAULT_NPU_LIBRARY_PATH
    img_size: int = DEFAULT_IMG_SIZE
    npu_box_confidence_threshold: float = DEFAULT_NPU_BOX_CONFIDENCE_THRESHOLD
    npu_nms_threshold: float = DEFAULT_NPU_NMS_THRESHOLD


@dataclass(frozen=True)
class WebRtcConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Path
    camera: CameraConfig
    vision: VisionConfig
    webrtc: WebRtcConfig


@dataclass(frozen=True)
class CameraConfig:
    width: int = DEFAULT_CAMERA_WIDTH
    height: int = DEFAULT_CAMERA_HEIGHT
    fps: int = DEFAULT_CAMERA_FPS

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height, and fps must be greater than 0")


@dataclass(frozen=True)
class F32CConfig:
    port: str = DEFAULT_SERIAL_PORT
    baudrate: int = DEFAULT_BAUDRATE
    x_id: int = DEFAULT_X_ID
    y_id: int = DEFAULT_Y_ID
    speed_rpm: int = DEFAULT_SPEED_RPM
    startup_delay: float = DEFAULT_STARTUP_DELAY
    command_interval: float = DEFAULT_COMMAND_INTERVAL
    enable_settle_delay: float = DEFAULT_ENABLE_SETTLE_DELAY
    debug_frames: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> F32CConfig:
        return cls(
            port=str(data.get("port", cls.port)),
            baudrate=int(data.get("baudrate", cls.baudrate)),
            x_id=int(data.get("x_id", cls.x_id)),
            y_id=int(data.get("y_id", cls.y_id)),
            speed_rpm=int(data.get("speed_rpm", cls.speed_rpm)),
            startup_delay=float(data.get("startup_delay", cls.startup_delay)),
            command_interval=float(data.get("command_interval", cls.command_interval)),
            enable_settle_delay=float(data.get("enable_settle_delay", cls.enable_settle_delay)),
            debug_frames=bool(data.get("debug_frames", cls.debug_frames)),
        )


@dataclass(frozen=True)
class PIDConfig:
    x: PIDAxisGains
    y: PIDAxisGains

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        x_gain_deg: float,
        y_gain_deg: float,
        max_step_deg: float,
    ) -> PIDConfig:
        return cls(
            x=pid_axis_from_dict(section(data, "x"), kp=x_gain_deg, output_limit_deg=max_step_deg),
            y=pid_axis_from_dict(section(data, "y"), kp=y_gain_deg, output_limit_deg=max_step_deg),
        )


def pid_axis_from_dict(data: dict[str, Any], *, kp: float, output_limit_deg: float) -> PIDAxisGains:
    return PIDAxisGains(
        kp=float(data.get("kp", kp)),
        ki=float(data.get("ki", 0.0)),
        kd=float(data.get("kd", 0.0)),
        integral_limit=float(data.get("integral_limit", 0.0)),
        output_limit_deg=float(data.get("output_limit_deg", output_limit_deg)),
        integral_separation_threshold=float(data.get("integral_separation_threshold", 0.0)),
        derivative_separation_threshold=float(data.get("derivative_separation_threshold", 0.0)),
        fuzzy_enabled=bool(data.get("fuzzy_enabled", False)),
        fuzzy_error_low=float(data.get("fuzzy_error_low", 0.02)),
        fuzzy_error_high=float(data.get("fuzzy_error_high", 0.18)),
        fuzzy_kp_near_scale=float(data.get("fuzzy_kp_near_scale", 0.65)),
        fuzzy_kp_far_scale=float(data.get("fuzzy_kp_far_scale", 1.25)),
    )


def feedforward_from_dict(data: dict[str, Any]) -> FeedForwardConfig:
    return FeedForwardConfig(
        enabled=bool(data.get("enabled", False)),
        lead_time=float(data.get("lead_time", 0.0)),
        max_prediction_error=float(data.get("max_prediction_error", 0.05)),
        max_velocity=float(data.get("max_velocity", 2.0)),
        velocity_alpha=float(data.get("velocity_alpha", 0.5)),
    )


def target_filter_from_dict(data: dict[str, Any]) -> TargetCenterFilterConfig:
    return TargetCenterFilterConfig(
        enabled=bool(data.get("enabled", False)),
        ema_alpha=float(data.get("ema_alpha", 0.35)),
        max_jump=float(data.get("max_jump", 0.08)),
        jump_confirm_frames=int(data.get("jump_confirm_frames", 2)),
    )


@dataclass(frozen=True)
class CenterConfig:
    target_x: float = 0.5
    target_y: float = 0.5
    x_gain_deg: float = -8.0
    y_gain_deg: float = 8.0
    max_step_deg: float = 1.0
    deadband: float = 0.006
    loop_hz: float = 15.0
    stale_target_seconds: float = 0.3
    stale_target_step_scale: float = 0.5
    timeout: float = 100.0
    stable_frames: int = 3
    pid: PIDConfig | None = None
    feedforward: FeedForwardConfig = FeedForwardConfig()
    target_filter: TargetCenterFilterConfig = TargetCenterFilterConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CenterConfig:
        stable_frames = int(data.get("stable_frames", cls.stable_frames))
        if stable_frames <= 0:
            raise ValueError("center.stable_frames must be greater than 0")
        timeout = float(data.get("timeout", cls.timeout))
        if timeout < 0:
            raise ValueError("center.timeout must be non-negative")
        x_gain_deg = float(data.get("x_gain_deg", cls.x_gain_deg))
        y_gain_deg = float(data.get("y_gain_deg", cls.y_gain_deg))
        max_step_deg = float(data.get("max_step_deg", cls.max_step_deg))
        stale_target_seconds = float(data.get("stale_target_seconds", cls.stale_target_seconds))
        if stale_target_seconds < 0:
            raise ValueError("center.stale_target_seconds must be non-negative")
        stale_target_step_scale = float(data.get("stale_target_step_scale", cls.stale_target_step_scale))
        if not 0.0 <= stale_target_step_scale <= 1.0:
            raise ValueError("center.stale_target_step_scale must be between 0 and 1")
        pid_data = section(data, "pid") if "pid" in data else {}
        feedforward_data = section(data, "feedforward") if "feedforward" in data else {}
        target_filter_data = section(data, "target_filter") if "target_filter" in data else {}
        return cls(
            target_x=float(data.get("target_x", cls.target_x)),
            target_y=float(data.get("target_y", cls.target_y)),
            x_gain_deg=x_gain_deg,
            y_gain_deg=y_gain_deg,
            max_step_deg=max_step_deg,
            deadband=float(data.get("deadband", cls.deadband)),
            loop_hz=float(data.get("loop_hz", cls.loop_hz)),
            stale_target_seconds=stale_target_seconds,
            stale_target_step_scale=stale_target_step_scale,
            timeout=timeout,
            stable_frames=stable_frames,
            pid=PIDConfig.from_dict(
                pid_data,
                x_gain_deg=x_gain_deg,
                y_gain_deg=y_gain_deg,
                max_step_deg=max_step_deg,
            ),
            feedforward=feedforward_from_dict(feedforward_data),
            target_filter=target_filter_from_dict(target_filter_data),
        )


def track_closed_loop_from_dict(data: dict[str, Any]) -> TrackClosedLoopConfig:
    defaults = TrackClosedLoopConfig()
    motor_loop_hz = float(data.get("motor_loop_hz", defaults.motor_loop_hz))
    feedback_timeout = float(data.get("feedback_timeout", defaults.feedback_timeout))
    max_vision_age_seconds = float(
        data.get("max_vision_age_seconds", defaults.max_vision_age_seconds)
    )
    angle_deadband_deg = float(data.get("angle_deadband_deg", defaults.angle_deadband_deg))
    max_visual_correction_deg = float(
        data.get("max_visual_correction_deg", defaults.max_visual_correction_deg)
    )
    if motor_loop_hz <= 0:
        raise ValueError("closed_loop.motor_loop_hz must be greater than 0")
    if feedback_timeout <= 0:
        raise ValueError("closed_loop.feedback_timeout must be greater than 0")
    if max_vision_age_seconds < 0:
        raise ValueError("closed_loop.max_vision_age_seconds must be non-negative")
    if angle_deadband_deg < 0:
        raise ValueError("closed_loop.angle_deadband_deg must be non-negative")
    if max_visual_correction_deg <= 0:
        raise ValueError("closed_loop.max_visual_correction_deg must be greater than 0")
    pid_data = section(data, "pid") if "pid" in data else {}
    feedforward_data = section(data, "feedforward") if "feedforward" in data else {}
    feedforward = feedforward_from_dict(feedforward_data)
    if feedforward.lead_time < 0:
        raise ValueError("closed_loop.feedforward.lead_time must be non-negative")
    if feedforward.max_prediction_error < 0:
        raise ValueError(
            "closed_loop.feedforward.max_prediction_error must be non-negative"
        )
    if feedforward.max_velocity < 0:
        raise ValueError("closed_loop.feedforward.max_velocity must be non-negative")
    if not 0.0 <= feedforward.velocity_alpha <= 1.0:
        raise ValueError(
            "closed_loop.feedforward.velocity_alpha must be between 0 and 1"
        )
    x_default = defaults.x_pid
    y_default = defaults.y_pid
    x_pid_data = {
        "ki": x_default.ki,
        "kd": x_default.kd,
        "integral_limit": x_default.integral_limit,
        **section(pid_data, "x"),
    }
    y_pid_data = {
        "ki": y_default.ki,
        "kd": y_default.kd,
        "integral_limit": y_default.integral_limit,
        **section(pid_data, "y"),
    }
    return TrackClosedLoopConfig(
        motor_loop_hz=motor_loop_hz,
        feedback_timeout=feedback_timeout,
        max_vision_age_seconds=max_vision_age_seconds,
        angle_deadband_deg=angle_deadband_deg,
        x_angle_gain_deg=float(data.get("x_angle_gain_deg", defaults.x_angle_gain_deg)),
        y_angle_gain_deg=float(data.get("y_angle_gain_deg", defaults.y_angle_gain_deg)),
        max_visual_correction_deg=max_visual_correction_deg,
        feedforward=feedforward,
        x_pid=pid_axis_from_dict(
            x_pid_data,
            kp=x_default.kp,
            output_limit_deg=x_default.output_limit_deg,
        ),
        y_pid=pid_axis_from_dict(
            y_pid_data,
            kp=y_default.kp,
            output_limit_deg=y_default.output_limit_deg,
        ),
    )

@dataclass(frozen=True)
class LaserConfig:
    hold_seconds: float = 5.0
    on_during_run: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaserConfig:
        hold_seconds = float(data.get("hold_seconds", cls.hold_seconds))
        if hold_seconds < 0:
            raise ValueError("laser.hold_seconds must be non-negative")
        return cls(
            hold_seconds=hold_seconds,
            on_during_run=bool(data.get("on_during_run", cls.on_during_run)),
        )


@dataclass(frozen=True)
class BehaviorConfig:
    fire_after_timeout: bool = True
    exit_after_fire: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehaviorConfig:
        return cls(
            fire_after_timeout=bool(data.get("fire_after_timeout", cls.fire_after_timeout)),
            exit_after_fire=bool(data.get("exit_after_fire", cls.exit_after_fire)),
        )


@dataclass(frozen=True)
class RecordingConfig:
    save_raw_video: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordingConfig:
        return cls(save_raw_video=bool(data.get("save_raw_video", cls.save_raw_video)))


@dataclass(frozen=True)
class UnsupportedTaskConfig:
    mode: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class CenterThenFlashConfig:
    mode: str
    f32c: F32CConfig
    center: CenterConfig
    laser: LaserConfig
    behavior: BehaviorConfig
    recording: RecordingConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CenterThenFlashConfig:
        mode = str(data.get("mode", ""))
        if mode != "center_then_flash":
            raise ValueError(f"center_then_flash config must declare mode='center_then_flash', got {mode!r}")
        return cls(
            mode=mode,
            f32c=F32CConfig.from_dict(section(data, "f32c")),
            center=CenterConfig.from_dict(section(data, "center")),
            laser=LaserConfig.from_dict(section(data, "laser")),
            behavior=BehaviorConfig.from_dict(section(data, "behavior")),
            recording=RecordingConfig.from_dict(section(data, "recording")),
        )


@dataclass(frozen=True)
class CenterFlashTrackConfig:
    mode: str
    f32c: F32CConfig
    center: CenterConfig
    laser: LaserConfig
    behavior: BehaviorConfig
    recording: RecordingConfig
    closed_loop: TrackClosedLoopConfig = TrackClosedLoopConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CenterFlashTrackConfig:
        mode = str(data.get("mode", ""))
        if mode != "center_flash_track":
            raise ValueError(f"center_flash_track config must declare mode='center_flash_track', got {mode!r}")
        return cls(
            mode=mode,
            f32c=F32CConfig.from_dict(section(data, "f32c")),
            center=CenterConfig.from_dict(section(data, "center")),
            laser=LaserConfig.from_dict(section(data, "laser")),
            behavior=BehaviorConfig.from_dict(section(data, "behavior")),
            recording=RecordingConfig.from_dict(section(data, "recording")),
            closed_loop=track_closed_loop_from_dict(section(data, "closed_loop")),
        )


TaskConfig = CenterThenFlashConfig | CenterFlashTrackConfig | UnsupportedTaskConfig


def runtime_config_from_env(config_path: Path | None = None) -> RuntimeConfig:
    path = config_path or Path(env_str("TWOPOINT_CONFIG", str(DEFAULT_CONFIG_PATH)))
    return RuntimeConfig(
        config_path=path,
        camera=CameraConfig(
            width=env_int("TWOPOINT_CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH),
            height=env_int("TWOPOINT_CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT),
            fps=env_int("TWOPOINT_CAMERA_FPS", DEFAULT_CAMERA_FPS),
        ),
        vision=VisionConfig(
            backend=env_str("TWOPOINT_VISION_BACKEND", DEFAULT_VISION_BACKEND),
            onnx_path=env_str("TWOPOINT_ONNX_PATH", DEFAULT_ONNX_PATH),
            npu_model_path=env_str("TWOPOINT_NPU_MODEL_PATH", DEFAULT_NPU_MODEL_PATH),
            npu_library_path=env_str("TWOPOINT_NPU_LIBRARY_PATH", DEFAULT_NPU_LIBRARY_PATH),
            img_size=env_int("TWOPOINT_IMG_SIZE", DEFAULT_IMG_SIZE),
            npu_box_confidence_threshold=env_float(
                "TWOPOINT_NPU_BOX_CONFIDENCE_THRESHOLD",
                DEFAULT_NPU_BOX_CONFIDENCE_THRESHOLD,
            ),
            npu_nms_threshold=env_float(
                "TWOPOINT_NPU_NMS_THRESHOLD",
                DEFAULT_NPU_NMS_THRESHOLD,
            ),
        ),
        webrtc=WebRtcConfig(
            enabled=env_bool("TWOPOINT_WEBRTC_ENABLED", False),
            host=env_str("TWOPOINT_WEBRTC_HOST", "0.0.0.0"),
            port=env_int("TWOPOINT_WEBRTC_PORT", 8080),
        ),
    )


def load_task_config(path: Path) -> TaskConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("task config must be a JSON object")
    mode = str(data.get("mode", ""))
    if not mode:
        raise ValueError("task config must include mode")
    if mode == "center_then_flash":
        return CenterThenFlashConfig.from_dict(data)
    if mode == "center_flash_track":
        return CenterFlashTrackConfig.from_dict(data)
    if mode == "search_center_then_flash":
        return UnsupportedTaskConfig(mode=mode, raw=data)
    raise ValueError(f"unsupported task mode: {mode}")
