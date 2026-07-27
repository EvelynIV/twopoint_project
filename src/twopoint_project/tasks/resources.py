from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from twopoint_project.config import (
    CenterFlashTrackConfig,
    CenterThenFlashConfig,
    RuntimeConfig,
)
from twopoint_project.f32c.gimbal import open_serial_gimbal
from twopoint_project.flash import open_laser_pointer
from twopoint_project.tools.monitor import CenterRunMonitor, default_monitor_output_path
from twopoint_project.vision.inferencer import build_vision_inferencer
from twopoint_project.vision.pipeline import VisionProducer


TaskRuntimeConfig = CenterThenFlashConfig | CenterFlashTrackConfig


@dataclass
class TaskResources:
    inferencer: Any
    capture: Any
    gimbal: Any
    vision: VisionProducer
    laser: Any
    monitor: CenterRunMonitor


def open_camera_capture(
    *,
    width: int | None,
    height: int | None,
    fps: int | None,
    camera_index: int | None = None,
) -> object:
    from twopoint_project.vision.capture import CameraCapture

    return CameraCapture(
        camera_index=camera_index,
        width=width or 1280,
        height=height or 720,
        fps=fps or 30,
    )


def validate_runtime(task_config: TaskRuntimeConfig, runtime_config: RuntimeConfig) -> None:
    if runtime_config.webrtc.enabled and task_config.center.loop_hz <= 0:
        raise ValueError("center.loop_hz must be greater than 0 when WebRTC is enabled")
    if runtime_config.webrtc.port <= 0:
        raise ValueError("TWOPOINT_WEBRTC_PORT must be greater than 0")


def monitor_fps(task_config: TaskRuntimeConfig, runtime_config: RuntimeConfig) -> float:
    if isinstance(task_config, CenterFlashTrackConfig):
        return min(
            float(runtime_config.camera.fps),
            task_config.closed_loop.motor_loop_hz,
        )
    return task_config.center.loop_hz


@contextmanager
def open_task_resources(
    task_config: TaskRuntimeConfig,
    runtime_config: RuntimeConfig,
) -> Iterator[TaskResources]:
    """Build and safely tear down the hardware and diagnostic resource graph."""
    validate_runtime(task_config, runtime_config)
    inferencer = build_vision_inferencer(
        backend=runtime_config.vision.backend,
        onnx_path=runtime_config.vision.onnx_path,
        npu_model_path=runtime_config.vision.npu_model_path,
        npu_library_path=runtime_config.vision.npu_library_path,
        img_size=runtime_config.vision.img_size,
        npu_box_confidence_threshold=runtime_config.vision.npu_box_confidence_threshold,
        npu_nms_threshold=runtime_config.vision.npu_nms_threshold,
    )
    output_path = default_monitor_output_path(task_config.mode)

    with ExitStack() as stack:
        capture = stack.enter_context(
            open_camera_capture(
                width=runtime_config.camera.width,
                height=runtime_config.camera.height,
                fps=runtime_config.camera.fps,
            )
        )
        gimbal = stack.enter_context(
            open_serial_gimbal(
                port=task_config.f32c.port,
                baudrate=task_config.f32c.baudrate,
                x_id=task_config.f32c.x_id,
                y_id=task_config.f32c.y_id,
                speed_rpm=task_config.f32c.speed_rpm,
                startup_delay=task_config.f32c.startup_delay,
                command_interval=task_config.f32c.command_interval,
                enable_settle_delay=task_config.f32c.enable_settle_delay,
                debug_frames=task_config.f32c.debug_frames,
            )
        )
        stack.callback(gimbal.disable)
        vision = stack.enter_context(
            VisionProducer(
                capture=capture,
                inferencer=inferencer,
                target_filter_config=task_config.center.target_filter,
            )
        )
        laser = stack.enter_context(open_laser_pointer(initial_on=False))
        stack.callback(laser.off)
        monitor = CenterRunMonitor(
            enabled=runtime_config.webrtc.enabled,
            output_path=output_path,
            fps=monitor_fps(task_config, runtime_config),
            save_raw_video=task_config.recording.save_raw_video,
            webrtc_host=runtime_config.webrtc.host,
            webrtc_port=runtime_config.webrtc.port,
            backend=runtime_config.vision.backend,
            providers=inferencer.providers,
            task_name=task_config.mode,
        )
        stack.callback(monitor.close)
        resources = TaskResources(inferencer, capture, gimbal, vision, laser, monitor)
        yield resources
