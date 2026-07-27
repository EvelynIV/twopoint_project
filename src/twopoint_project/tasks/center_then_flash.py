from __future__ import annotations

import time

from twopoint_project.config import CenterThenFlashConfig, RuntimeConfig, TaskConfig
from twopoint_project.contrl.control_loop import run_target_centering_loop
from twopoint_project.contrl.target_center_servo import TargetCenterServo
from twopoint_project.tasks.resources import TaskResources, open_task_resources


def run(task_config: TaskConfig, runtime_config: RuntimeConfig) -> bool:
    if not isinstance(task_config, CenterThenFlashConfig):
        raise TypeError("center_then_flash runner requires CenterThenFlashConfig")

    servo = _build_servo(task_config)
    with open_task_resources(task_config, runtime_config) as resources:
        _log_configuration(task_config, runtime_config, resources)
        _initialize(resources)
        resources.monitor.start()
        centered = _center(task_config, resources, servo)
        _fire_if_allowed(task_config, resources, centered)
        return centered


def _build_servo(task_config: CenterThenFlashConfig) -> TargetCenterServo:
    return TargetCenterServo(
        center_x=task_config.center.target_x,
        center_y=task_config.center.target_y,
        x_gain_deg=task_config.center.x_gain_deg,
        y_gain_deg=task_config.center.y_gain_deg,
        max_step_deg=task_config.center.max_step_deg,
        deadband=task_config.center.deadband,
        x_pid=task_config.center.pid.x if task_config.center.pid is not None else None,
        y_pid=task_config.center.pid.y if task_config.center.pid is not None else None,
        feedforward=task_config.center.feedforward,
    )


def _initialize(resources: TaskResources) -> None:
    resources.laser.off()
    resources.gimbal.initialize()


def _center(
    task_config: CenterThenFlashConfig,
    resources: TaskResources,
    servo: TargetCenterServo,
) -> bool:
    return run_target_centering_loop(
        vision=resources.vision,
        gimbal=resources.gimbal,
        servo=servo,
        loop_hz=task_config.center.loop_hz,
        timeout=task_config.center.timeout,
        stable_frames=task_config.center.stable_frames,
        stale_target_seconds=task_config.center.stale_target_seconds,
        stale_target_step_scale=task_config.center.stale_target_step_scale,
        on_frame=resources.monitor.on_frame if resources.monitor.enabled else None,
    )


def _fire_if_allowed(
    task_config: CenterThenFlashConfig,
    resources: TaskResources,
    centered: bool,
) -> None:
    if not centered and not task_config.behavior.fire_after_timeout:
        print("laser: skipped because centering did not settle")
        return
    print("laser: on")
    resources.laser.on()
    time.sleep(task_config.laser.hold_seconds)
    print("laser: off")
    resources.laser.off()


def _log_configuration(
    task_config: CenterThenFlashConfig,
    runtime_config: RuntimeConfig,
    resources: TaskResources,
) -> None:
    print(f"task: {task_config.mode}")
    print(f"task: backend={runtime_config.vision.backend}")
    print(f"task: providers={resources.inferencer.providers}")
    print(f"task: center_timeout={task_config.center.timeout:.2f}s")
    print(f"task: laser_hold_seconds={task_config.laser.hold_seconds:.2f}s")
    print(f"monitor: record_webrtc={runtime_config.webrtc.enabled}")
