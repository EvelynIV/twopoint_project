from __future__ import annotations

from contextlib import nullcontext
import time
from typing import ContextManager

from twopoint_project.config import CenterFlashTrackConfig, RuntimeConfig, TaskConfig
from twopoint_project.contrl.control_loop import StopCallback, run_laser_alignment_loop
from twopoint_project.contrl.feedback import GimbalFeedbackReader
from twopoint_project.contrl.laser_alignment_servo import LaserAlignmentServo
from twopoint_project.tasks.resources import TaskResources, open_task_resources
from twopoint_project.tools.monitor import EscKeyStopper


def run(
    task_config: TaskConfig,
    runtime_config: RuntimeConfig,
    stop_requested: StopCallback | None = None,
) -> bool:
    if not isinstance(task_config, CenterFlashTrackConfig):
        raise TypeError("center_flash_track runner requires CenterFlashTrackConfig")

    servo = LaserAlignmentServo(
        config=task_config.closed_loop,
        visual_deadband=task_config.center.deadband,
    )
    try:
        with open_task_resources(task_config, runtime_config) as resources:
            _log_configuration(task_config, runtime_config, resources)
            feedback = _initialize(task_config, resources, servo)
            resources.monitor.start()
            if task_config.laser.on_during_run:
                return _run_continuous_tracking(
                    task_config,
                    resources,
                    feedback,
                    servo,
                    stop_requested,
                )
            centered = _center(task_config, resources, feedback, servo)
            _track_after_center(
                task_config,
                resources,
                feedback,
                servo,
                centered,
                stop_requested,
            )
            return centered
    except KeyboardInterrupt:
        print("track: interrupted")
        return False


def _initialize(
    task_config: CenterFlashTrackConfig,
    resources: TaskResources,
    servo: LaserAlignmentServo,
) -> GimbalFeedbackReader:
    resources.laser.off()
    print("gimbal: initializing and enabling motors")
    resources.gimbal.initialize()
    print("gimbal: motors enabled")
    print("gimbal: commanding startup position x=0.0deg y=0.0deg")
    resources.gimbal.move_to(0.0, 0.0)
    feedback = GimbalFeedbackReader(
        resources.gimbal,
        timeout=task_config.closed_loop.feedback_timeout,
    )
    initial_angles = feedback.initialize()
    if initial_angles is not None:
        servo.record_angles(initial_angles)
    return feedback


def _run_continuous_tracking(
    task_config: CenterFlashTrackConfig,
    resources: TaskResources,
    feedback: GimbalFeedbackReader,
    servo: LaserAlignmentServo,
    stop_requested: StopCallback | None,
) -> bool:
    stopper_context: ContextManager[EscKeyStopper | None]
    if stop_requested is None:
        stopper_context = EscKeyStopper()
    else:
        print("track: using injected stop callback")
        stopper_context = nullcontext(None)

    with stopper_context as stopper:
        operator_stop = stop_requested if stop_requested is not None else stopper.should_stop

        def should_stop() -> bool:
            return resources.monitor.stop_requested() or operator_stop()

        print("laser: on for continuous tracking")
        resources.laser.on()
        print("track: continuous tracking started")
        run_laser_alignment_loop(
            vision=resources.vision,
            gimbal=resources.gimbal,
            feedback=feedback,
            servo=servo,
            loop_hz=task_config.closed_loop.motor_loop_hz,
            stop_requested=should_stop,
            on_frame=resources.monitor.on_frame if resources.monitor.enabled else None,
            phase="track",
        )
    print("track: continuous tracking stopped")
    return True


def _center(
    task_config: CenterFlashTrackConfig,
    resources: TaskResources,
    feedback: GimbalFeedbackReader,
    servo: LaserAlignmentServo,
) -> bool:
    return run_laser_alignment_loop(
        vision=resources.vision,
        gimbal=resources.gimbal,
        feedback=feedback,
        servo=servo,
        loop_hz=task_config.closed_loop.motor_loop_hz,
        deadline=time.monotonic() + task_config.center.timeout,
        stable_frames=task_config.center.stable_frames,
        stop_requested=resources.monitor.stop_requested,
        on_frame=resources.monitor.on_frame if resources.monitor.enabled else None,
        phase="center",
    )


def _track_after_center(
    task_config: CenterFlashTrackConfig,
    resources: TaskResources,
    feedback: GimbalFeedbackReader,
    servo: LaserAlignmentServo,
    centered: bool,
    stop_requested: StopCallback | None,
) -> None:
    stopper_context: ContextManager[EscKeyStopper | None]
    if stop_requested is None:
        stopper_context = EscKeyStopper()
    else:
        print("track: using injected stop callback")
        stopper_context = nullcontext(None)

    with stopper_context as stopper:
        operator_stop = stop_requested if stop_requested is not None else stopper.should_stop

        def should_stop() -> bool:
            return resources.monitor.stop_requested() or operator_stop()

        if task_config.laser.on_during_run:
            print("laser: keeping on during tracking")
        elif centered or task_config.behavior.fire_after_timeout:
            print("laser: on")
            resources.laser.on()
            stopped_during_fire = _sleep_with_stop(
                task_config.laser.hold_seconds,
                should_stop,
            )
            print("laser: off")
            resources.laser.off()
            if stopped_during_fire:
                return
        else:
            print("laser: skipped because centering did not settle")

        run_laser_alignment_loop(
            vision=resources.vision,
            gimbal=resources.gimbal,
            feedback=feedback,
            servo=servo,
            loop_hz=task_config.closed_loop.motor_loop_hz,
            stop_requested=should_stop,
            on_frame=resources.monitor.on_frame if resources.monitor.enabled else None,
            phase="track",
        )


def _sleep_with_stop(seconds: float, stop_requested: StopCallback) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_requested():
            return True
        time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
    return stop_requested()


def _log_configuration(
    task_config: CenterFlashTrackConfig,
    runtime_config: RuntimeConfig,
    resources: TaskResources,
) -> None:
    print(f"task: {task_config.mode}")
    print(f"task: backend={runtime_config.vision.backend}")
    print(f"task: providers={resources.inferencer.providers}")
    print(f"task: center_timeout={task_config.center.timeout:.2f}s")
    print(
        f"task: motor_loop={task_config.closed_loop.motor_loop_hz:.1f}Hz "
        f"feedback_timeout={task_config.closed_loop.feedback_timeout * 1000.0:.1f}ms"
    )
    feedforward = task_config.closed_loop.feedforward
    print(
        f"task: feedforward={'on' if feedforward.enabled else 'off'} "
        f"lead={feedforward.lead_time * 1000.0:.1f}ms "
        f"max_prediction={feedforward.max_prediction_error:.3f} "
        f"max_velocity={feedforward.max_velocity:.3f}/s "
        f"velocity_alpha={feedforward.velocity_alpha:.2f}"
    )
    if task_config.laser.on_during_run:
        print("task: laser_on_during_run=true")
    else:
        print(f"task: laser_hold_seconds={task_config.laser.hold_seconds:.2f}s")
    print(f"monitor: record_webrtc={runtime_config.webrtc.enabled}")
