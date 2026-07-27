from __future__ import annotations

import time
from typing import Any, Callable

from twopoint_project.contrl.feedback import GimbalFeedbackReader
from twopoint_project.contrl.laser_alignment_servo import LaserAlignmentServo
from twopoint_project.contrl.scheduler import FixedDeadlineScheduler, wait_for_motor_deadline
from twopoint_project.contrl.target_center_servo import (
    AimUpdate,
    PointPrediction,
    TargetCenterServo,
    sleep_for_loop_rate,
)
from twopoint_project.vision.inferencer import TargetCorners


FrameCallback = Callable[
    [Any, list[PointPrediction], AimUpdate, TargetCorners, PointPrediction | None],
    None,
]
StopCallback = Callable[[], bool]


def captured_age_seconds(captured: Any, now_monotonic: float | None = None) -> float:
    now = time.monotonic() if now_monotonic is None else now_monotonic
    captured_ns = getattr(captured, "captured_at_monotonic_ns", None)
    if captured_ns is not None:
        return max(now - int(captured_ns) / 1_000_000_000.0, 0.0)
    timestamp = getattr(captured, "timestamp", None)
    if timestamp is not None:
        return max(now - float(timestamp), 0.0)
    return 0.0


def run_target_centering_loop(
    *,
    vision: Any,
    gimbal: Any,
    servo: TargetCenterServo,
    loop_hz: float,
    timeout: float,
    stable_frames: int,
    stale_target_seconds: float,
    stale_target_step_scale: float,
    on_frame: FrameCallback | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    settled_frames = 0
    cached_frame: Any | None = None

    while time.monotonic() < deadline:
        loop_started_at = time.monotonic()
        vision_frame = vision.read_nowait_latest()
        if vision_frame is None and cached_frame is not None:
            if captured_age_seconds(cached_frame.captured, loop_started_at) <= stale_target_seconds:
                vision_frame = cached_frame
            else:
                cached_frame = None

        if vision_frame is None:
            sleep_for_loop_rate(loop_started_at, loop_hz)
            continue

        captured = vision_frame.captured
        points = vision_frame.points
        fresh = vision_frame is not cached_frame
        source_age = captured_age_seconds(captured, loop_started_at)
        if fresh and stale_target_seconds > 0 and source_age > stale_target_seconds:
            print(f"center: frame={captured.frame_id} stale source age={source_age * 1000.0:.1f}ms")
            cached_frame = None
            sleep_for_loop_rate(loop_started_at, loop_hz)
            continue

        update = servo.update(
            gimbal,
            points,
            step_scale=1.0 if fresh else stale_target_step_scale,
        )
        if fresh:
            cached_frame = vision_frame if update.valid else None
        if on_frame is not None:
            on_frame(
                captured,
                points,
                update,
                getattr(vision_frame, "target_corners_normalized", ()),
                getattr(vision_frame, "raw_target_center", None),
            )

        if update.settled and fresh:
            settled_frames += 1
        elif not update.settled:
            settled_frames = 0
        _log_target_update("center", captured.frame_id, fresh, update)
        if settled_frames >= stable_frames:
            print(f"center: settled for {settled_frames} frame(s)")
            return True
        sleep_for_loop_rate(loop_started_at, loop_hz)

    print(f"center: timeout after {timeout:.2f}s")
    return False


def run_target_tracking_loop(
    *,
    vision: Any,
    gimbal: Any,
    servo: TargetCenterServo,
    loop_hz: float,
    stale_target_seconds: float,
    stale_target_step_scale: float,
    stop_requested: StopCallback,
    on_frame: FrameCallback | None = None,
) -> bool:
    cached_frame: Any | None = None
    last_settled = False
    while not stop_requested():
        loop_started_at = time.monotonic()
        vision_frame = vision.read_nowait_latest()
        if vision_frame is None and cached_frame is not None:
            if captured_age_seconds(cached_frame.captured, loop_started_at) <= stale_target_seconds:
                vision_frame = cached_frame
            else:
                cached_frame = None

        if vision_frame is None:
            sleep_for_loop_rate(loop_started_at, loop_hz)
            continue

        captured = vision_frame.captured
        points = vision_frame.points
        fresh = vision_frame is not cached_frame
        source_age = captured_age_seconds(captured, loop_started_at)
        if fresh and stale_target_seconds > 0 and source_age > stale_target_seconds:
            print(f"track: frame={captured.frame_id} stale source age={source_age * 1000.0:.1f}ms")
            cached_frame = None
            last_settled = False
            sleep_for_loop_rate(loop_started_at, loop_hz)
            continue

        update = servo.update(
            gimbal,
            points,
            step_scale=1.0 if fresh else stale_target_step_scale,
        )
        if fresh:
            cached_frame = vision_frame if update.valid else None
        last_settled = update.settled
        if on_frame is not None:
            on_frame(
                captured,
                points,
                update,
                getattr(vision_frame, "target_corners_normalized", ()),
                getattr(vision_frame, "raw_target_center", None),
            )
        _log_target_update("track", captured.frame_id, fresh, update)
        sleep_for_loop_rate(loop_started_at, loop_hz)

    print("track: stop requested")
    return last_settled


def _log_target_update(phase: str, frame_id: int, fresh: bool, update: AimUpdate) -> None:
    if update.step is None:
        print(f"{phase}: frame={frame_id} target_center invalid: {update.reason}")
        return
    print(
        f"{phase}: frame={frame_id}{'' if fresh else ' stale'} "
        f"err=({update.step.error.x:+.4f},{update.step.error.y:+.4f}) "
        f"step=({update.step.x_delta_deg:+.3f},{update.step.y_delta_deg:+.3f}) "
        f"settled={update.settled}"
    )


def run_laser_alignment_loop(
    *,
    vision: Any,
    gimbal: Any,
    feedback: GimbalFeedbackReader,
    servo: LaserAlignmentServo,
    loop_hz: float,
    stop_requested: StopCallback,
    deadline: float | None = None,
    stable_frames: int | None = None,
    on_frame: FrameCallback | None = None,
    phase: str,
) -> bool:
    settled_frames = 0
    scheduler = FixedDeadlineScheduler(loop_hz)
    while not stop_requested() and (deadline is None or time.monotonic() < deadline):
        loop_started_at = time.monotonic()
        angles = feedback.read()
        if angles is None:
            settled_frames = 0
            servo.handle_feedback_loss()
            wait_for_motor_deadline(scheduler, loop_started_at, phase=phase)
            continue
        if feedback.recovered_this_read:
            gimbal.sync_commanded_angles(angles)
        servo.record_angles(angles)

        vision_frame = vision.read_nowait_latest()
        if vision_frame is not None:
            aim_update = servo.accept_vision(
                vision_frame,
                angles,
                now_monotonic_ns=time.monotonic_ns(),
            )
            if on_frame is not None:
                on_frame(
                    vision_frame.captured,
                    vision_frame.points,
                    aim_update,
                    vision_frame.target_corners_normalized,
                    vision_frame.raw_target_center,
                )
            _log_alignment_update(phase, vision_frame, angles, aim_update, servo)
            if aim_update.valid and aim_update.settled:
                settled_frames += 1
            else:
                settled_frames = 0
            if stable_frames is not None and settled_frames >= stable_frames:
                print(f"{phase}: visually settled for {settled_frames} fresh frame(s)")
                return True

        motor_update = servo.compute_motor_update(angles, now=time.monotonic())
        if motor_update.reason == "target_expired":
            print(f"{phase}: visual angle target expired; holding position")
        elif motor_update.target is not None:
            command = (
                "hold"
                if motor_update.x_command_deg is None or motor_update.y_command_deg is None
                else f"({motor_update.x_command_deg:+.3f},{motor_update.y_command_deg:+.3f})deg"
            )
            print(
                f"{phase}: pid frame={motor_update.target.source_frame_id} "
                f"remaining=({motor_update.x_error_deg:+.3f},{motor_update.y_error_deg:+.3f})deg "
                f"encoder_moved=({motor_update.x_cumulative_moved_deg:+.3f},"
                f"{motor_update.y_cumulative_moved_deg:+.3f})deg "
                f"output=({motor_update.x_delta_deg:+.3f},{motor_update.y_delta_deg:+.3f})deg "
                f"command={command} settled={motor_update.settled}"
            )
        if motor_update.x_command_deg is not None and motor_update.y_command_deg is not None:
            gimbal.move_to(motor_update.x_command_deg, motor_update.y_command_deg)
        wait_for_motor_deadline(scheduler, loop_started_at, phase=phase)

    if deadline is not None and time.monotonic() >= deadline:
        print(f"{phase}: timeout")
    else:
        print(f"{phase}: stop requested")
    return False


def _log_alignment_update(
    phase: str,
    vision_frame: Any,
    angles: Any,
    aim_update: AimUpdate,
    servo: LaserAlignmentServo,
) -> None:
    if not aim_update.valid or aim_update.step is None:
        print(
            f"{phase}: frame={vision_frame.source_frame_id} "
            f"vision target invalid: {aim_update.reason}"
        )
        return
    target = servo.target
    assert target is not None
    print(
        f"{phase}: frame={vision_frame.source_frame_id} "
        f"visual_err=({target.visual_error_x:+.4f},{target.visual_error_y:+.4f}) "
        f"raw_correction=({target.raw_x_correction_deg:+.3f},{target.raw_y_correction_deg:+.3f})deg "
        f"target_velocity=({target.target_velocity_x_deg_s:+.3f},{target.target_velocity_y_deg_s:+.3f})deg/s "
        f"feedforward=({target.feedforward_x_deg:+.3f},{target.feedforward_y_deg:+.3f})deg "
        f"horizon={target.prediction_horizon_seconds * 1000.0:.1f}ms "
        f"correction=({target.x_correction_deg:+.3f},{target.y_correction_deg:+.3f})deg "
        f"angle_target=({target.desired_x_deg:+.3f},{target.desired_y_deg:+.3f})deg "
        f"feedback=({angles.x_deg:+.3f},{angles.y_deg:+.3f})deg source=encoder "
        f"distance={vision_frame.target_distance_cm:.1f}cm"
    )
