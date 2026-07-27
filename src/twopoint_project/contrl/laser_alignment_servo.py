"""Two-rate laser alignment control using vision targets and motor feedback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Sequence

from twopoint_project.contrl.target_center_servo import (
    AimUpdate,
    CenteringError,
    FeedForwardConfig,
    GimbalStep,
    PIDAxisGains,
    PIDAxisState,
    PIDOutput,
    PointPrediction,
    TargetCenterObservation,
    clamp,
)
from twopoint_project.f32c.gimbal import GimbalAngles
from twopoint_project.vision.pipeline import VisionResult
from twopoint_project.vision3.laser_area_mapping import LASER_POINT_LABEL


TARGET_CENTER_LABEL = "target_center"


@dataclass(frozen=True)
class TrackClosedLoopConfig:
    """Parameters owned by the encoder-feedback laser alignment controller."""

    motor_loop_hz: float = 50.0
    feedback_timeout: float = 0.003
    max_vision_age_seconds: float = 0.3
    angle_deadband_deg: float = 0.1
    x_angle_gain_deg: float = 10.0
    y_angle_gain_deg: float = -10.0
    max_visual_correction_deg: float = 10.0
    feedforward: FeedForwardConfig = FeedForwardConfig()
    x_pid: PIDAxisGains = PIDAxisGains(
        kp=0.8,
        ki=0.0,
        kd=0.02,
        integral_limit=0.5,
        output_limit_deg=2.0,
    )
    y_pid: PIDAxisGains = PIDAxisGains(
        kp=0.8,
        ki=0.0,
        kd=0.02,
        integral_limit=0.5,
        output_limit_deg=2.0,
    )


@dataclass(frozen=True)
class AngularTarget:
    source_frame_id: int
    captured_at_monotonic_ns: int
    visual_error_x: float
    visual_error_y: float
    raw_x_correction_deg: float
    raw_y_correction_deg: float
    feedforward_x_deg: float
    feedforward_y_deg: float
    target_velocity_x_deg_s: float
    target_velocity_y_deg_s: float
    prediction_horizon_seconds: float
    x_correction_deg: float
    y_correction_deg: float
    capture_x_deg: float
    capture_y_deg: float
    desired_x_deg: float
    desired_y_deg: float


@dataclass(frozen=True)
class MotorControlUpdate:
    target: AngularTarget | None
    actual: GimbalAngles
    x_error_deg: float
    y_error_deg: float
    x_cumulative_moved_deg: float
    y_cumulative_moved_deg: float
    x_command_deg: float | None
    y_command_deg: float | None
    x_delta_deg: float
    y_delta_deg: float
    settled: bool
    reason: str | None = None
    x_output: PIDOutput | None = None
    y_output: PIDOutput | None = None


class AngleHistory:
    def __init__(self, max_samples: int = 256) -> None:
        self._samples: deque[GimbalAngles] = deque(maxlen=max_samples)
        self._reject_before_first = False

    def append(self, angles: GimbalAngles) -> None:
        if self._samples and angles.sampled_at_monotonic_ns < self._samples[-1].sampled_at_monotonic_ns:
            raise ValueError("gimbal angle timestamps must be monotonic")
        self._samples.append(angles)

    def clear(self, *, reject_before_first: bool = False) -> None:
        self._samples.clear()
        self._reject_before_first = reject_before_first

    def at(self, timestamp_ns: int) -> GimbalAngles | None:
        if not self._samples:
            return None
        if timestamp_ns < self._samples[0].sampled_at_monotonic_ns:
            return None if self._reject_before_first else self._samples[0]
        if timestamp_ns == self._samples[0].sampled_at_monotonic_ns:
            return self._samples[0]
        if timestamp_ns >= self._samples[-1].sampled_at_monotonic_ns:
            return self._samples[-1]

        previous = self._samples[0]
        for current in tuple(self._samples)[1:]:
            if current.sampled_at_monotonic_ns >= timestamp_ns:
                interval = current.sampled_at_monotonic_ns - previous.sampled_at_monotonic_ns
                ratio = 0.0 if interval <= 0 else (
                    (timestamp_ns - previous.sampled_at_monotonic_ns) / interval
                )
                return GimbalAngles(
                    x_deg=previous.x_deg + (current.x_deg - previous.x_deg) * ratio,
                    y_deg=previous.y_deg + (current.y_deg - previous.y_deg) * ratio,
                    sampled_at_monotonic_ns=timestamp_ns,
                    feedback_valid=previous.feedback_valid and current.feedback_valid,
                )
            previous = current
        return self._samples[-1]


def select_point(
    points: Sequence[PointPrediction],
    label: str,
) -> PointPrediction | None:
    candidates = [
        point
        for point in points
        if point["label"] == label
    ]
    return max(candidates, key=lambda point: point["confidence"], default=None)


class LaserAlignmentServo:
    """Turn fresh target/laser observations into an encoder-feedback angle target."""

    def __init__(
        self,
        *,
        config: TrackClosedLoopConfig,
        visual_deadband: float,
    ) -> None:
        self.config = config
        self.visual_deadband = abs(visual_deadband)
        self.history = AngleHistory()
        self.target: AngularTarget | None = None
        self._x_state = PIDAxisState()
        self._y_state = PIDAxisState()
        self._previous_raw_target_x_deg: float | None = None
        self._previous_raw_target_y_deg: float | None = None
        self._previous_vision_capture_ns: int | None = None
        self._target_velocity_x_deg_s = 0.0
        self._target_velocity_y_deg_s = 0.0

    def record_angles(self, angles: GimbalAngles) -> None:
        self.history.append(angles)

    def clear_target(self) -> None:
        self.target = None
        self._x_state.reset()
        self._y_state.reset()
        self.reset_feedforward()

    def reset_feedforward(self) -> None:
        self._previous_raw_target_x_deg = None
        self._previous_raw_target_y_deg = None
        self._previous_vision_capture_ns = None
        self._target_velocity_x_deg_s = 0.0
        self._target_velocity_y_deg_s = 0.0

    def handle_feedback_loss(self) -> None:
        """Stop the active correction and discard angles spanning a feedback gap."""
        self.clear_target()
        self.history.clear(reject_before_first=True)

    def apply_feedforward(
        self,
        *,
        capture_angles: GimbalAngles,
        raw_x_correction_deg: float,
        raw_y_correction_deg: float,
        captured_at_monotonic_ns: int,
        vision_age_seconds: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Predict the absolute angular target after removing gimbal self-motion."""
        config = self.config.feedforward
        raw_target_x = capture_angles.x_deg + raw_x_correction_deg
        raw_target_y = capture_angles.y_deg + raw_y_correction_deg
        velocity_x = self._target_velocity_x_deg_s
        velocity_y = self._target_velocity_y_deg_s

        previous_time = self._previous_vision_capture_ns
        previous_x = self._previous_raw_target_x_deg
        previous_y = self._previous_raw_target_y_deg
        if previous_time is not None and previous_x is not None and previous_y is not None:
            dt = (captured_at_monotonic_ns - previous_time) / 1_000_000_000.0
            gap_limit = self.config.max_vision_age_seconds
            if dt > 0 and (gap_limit <= 0 or dt <= gap_limit):
                raw_velocity_x = (raw_target_x - previous_x) / dt
                raw_velocity_y = (raw_target_y - previous_y) / dt
                velocity_limit = abs(config.max_velocity)
                if velocity_limit > 0:
                    x_limit_deg_s = velocity_limit * abs(self.config.x_angle_gain_deg)
                    y_limit_deg_s = velocity_limit * abs(self.config.y_angle_gain_deg)
                    raw_velocity_x = clamp(
                        raw_velocity_x,
                        -x_limit_deg_s,
                        x_limit_deg_s,
                    )
                    raw_velocity_y = clamp(
                        raw_velocity_y,
                        -y_limit_deg_s,
                        y_limit_deg_s,
                    )
                alpha = clamp(config.velocity_alpha, 0.0, 1.0)
                velocity_x += (raw_velocity_x - velocity_x) * alpha
                velocity_y += (raw_velocity_y - velocity_y) * alpha
            else:
                velocity_x = 0.0
                velocity_y = 0.0
        else:
            velocity_x = 0.0
            velocity_y = 0.0

        self._previous_raw_target_x_deg = raw_target_x
        self._previous_raw_target_y_deg = raw_target_y
        self._previous_vision_capture_ns = captured_at_monotonic_ns
        self._target_velocity_x_deg_s = velocity_x
        self._target_velocity_y_deg_s = velocity_y

        if not config.enabled:
            return (
                raw_x_correction_deg,
                raw_y_correction_deg,
                0.0,
                0.0,
                velocity_x,
                velocity_y,
                0.0,
            )

        horizon = vision_age_seconds + config.lead_time
        feedforward_x = velocity_x * horizon
        feedforward_y = velocity_y * horizon
        prediction_limit = abs(config.max_prediction_error)
        if prediction_limit > 0:
            x_limit_deg = prediction_limit * abs(self.config.x_angle_gain_deg)
            y_limit_deg = prediction_limit * abs(self.config.y_angle_gain_deg)
            feedforward_x = clamp(feedforward_x, -x_limit_deg, x_limit_deg)
            feedforward_y = clamp(feedforward_y, -y_limit_deg, y_limit_deg)

        return (
            raw_x_correction_deg + feedforward_x,
            raw_y_correction_deg + feedforward_y,
            feedforward_x,
            feedforward_y,
            velocity_x,
            velocity_y,
            horizon,
        )

    def accept_vision(
        self,
        vision: VisionResult,
        actual: GimbalAngles,
        *,
        now_monotonic_ns: int,
    ) -> AimUpdate:
        age_seconds = max(
            now_monotonic_ns - vision.captured_at_monotonic_ns,
            0,
        ) / 1_000_000_000.0
        if age_seconds > self.config.max_vision_age_seconds:
            return AimUpdate(False, False, False, "vision_result_stale", None, None)

        target_point = select_point(
            vision.points,
            TARGET_CENTER_LABEL,
        )
        laser_point = select_point(
            vision.points,
            LASER_POINT_LABEL,
        )
        if target_point is None:
            return AimUpdate(False, False, False, "target_center_unavailable", None, None)
        if (
            laser_point is None
            or laser_point["confidence"] <= 0.0
            or vision.target_distance_cm is None
        ):
            return AimUpdate(False, False, False, "laser_or_distance_unavailable", None, None)

        error_x = float(target_point["x"] - laser_point["x"])
        error_y = float(target_point["y"] - laser_point["y"])
        limit = abs(self.config.max_visual_correction_deg)
        raw_correction_x = clamp(
            error_x * self.config.x_angle_gain_deg,
            -limit,
            limit,
        )
        raw_correction_y = clamp(
            error_y * self.config.y_angle_gain_deg,
            -limit,
            limit,
        )
        capture_angles = self.history.at(vision.captured_at_monotonic_ns)
        if capture_angles is None:
            return AimUpdate(
                False,
                False,
                False,
                "angle_feedback_unavailable_at_capture",
                None,
                None,
            )
        (
            correction_x,
            correction_y,
            feedforward_x,
            feedforward_y,
            target_velocity_x,
            target_velocity_y,
            prediction_horizon,
        ) = self.apply_feedforward(
            capture_angles=capture_angles,
            raw_x_correction_deg=raw_correction_x,
            raw_y_correction_deg=raw_correction_y,
            captured_at_monotonic_ns=vision.captured_at_monotonic_ns,
            vision_age_seconds=age_seconds,
        )
        correction_x = clamp(correction_x, -limit, limit)
        correction_y = clamp(correction_y, -limit, limit)
        self.target = AngularTarget(
            source_frame_id=vision.source_frame_id,
            captured_at_monotonic_ns=vision.captured_at_monotonic_ns,
            visual_error_x=error_x,
            visual_error_y=error_y,
            raw_x_correction_deg=raw_correction_x,
            raw_y_correction_deg=raw_correction_y,
            feedforward_x_deg=feedforward_x,
            feedforward_y_deg=feedforward_y,
            target_velocity_x_deg_s=target_velocity_x,
            target_velocity_y_deg_s=target_velocity_y,
            prediction_horizon_seconds=prediction_horizon,
            x_correction_deg=correction_x,
            y_correction_deg=correction_y,
            capture_x_deg=capture_angles.x_deg,
            capture_y_deg=capture_angles.y_deg,
            desired_x_deg=capture_angles.x_deg + correction_x,
            desired_y_deg=capture_angles.y_deg + correction_y,
        )
        # A new vision frame changes the angle setpoint discontinuously. Resetting
        # the motor-loop history avoids derivative kick from that target change.
        self._x_state.reset()
        self._y_state.reset()

        visual_error = CenteringError(
            x=error_x,
            y=error_y,
            distance=hypot(error_x, error_y),
        )
        settled = abs(error_x) <= self.visual_deadband and abs(error_y) <= self.visual_deadband
        observation = TargetCenterObservation(
            x=float(target_point["x"]),
            y=float(target_point["y"]),
            confidence=float(target_point["confidence"]),
        )
        return AimUpdate(
            valid=True,
            moved=False,
            settled=settled,
            reason=None,
            target=observation,
            step=GimbalStep(
                x_delta_deg=correction_x,
                y_delta_deg=correction_y,
                error=visual_error,
                settled=settled,
            ),
        )

    @staticmethod
    def _bounded_delta(error: float, output: float) -> float:
        return clamp(output, min(0.0, error), max(0.0, error))

    def compute_motor_update(
        self,
        actual: GimbalAngles,
        *,
        now: float,
    ) -> MotorControlUpdate:
        target = self.target
        if target is None:
            return MotorControlUpdate(
                target=None,
                actual=actual,
                x_error_deg=0.0,
                y_error_deg=0.0,
                x_cumulative_moved_deg=0.0,
                y_cumulative_moved_deg=0.0,
                x_command_deg=None,
                y_command_deg=None,
                x_delta_deg=0.0,
                y_delta_deg=0.0,
                settled=False,
                reason="no_target",
            )

        target_age_seconds = max(
            now - target.captured_at_monotonic_ns / 1_000_000_000.0,
            0.0,
        )
        if target_age_seconds > self.config.max_vision_age_seconds:
            self.clear_target()
            return MotorControlUpdate(
                target=None,
                actual=actual,
                x_error_deg=0.0,
                y_error_deg=0.0,
                x_cumulative_moved_deg=0.0,
                y_cumulative_moved_deg=0.0,
                x_command_deg=None,
                y_command_deg=None,
                x_delta_deg=0.0,
                y_delta_deg=0.0,
                settled=False,
                reason="target_expired",
            )

        # Each visual result defines a total angular correction at capture time.
        # The 50 Hz motor loop subtracts the encoder-measured cumulative motion
        # since that capture, leaving the angle that this PID cycle still needs
        # to execute. This is algebraically equivalent to desired - actual, but
        # keeps the control contract explicit and never relies on commanded motion.
        x_cumulative_moved = actual.x_deg - target.capture_x_deg
        y_cumulative_moved = actual.y_deg - target.capture_y_deg
        x_error = target.x_correction_deg - x_cumulative_moved
        y_error = target.y_correction_deg - y_cumulative_moved
        x_settled = abs(x_error) <= self.config.angle_deadband_deg
        y_settled = abs(y_error) <= self.config.angle_deadband_deg

        if x_settled:
            self._x_state.reset()
            x_output = None
            x_delta = 0.0
        else:
            x_output = self._x_state.update(x_error, self.config.x_pid, now)
            x_delta = self._bounded_delta(x_error, x_output.clamped)

        if y_settled:
            self._y_state.reset()
            y_output = None
            y_delta = 0.0
        else:
            y_output = self._y_state.update(y_error, self.config.y_pid, now)
            y_delta = self._bounded_delta(y_error, y_output.clamped)

        settled = x_settled and y_settled
        return MotorControlUpdate(
            target=target,
            actual=actual,
            x_error_deg=x_error,
            y_error_deg=y_error,
            x_cumulative_moved_deg=x_cumulative_moved,
            y_cumulative_moved_deg=y_cumulative_moved,
            x_command_deg=None if settled else actual.x_deg + x_delta,
            y_command_deg=None if settled else actual.y_deg + y_delta,
            x_delta_deg=x_delta,
            y_delta_deg=y_delta,
            settled=settled,
            x_output=x_output,
            y_output=y_output,
        )
