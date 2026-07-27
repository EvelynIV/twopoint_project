from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from twopoint_project.config import TrackClosedLoopConfig
from twopoint_project.contrl.laser_alignment_servo import LaserAlignmentServo
from twopoint_project.contrl.target_center_servo import FeedForwardConfig, PIDAxisGains
from twopoint_project.f32c.gimbal import GimbalAngles
from twopoint_project.vision.pipeline import VisionResult


def make_vision(
    *,
    frame_id: int,
    captured_at_ns: int,
    target_x: float,
    target_y: float,
    laser_x: float,
    laser_y: float,
    distance_cm: float | None = 100.0,
) -> VisionResult:
    return VisionResult(
        camera_id="test",
        stream_generation=1,
        source_frame_id=frame_id,
        captured_at_monotonic_ns=captured_at_ns,
        inference_started_monotonic_ns=captured_at_ns + 1,
        inference_finished_monotonic_ns=captured_at_ns + 2,
        captured=SimpleNamespace(frame_id=frame_id, frame_bgr=object()),
        points=[
            {"label": "target_center", "x": target_x, "y": target_y, "confidence": 1.0},
            {"label": "laser_point", "x": laser_x, "y": laser_y, "confidence": 1.0},
        ],
        target_distance_cm=distance_cm,
    )


def make_config() -> TrackClosedLoopConfig:
    gains = PIDAxisGains(kp=0.5, output_limit_deg=2.0)
    return TrackClosedLoopConfig(
        max_vision_age_seconds=1.0,
        angle_deadband_deg=0.01,
        x_angle_gain_deg=10.0,
        y_angle_gain_deg=-10.0,
        max_visual_correction_deg=5.0,
        x_pid=gains,
        y_pid=gains,
    )


class LaserAlignmentServoTest(unittest.TestCase):
    def test_feedforward_predicts_moving_absolute_angle_target(self) -> None:
        config = replace(
            make_config(),
            feedforward=FeedForwardConfig(
                enabled=True,
                lead_time=0.05,
                max_prediction_error=0.05,
                max_velocity=2.0,
                velocity_alpha=1.0,
            ),
        )
        servo = LaserAlignmentServo(
            config=config,
            visual_deadband=0.006,
        )
        base = 6_000_000_000
        first_angles = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(first_angles)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            first_angles,
            now_monotonic_ns=base + 20_000_000,
        )
        assert servo.target is not None
        self.assertAlmostEqual(servo.target.raw_x_correction_deg, 1.0)
        self.assertAlmostEqual(servo.target.feedforward_x_deg, 0.0)
        self.assertAlmostEqual(servo.target.x_correction_deg, 1.0)

        second_time = base + 100_000_000
        second_angles = GimbalAngles(0.0, 0.0, second_time)
        servo.record_angles(second_angles)
        servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=second_time,
                target_x=0.62,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            second_angles,
            now_monotonic_ns=second_time + 20_000_000,
        )

        assert servo.target is not None
        self.assertAlmostEqual(servo.target.raw_x_correction_deg, 1.2)
        self.assertAlmostEqual(servo.target.target_velocity_x_deg_s, 2.0)
        self.assertAlmostEqual(servo.target.prediction_horizon_seconds, 0.07)
        self.assertAlmostEqual(servo.target.feedforward_x_deg, 0.14)
        self.assertAlmostEqual(servo.target.x_correction_deg, 1.34)

    def test_feedforward_does_not_treat_gimbal_self_motion_as_target_motion(self) -> None:
        config = replace(
            make_config(),
            feedforward=FeedForwardConfig(
                enabled=True,
                lead_time=0.05,
                max_prediction_error=0.05,
                max_velocity=2.0,
                velocity_alpha=1.0,
            ),
        )
        servo = LaserAlignmentServo(
            config=config,
            visual_deadband=0.006,
        )
        base = 7_000_000_000
        first_angles = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(first_angles)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            first_angles,
            now_monotonic_ns=base,
        )

        second_time = base + 100_000_000
        second_angles = GimbalAngles(0.5, 0.0, second_time)
        servo.record_angles(second_angles)
        servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=second_time,
                target_x=0.55,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            second_angles,
            now_monotonic_ns=second_time,
        )

        assert servo.target is not None
        self.assertAlmostEqual(servo.target.raw_x_correction_deg, 0.5)
        self.assertAlmostEqual(servo.target.capture_x_deg, 0.5)
        self.assertAlmostEqual(servo.target.target_velocity_x_deg_s, 0.0)
        self.assertAlmostEqual(servo.target.feedforward_x_deg, 0.0)
        self.assertAlmostEqual(servo.target.desired_x_deg, 1.0)

    def test_feedforward_prediction_is_bounded(self) -> None:
        config = replace(
            make_config(),
            max_visual_correction_deg=10.0,
            feedforward=FeedForwardConfig(
                enabled=True,
                lead_time=0.1,
                max_prediction_error=0.02,
                max_velocity=100.0,
                velocity_alpha=1.0,
            ),
        )
        servo = LaserAlignmentServo(
            config=config,
            visual_deadband=0.006,
        )
        base = 8_000_000_000
        first_angles = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(first_angles)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.5,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            first_angles,
            now_monotonic_ns=base,
        )

        second_time = base + 100_000_000
        second_angles = GimbalAngles(0.0, 0.0, second_time)
        servo.record_angles(second_angles)
        servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=second_time,
                target_x=0.8,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            second_angles,
            now_monotonic_ns=second_time,
        )

        assert servo.target is not None
        self.assertAlmostEqual(servo.target.raw_x_correction_deg, 3.0)
        self.assertAlmostEqual(servo.target.feedforward_x_deg, 0.2)
        self.assertAlmostEqual(servo.target.x_correction_deg, 3.2)

    def test_visual_target_uses_motor_angle_at_capture_time(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            visual_deadband=0.006,
        )
        base = 1_000_000_000
        servo.record_angles(GimbalAngles(0.0, 2.0, base))
        servo.record_angles(GimbalAngles(2.0, 4.0, base + 100_000_000))
        current = GimbalAngles(2.0, 4.0, base + 100_000_000)

        update = servo.accept_vision(
            make_vision(
                frame_id=7,
                captured_at_ns=base + 50_000_000,
                target_x=0.6,
                target_y=0.4,
                laser_x=0.5,
                laser_y=0.5,
            ),
            current,
            now_monotonic_ns=base + 100_000_000,
        )

        self.assertTrue(update.valid)
        assert servo.target is not None
        self.assertAlmostEqual(servo.target.x_correction_deg, 1.0)
        self.assertAlmostEqual(servo.target.y_correction_deg, 1.0)
        self.assertAlmostEqual(servo.target.desired_x_deg, 2.0)
        self.assertAlmostEqual(servo.target.desired_y_deg, 4.0)

    def test_motor_pid_recomputes_remaining_angle_between_vision_frames(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            visual_deadband=0.006,
        )
        base = 2_000_000_000
        initial = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(initial)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            initial,
            now_monotonic_ns=base,
        )

        first = servo.compute_motor_update(initial, now=2.0)
        halfway = GimbalAngles(0.5, 0.0, base + 33_000_000)
        second = servo.compute_motor_update(halfway, now=2.033)

        self.assertAlmostEqual(first.x_error_deg, 1.0)
        self.assertAlmostEqual(first.x_cumulative_moved_deg, 0.0)
        self.assertAlmostEqual(first.x_delta_deg, 0.5)
        self.assertAlmostEqual(first.x_command_deg or 0.0, 0.5)
        self.assertAlmostEqual(second.x_error_deg, 0.5)
        self.assertAlmostEqual(second.x_cumulative_moved_deg, 0.5)
        self.assertAlmostEqual(second.x_delta_deg, 0.25)
        self.assertAlmostEqual(second.x_command_deg or 0.0, 0.75)

    def test_motor_pid_uses_actual_cumulative_motion_not_commanded_motion(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            visual_deadband=0.006,
        )
        base = 2_500_000_000
        initial = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(initial)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            initial,
            now_monotonic_ns=base,
        )

        first = servo.compute_motor_update(initial, now=2.5)
        lagging = GimbalAngles(0.2, 0.0, base + 20_000_000)
        second = servo.compute_motor_update(lagging, now=2.52)
        later = GimbalAngles(0.3, 0.0, base + 40_000_000)
        third = servo.compute_motor_update(later, now=2.54)

        self.assertAlmostEqual(first.x_command_deg or 0.0, 0.5)
        self.assertAlmostEqual(second.x_cumulative_moved_deg, 0.2)
        self.assertAlmostEqual(second.x_error_deg, 0.8)
        self.assertAlmostEqual(third.x_cumulative_moved_deg, 0.3)
        self.assertAlmostEqual(third.x_error_deg, 0.7)

    def test_new_visual_frame_replaces_target_and_resets_pid_history(self) -> None:
        gains = PIDAxisGains(
            kp=0.5,
            ki=1.0,
            kd=1.0,
            integral_limit=1.0,
            output_limit_deg=10.0,
        )
        servo = LaserAlignmentServo(
            config=replace(make_config(), x_pid=gains, y_pid=gains),
            visual_deadband=0.006,
        )
        base = 3_000_000_000
        initial = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(initial)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            initial,
            now_monotonic_ns=base,
        )
        servo.compute_motor_update(initial, now=3.0)
        servo.compute_motor_update(GimbalAngles(0.1, 0.0, base + 20_000_000), now=3.02)

        replacement_angles = GimbalAngles(0.1, 0.0, base + 40_000_000)
        servo.record_angles(replacement_angles)
        servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=base + 40_000_000,
                target_x=0.4,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            replacement_angles,
            now_monotonic_ns=base + 40_000_000,
        )
        update = servo.compute_motor_update(replacement_angles, now=3.04)

        self.assertEqual(update.target.source_frame_id if update.target else None, 2)
        self.assertIsNotNone(update.x_output)
        assert update.x_output is not None
        self.assertAlmostEqual(update.x_output.i, 0.0)
        self.assertAlmostEqual(update.x_output.d, 0.0)
        self.assertAlmostEqual(update.x_output.derivative, 0.0)

    def test_visual_target_expires_from_capture_time(self) -> None:
        servo = LaserAlignmentServo(
            config=replace(make_config(), max_vision_age_seconds=0.3),
            visual_deadband=0.006,
        )
        base = 4_000_000_000
        actual = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(actual)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            actual,
            now_monotonic_ns=base,
        )

        still_valid = servo.compute_motor_update(actual, now=4.3)
        expired = servo.compute_motor_update(actual, now=4.3001)

        self.assertIsNotNone(still_valid.x_command_deg)
        self.assertEqual(expired.reason, "target_expired")
        self.assertIsNone(expired.x_command_deg)
        self.assertIsNone(servo.target)

    def test_feedback_gap_requires_a_frame_captured_after_recovery(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            visual_deadband=0.006,
        )
        base = 5_000_000_000
        servo.record_angles(GimbalAngles(0.0, 0.0, base))
        servo.handle_feedback_loss()
        recovered = GimbalAngles(0.2, 0.0, base + 100_000_000)
        servo.record_angles(recovered)

        during_gap = servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base + 50_000_000,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            recovered,
            now_monotonic_ns=base + 100_000_000,
        )
        after_recovery = servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=base + 100_000_000,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            recovered,
            now_monotonic_ns=base + 100_000_000,
        )

        self.assertFalse(during_gap.valid)
        self.assertEqual(during_gap.reason, "angle_feedback_unavailable_at_capture")
        self.assertTrue(after_recovery.valid)

    def test_invalid_distance_does_not_replace_existing_angle_target(self) -> None:
        servo = LaserAlignmentServo(
            config=make_config(),
            visual_deadband=0.006,
        )
        base = 3_000_000_000
        actual = GimbalAngles(0.0, 0.0, base)
        servo.record_angles(actual)
        servo.accept_vision(
            make_vision(
                frame_id=1,
                captured_at_ns=base,
                target_x=0.6,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
            ),
            actual,
            now_monotonic_ns=base,
        )
        original_target = servo.target

        invalid = servo.accept_vision(
            make_vision(
                frame_id=2,
                captured_at_ns=base + 10_000_000,
                target_x=0.4,
                target_y=0.5,
                laser_x=0.5,
                laser_y=0.5,
                distance_cm=None,
            ),
            actual,
            now_monotonic_ns=base + 10_000_000,
        )

        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.reason, "laser_or_distance_unavailable")
        self.assertIs(servo.target, original_target)


if __name__ == "__main__":
    unittest.main()
