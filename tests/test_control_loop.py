from __future__ import annotations

from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

from tests.center_fakes import FakeGimbal, FakeVisionFrames, StopAfterCalls
from twopoint_project.contrl.control_loop import (
    run_laser_alignment_loop,
    run_target_centering_loop,
    run_target_tracking_loop,
)
from twopoint_project.contrl.target_center_servo import TargetCenterServo


class ControlLoopTest(unittest.TestCase):
    def test_centering_returns_true_after_required_stable_frame(self) -> None:
        gimbal = FakeGimbal()
        vision = FakeVisionFrames([
            SimpleNamespace(
                captured=SimpleNamespace(frame_id=1),
                points=[{"label": "target_center", "x": 0.5, "y": 0.5, "confidence": 1.0}],
            )
        ])

        centered = run_target_centering_loop(
            vision=vision,
            gimbal=gimbal,
            servo=TargetCenterServo(),
            loop_hz=0,
            timeout=1.0,
            stable_frames=1,
            stale_target_seconds=0.3,
            stale_target_step_scale=0.5,
        )

        self.assertTrue(centered)

    def test_motor_loop_does_not_move_without_encoder_feedback(self) -> None:
        gimbal = FakeGimbal()
        feedback = SimpleNamespace(read=lambda: None, recovered_this_read=False)
        servo = Mock()

        run_laser_alignment_loop(
            vision=FakeVisionFrames([]),
            gimbal=gimbal,
            feedback=feedback,
            servo=servo,
            loop_hz=0,
            stop_requested=StopAfterCalls(2),
            phase="test",
        )

        self.assertEqual(gimbal.moves, [])
        servo.handle_feedback_loss.assert_called_once_with()

    def test_tracking_reuses_last_valid_frame_with_reduced_step(self) -> None:
        gimbal = FakeGimbal()
        vision = FakeVisionFrames([
            SimpleNamespace(
                captured=SimpleNamespace(frame_id=1),
                points=[{"label": "target_center", "x": 0.6, "y": 0.45, "confidence": 1.0}],
            )
        ])

        run_target_tracking_loop(
            vision=vision,
            gimbal=gimbal,
            servo=TargetCenterServo(
                x_gain_deg=10.0,
                y_gain_deg=-10.0,
                max_step_deg=2.0,
                deadband=0.01,
            ),
            loop_hz=0,
            stale_target_seconds=1.0,
            stale_target_step_scale=0.5,
            stop_requested=StopAfterCalls(3),
        )

        self.assertEqual(len(gimbal.moves), 2)
        self.assertAlmostEqual(gimbal.moves[0][0], 1.0)
        self.assertAlmostEqual(gimbal.moves[0][1], 0.5)
        self.assertAlmostEqual(gimbal.moves[1][0], 0.5)
        self.assertAlmostEqual(gimbal.moves[1][1], 0.25)

    def test_tracking_rejects_old_source_frame(self) -> None:
        gimbal = FakeGimbal()
        vision = FakeVisionFrames([
            SimpleNamespace(
                captured=SimpleNamespace(
                    frame_id=1,
                    captured_at_monotonic_ns=time.monotonic_ns() - 1_000_000_000,
                ),
                points=[{"label": "target_center", "x": 0.6, "y": 0.45, "confidence": 1.0}],
            )
        ])

        run_target_tracking_loop(
            vision=vision,
            gimbal=gimbal,
            servo=TargetCenterServo(),
            loop_hz=0,
            stale_target_seconds=0.1,
            stale_target_step_scale=0.5,
            stop_requested=StopAfterCalls(2),
        )

        self.assertEqual(gimbal.moves, [])


if __name__ == "__main__":
    unittest.main()
