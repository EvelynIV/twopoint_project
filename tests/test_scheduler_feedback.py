from __future__ import annotations

import unittest
from unittest.mock import Mock

from tests.center_fakes import FakeClock
from twopoint_project.contrl.feedback import GimbalFeedbackReader
from twopoint_project.contrl.scheduler import FixedDeadlineScheduler
from twopoint_project.f32c.gimbal import GimbalAngles


class SchedulerFeedbackTest(unittest.TestCase):
    def test_fixed_deadline_scheduler_skips_missed_slots(self) -> None:
        clock = FakeClock()
        scheduler = FixedDeadlineScheduler(50.0, clock=clock, sleeper=clock.sleep)

        started = clock()
        clock.advance(0.005)
        first = scheduler.wait_for_next(started)
        self.assertFalse(first.overran)
        self.assertAlmostEqual(clock.now, 0.02)

        started = clock()
        clock.advance(0.045)
        second = scheduler.wait_for_next(started)
        self.assertTrue(second.overran)
        self.assertEqual(second.skipped_deadlines, 2)

    def test_feedback_reader_pauses_and_reports_recovery(self) -> None:
        recovered = GimbalAngles(1.0, 2.0, 123)
        gimbal = Mock()
        gimbal.read_angles.side_effect = [TimeoutError("missing"), recovered]
        reader = GimbalFeedbackReader(gimbal, timeout=0.01)

        self.assertIsNone(reader.read())
        gimbal.commanded_angles.assert_not_called()
        self.assertIs(reader.read(), recovered)
        self.assertTrue(reader.recovered_this_read)

    def test_feedback_reader_rejects_non_encoder_data(self) -> None:
        gimbal = Mock()
        gimbal.read_angles.return_value = GimbalAngles(
            1.0,
            2.0,
            123,
            feedback_valid=False,
        )
        reader = GimbalFeedbackReader(gimbal, timeout=0.01)

        self.assertIsNone(reader.read())
        self.assertTrue(reader.feedback_unavailable)


if __name__ == "__main__":
    unittest.main()

