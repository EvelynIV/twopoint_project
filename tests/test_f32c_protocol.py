from __future__ import annotations

import unittest

from twopoint_project.f32c import protocol


class F32CProtocolTest(unittest.TestCase):
    def test_enable_frames_match_manual(self) -> None:
        self.assertEqual(protocol.build_enable(1), bytes.fromhex("7A 01 06 7D 7B"))
        self.assertEqual(protocol.build_enable(2), bytes.fromhex("7A 02 06 7E 7B"))

    def test_disable_frames_match_manual(self) -> None:
        self.assertEqual(protocol.build_disable(1), bytes.fromhex("7A 01 05 7E 7B"))
        self.assertEqual(protocol.build_disable(2), bytes.fromhex("7A 02 05 7D 7B"))

    def test_multi_turn_passthrough_frames_match_manual(self) -> None:
        self.assertEqual(protocol.build_set_multi_turn_passthrough(1), bytes.fromhex("7A 01 00 00 03 78 7B"))
        self.assertEqual(protocol.build_set_multi_turn_passthrough(2), bytes.fromhex("7A 02 00 00 03 7B 7B"))

    def test_speed_frames_match_manual(self) -> None:
        self.assertEqual(protocol.build_set_speed(1, 100), bytes.fromhex("7A 01 01 00 64 1E 7B"))
        self.assertEqual(protocol.build_set_speed(2, 100), bytes.fromhex("7A 02 01 00 64 1D 7B"))

    def test_multi_turn_angle_frames_match_manual(self) -> None:
        self.assertEqual(protocol.build_multi_turn_angle(1, 360), bytes.fromhex("7A 01 02 00 00 0E 10 67 7B"))
        self.assertEqual(protocol.build_multi_turn_angle(2, 360), bytes.fromhex("7A 02 02 00 00 0E 10 64 7B"))

    def test_negative_angle_uses_int32_twos_complement(self) -> None:
        self.assertEqual(protocol.build_multi_turn_angle(1, -360), bytes.fromhex("7A 01 02 FF FF F1 F0 78 7B"))
        self.assertEqual(protocol.build_multi_turn_angle(2, -360), bytes.fromhex("7A 02 02 FF FF F1 F0 7B 7B"))

    def test_checksum_can_equal_frame_tail_without_escaping(self) -> None:
        self.assertEqual(protocol.build_set_multi_turn_passthrough(2)[-2], protocol.FRAME_TAIL)
        self.assertEqual(protocol.build_set_multi_turn_passthrough(2), bytes.fromhex("7A 02 00 00 03 7B 7B"))

    def test_multi_turn_feedback_request_and_response(self) -> None:
        self.assertEqual(
            protocol.build_request_feedback(1, protocol.FeedbackType.MULTI_TURN_ANGLE),
            bytes.fromhex("7A 01 0E 01 74 7B"),
        )
        response = protocol.build_frame(
            1,
            protocol.FeedbackType.MULTI_TURN_ANGLE,
            (-123).to_bytes(4, "big", signed=True),
        )

        parsed = protocol.parse_feedback_frame(
            response,
            expected_motor_id=1,
            expected_type=protocol.FeedbackType.MULTI_TURN_ANGLE,
        )

        self.assertEqual(parsed.raw_value, -123)
        self.assertEqual(parsed.angle_deg, -12.3)

    def test_feedback_parser_rejects_bad_checksum(self) -> None:
        response = bytearray(
            protocol.build_frame(
                2,
                protocol.FeedbackType.MULTI_TURN_ANGLE,
                (100).to_bytes(4, "big", signed=True),
            )
        )
        response[-2] ^= 0x01

        with self.assertRaisesRegex(ValueError, "checksum"):
            protocol.parse_feedback_frame(bytes(response))


if __name__ == "__main__":
    unittest.main()
