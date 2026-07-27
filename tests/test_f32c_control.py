from __future__ import annotations

import unittest
from argparse import Namespace
from unittest.mock import patch

from twopoint_project.f32c import cli
from twopoint_project.f32c import protocol
from twopoint_project.f32c.gimbal import F32CGimbal
from twopoint_project.f32c.motor import F32CMotor


class FakeSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class FakeFeedbackSerial(FakeSerial):
    def __init__(self, response: bytes) -> None:
        super().__init__()
        self.response = bytearray(response)

    def reset_input_buffer(self) -> None:
        pass

    def read(self, size: int = 1) -> bytes:
        chunk = bytes(self.response[:size])
        del self.response[:size]
        return chunk


class F32CMotorTest(unittest.TestCase):
    def test_move_by_accumulates_target_angle(self) -> None:
        serial_port = FakeSerial()
        motor = F32CMotor(serial_port, 1, command_interval=0)

        motor.move_by_angle(10)
        motor.move_by_angle(5)

        self.assertEqual(motor.target_angle_deg, 15)
        self.assertEqual(serial_port.writes[0], protocol.build_multi_turn_angle(1, 10))
        self.assertEqual(serial_port.writes[1], protocol.build_multi_turn_angle(1, 15))

    def test_read_multi_turn_angle_returns_encoder_feedback(self) -> None:
        response = protocol.build_frame(
            1,
            protocol.FeedbackType.MULTI_TURN_ANGLE,
            (123).to_bytes(4, "big", signed=True),
        )
        serial_port = FakeFeedbackSerial(response)
        motor = F32CMotor(serial_port, 1, command_interval=0)

        angle = motor.read_multi_turn_angle(timeout=0.01)

        self.assertEqual(angle, 12.3)
        self.assertEqual(
            serial_port.writes,
            [protocol.build_request_feedback(1, protocol.FeedbackType.MULTI_TURN_ANGLE)],
        )

class F32CGimbalTest(unittest.TestCase):
    def test_initialize_order(self) -> None:
        serial_port = FakeSerial()
        gimbal = F32CGimbal(
            serial_port,
            speed_rpm=100,
            startup_delay=0,
            command_interval=0,
            enable_settle_delay=0,
        )

        gimbal.initialize()

        self.assertEqual(
            serial_port.writes,
            [
                protocol.build_enable(1),
                protocol.build_enable(2),
                protocol.build_set_multi_turn_passthrough(1),
                protocol.build_set_multi_turn_passthrough(2),
                protocol.build_set_speed(1, 100),
                protocol.build_set_speed(2, 100),
            ],
        )

    def test_move_by_sends_both_axes(self) -> None:
        serial_port = FakeSerial()
        gimbal = F32CGimbal(serial_port, startup_delay=0, command_interval=0)

        gimbal.move_by(1.0, -0.5)

        self.assertEqual(serial_port.writes[0], protocol.build_multi_turn_angle(1, 1.0))
        self.assertEqual(serial_port.writes[1], protocol.build_multi_turn_angle(2, -0.5))

    def test_disable_sends_both_axes_without_initializing(self) -> None:
        serial_port = FakeSerial()
        gimbal = F32CGimbal(serial_port, startup_delay=0, command_interval=0)

        gimbal.disable()

        self.assertEqual(
            serial_port.writes,
            [
                protocol.build_disable(1),
                protocol.build_disable(2),
            ],
        )


class F32CCliTest(unittest.TestCase):
    def test_module_no_longer_exposes_standalone_parser(self) -> None:
        self.assertFalse(hasattr(cli, "parse_args"))
        self.assertFalse(hasattr(cli, "main"))

    def test_move_by_opens_gimbal_and_sends_offsets(self) -> None:
        serial_port = FakeSerial()
        gimbal = F32CGimbal(serial_port, startup_delay=0, command_interval=0)
        args = Namespace(command="move-by", x=1.0, y=-0.5)

        with patch("twopoint_project.f32c.cli.build_gimbal", return_value=gimbal) as build_gimbal:
            cli.run(args)

        build_gimbal.assert_called_once_with(args)
        self.assertEqual(serial_port.writes[-2], protocol.build_multi_turn_angle(1, 1.0))
        self.assertEqual(serial_port.writes[-1], protocol.build_multi_turn_angle(2, -0.5))


if __name__ == "__main__":
    unittest.main()
