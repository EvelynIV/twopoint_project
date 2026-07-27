from __future__ import annotations

import time
from typing import Protocol

from twopoint_project.f32c import protocol


class SerialLike(Protocol):
    def write(self, data: bytes) -> int | None:
        ...

    def read(self, size: int = 1) -> bytes:
        ...


class F32CMotor:
    def __init__(
        self,
        serial_port: SerialLike,
        motor_id: int,
        *,
        command_interval: float = 0.001,
        debug_frames: bool = False,
    ) -> None:
        self.serial_port = serial_port
        self.motor_id = motor_id
        self.command_interval = command_interval
        self.debug_frames = debug_frames
        self.target_angle_deg = 0.0

    def enable(self) -> None:
        self._send(protocol.build_enable(self.motor_id))

    def disable(self) -> None:
        self._send(protocol.build_disable(self.motor_id))

    def set_multi_turn_passthrough(self) -> None:
        self._send(protocol.build_set_multi_turn_passthrough(self.motor_id))

    def set_speed_rpm(self, rpm: int) -> None:
        self._send(protocol.build_set_speed(self.motor_id, abs(int(rpm))))

    def move_to_angle(self, angle_deg: float) -> None:
        self.target_angle_deg = float(angle_deg)
        self._send(protocol.build_multi_turn_angle(self.motor_id, self.target_angle_deg))

    def move_by_angle(self, delta_deg: float) -> None:
        self.move_to_angle(self.target_angle_deg + float(delta_deg))

    def read_multi_turn_angle(self, timeout: float = 0.01) -> float:
        if timeout <= 0:
            raise ValueError("feedback timeout must be greater than 0")
        reset_input_buffer = getattr(self.serial_port, "reset_input_buffer", None)
        if callable(reset_input_buffer):
            reset_input_buffer()
        self._send(
            protocol.build_request_feedback(
                self.motor_id,
                protocol.FeedbackType.MULTI_TURN_ANGLE,
            )
        )
        deadline = time.monotonic() + timeout
        buffer = bytearray()
        while time.monotonic() < deadline:
            chunk = self.serial_port.read(protocol.FEEDBACK_FRAME_LENGTH)
            if chunk:
                buffer.extend(chunk)
                frame = self._extract_feedback_frame(buffer)
                if frame is not None:
                    return protocol.parse_feedback_frame(
                        frame,
                        expected_motor_id=self.motor_id,
                        expected_type=protocol.FeedbackType.MULTI_TURN_ANGLE,
                    ).angle_deg
            else:
                time.sleep(0.001)
        raise TimeoutError(f"timed out reading multi-turn angle from motor {self.motor_id}")

    @staticmethod
    def _extract_feedback_frame(buffer: bytearray) -> bytes | None:
        while buffer:
            try:
                head_index = buffer.index(protocol.FRAME_HEAD)
            except ValueError:
                buffer.clear()
                return None
            if head_index:
                del buffer[:head_index]
            if len(buffer) < protocol.FEEDBACK_FRAME_LENGTH:
                return None
            candidate = bytes(buffer[: protocol.FEEDBACK_FRAME_LENGTH])
            try:
                protocol.parse_feedback_frame(candidate)
            except ValueError:
                del buffer[0]
                continue
            del buffer[: protocol.FEEDBACK_FRAME_LENGTH]
            return candidate
        return None

    def _send(self, frame: bytes) -> None:
        if self.debug_frames:
            print(f"F32C[{self.motor_id}] -> {frame.hex(' ').upper()}")
        self.serial_port.write(frame)
        if self.command_interval > 0:
            time.sleep(self.command_interval)
