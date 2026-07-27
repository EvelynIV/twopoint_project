from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import IntEnum


FRAME_HEAD = 0x7A
FRAME_TAIL = 0x7B
FEEDBACK_FRAME_LENGTH = 9


class Command(IntEnum):
    SET_MODE = 0x00
    SET_SPEED = 0x01
    SET_MULTI_TURN_ANGLE = 0x02
    SET_SINGLE_TURN_ANGLE = 0x03
    DISABLE = 0x05
    ENABLE = 0x06
    SET_ACCELERATION = 0x07
    SAVE_PARAMETERS = 0x08
    ZERO_SINGLE_TURN_ANGLE = 0x0A
    RESTORE_FACTORY = 0x0B
    SET_ADDRESS = 0x0D
    REQUEST_FEEDBACK = 0x0E


class ControlMode(IntEnum):
    SPEED = 0x0000
    MULTI_TURN_PLANNED = 0x0001
    SINGLE_TURN_PLANNED = 0x0002
    MULTI_TURN_PASSTHROUGH = 0x0003
    SINGLE_TURN_PASSTHROUGH = 0x0004


class FeedbackType(IntEnum):
    SPEED = 0x00
    MULTI_TURN_ANGLE = 0x01
    MECHANICAL_ANGLE = 0x02
    ACCELERATION = 0x03
    BUS_VOLTAGE = 0x04


@dataclass(frozen=True)
class FeedbackFrame:
    motor_id: int
    feedback_type: FeedbackType
    raw_value: int

    @property
    def angle_deg(self) -> float:
        if self.feedback_type not in {
            FeedbackType.MULTI_TURN_ANGLE,
            FeedbackType.MECHANICAL_ANGLE,
        }:
            raise ValueError(f"feedback type {self.feedback_type.name} is not an angle")
        return self.raw_value / 10.0


def checksum(payload: bytes) -> int:
    value = 0
    for byte in payload:
        value ^= byte
    return value


def build_frame(motor_id: int, command: int | Command, data: bytes = b"") -> bytes:
    _validate_byte("motor_id", motor_id)
    _validate_byte("command", int(command))
    body = bytes((FRAME_HEAD, motor_id, int(command))) + data
    return body + bytes((checksum(body), FRAME_TAIL))


def build_enable(motor_id: int) -> bytes:
    return build_frame(motor_id, Command.ENABLE)


def build_disable(motor_id: int) -> bytes:
    return build_frame(motor_id, Command.DISABLE)


def build_set_mode(motor_id: int, mode: int | ControlMode) -> bytes:
    return build_frame(motor_id, Command.SET_MODE, int(mode).to_bytes(2, "big", signed=False))


def build_set_multi_turn_passthrough(motor_id: int) -> bytes:
    return build_set_mode(motor_id, ControlMode.MULTI_TURN_PASSTHROUGH)


def build_set_speed(motor_id: int, rpm: int) -> bytes:
    return build_frame(motor_id, Command.SET_SPEED, int(rpm).to_bytes(2, "big", signed=True))


def build_multi_turn_angle(motor_id: int, angle_deg: float) -> bytes:
    scaled_angle = angle_to_tenths(angle_deg)
    return build_frame(
        motor_id,
        Command.SET_MULTI_TURN_ANGLE,
        scaled_angle.to_bytes(4, "big", signed=True),
    )


def build_request_feedback(motor_id: int, feedback_type: int | FeedbackType) -> bytes:
    _validate_byte("feedback_type", int(feedback_type))
    return build_frame(motor_id, Command.REQUEST_FEEDBACK, bytes((int(feedback_type),)))


def parse_feedback_frame(
    frame: bytes,
    *,
    expected_motor_id: int | None = None,
    expected_type: FeedbackType | None = None,
) -> FeedbackFrame:
    if len(frame) != FEEDBACK_FRAME_LENGTH:
        raise ValueError(
            f"feedback frame must be {FEEDBACK_FRAME_LENGTH} bytes, got {len(frame)}"
        )
    if frame[0] != FRAME_HEAD or frame[-1] != FRAME_TAIL:
        raise ValueError("invalid feedback frame markers")
    if checksum(frame[:-2]) != frame[-2]:
        raise ValueError("invalid feedback frame checksum")
    motor_id = int(frame[1])
    if expected_motor_id is not None and motor_id != expected_motor_id:
        raise ValueError(
            f"feedback motor id mismatch: expected {expected_motor_id}, got {motor_id}"
        )
    try:
        feedback_type = FeedbackType(frame[2])
    except ValueError as exc:
        raise ValueError(f"unknown feedback type: 0x{frame[2]:02X}") from exc
    if expected_type is not None and feedback_type != expected_type:
        raise ValueError(
            f"feedback type mismatch: expected {expected_type.name}, got {feedback_type.name}"
        )
    return FeedbackFrame(
        motor_id=motor_id,
        feedback_type=feedback_type,
        raw_value=int.from_bytes(frame[3:7], "big", signed=True),
    )


def angle_to_tenths(angle_deg: float) -> int:
    return int((Decimal(str(angle_deg)) * Decimal("10")).to_integral_value(rounding=ROUND_HALF_UP))


def _validate_byte(name: str, value: int) -> None:
    if not 0 <= value <= 0xFF:
        raise ValueError(f"{name} must be in range 0..255")
