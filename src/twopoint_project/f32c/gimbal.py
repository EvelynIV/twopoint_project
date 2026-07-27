from __future__ import annotations

import errno
from dataclasses import dataclass
import time
from types import TracebackType
from typing import Type

from twopoint_project.f32c.motor import F32CMotor, SerialLike


DEFAULT_SERIAL_PORT = "/dev/ttyAS4"
DEFAULT_BAUDRATE = 115200
DEFAULT_X_ID = 1
DEFAULT_Y_ID = 2
DEFAULT_SPEED_RPM = 50
DEFAULT_STARTUP_DELAY = 0.3
DEFAULT_COMMAND_INTERVAL = 0.001
DEFAULT_ENABLE_SETTLE_DELAY = 0.2
A7A_UART_PORT_NAME = "UART4"
A7A_UART_TX_PIN = 16
A7A_UART_RX_PIN = 18


@dataclass(frozen=True)
class GimbalAngles:
    x_deg: float
    y_deg: float
    sampled_at_monotonic_ns: int
    feedback_valid: bool = True


class F32CGimbal:
    def __init__(
        self,
        serial_port: SerialLike,
        *,
        x_id: int = DEFAULT_X_ID,
        y_id: int = DEFAULT_Y_ID,
        speed_rpm: int = DEFAULT_SPEED_RPM,
        startup_delay: float = DEFAULT_STARTUP_DELAY,
        command_interval: float = DEFAULT_COMMAND_INTERVAL,
        enable_settle_delay: float = DEFAULT_ENABLE_SETTLE_DELAY,
        debug_frames: bool = False,
    ) -> None:
        self.serial_port = serial_port
        self.x = F32CMotor(serial_port, x_id, command_interval=command_interval, debug_frames=debug_frames)
        self.y = F32CMotor(serial_port, y_id, command_interval=command_interval, debug_frames=debug_frames)
        self.speed_rpm = speed_rpm
        self.startup_delay = startup_delay
        self.enable_settle_delay = enable_settle_delay

    def __enter__(self) -> F32CGimbal:
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def initialize(self) -> None:
        if self.startup_delay > 0:
            time.sleep(self.startup_delay)
        self.enable()
        if self.enable_settle_delay > 0:
            time.sleep(self.enable_settle_delay)
        self.set_multi_turn_passthrough()
        self.set_speed_rpm(self.speed_rpm)
        self.x.target_angle_deg = 0.0
        self.y.target_angle_deg = 0.0

    def enable(self) -> None:
        self.x.enable()
        self.y.enable()

    def set_multi_turn_passthrough(self) -> None:
        self.x.set_multi_turn_passthrough()
        self.y.set_multi_turn_passthrough()

    def set_speed_rpm(self, rpm: int) -> None:
        self.x.set_speed_rpm(rpm)
        self.y.set_speed_rpm(rpm)

    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        self.x.move_by_angle(x_delta_deg)
        self.y.move_by_angle(y_delta_deg)

    def move_to(self, x_angle_deg: float, y_angle_deg: float) -> None:
        self.x.move_to_angle(x_angle_deg)
        self.y.move_to_angle(y_angle_deg)

    def read_angles(self, timeout: float = 0.01) -> GimbalAngles:
        started = time.monotonic_ns()
        x_deg = self.x.read_multi_turn_angle(timeout=timeout)
        y_deg = self.y.read_multi_turn_angle(timeout=timeout)
        finished = time.monotonic_ns()
        return GimbalAngles(
            x_deg=x_deg,
            y_deg=y_deg,
            sampled_at_monotonic_ns=(started + finished) // 2,
        )

    def commanded_angles(self) -> GimbalAngles:
        return GimbalAngles(
            x_deg=self.x.target_angle_deg,
            y_deg=self.y.target_angle_deg,
            sampled_at_monotonic_ns=time.monotonic_ns(),
            feedback_valid=False,
        )

    def sync_commanded_angles(self, angles: GimbalAngles) -> None:
        self.x.target_angle_deg = float(angles.x_deg)
        self.y.target_angle_deg = float(angles.y_deg)

    def disable(self) -> None:
        self.x.disable()
        self.y.disable()

    def close(self) -> None:
        close = getattr(self.serial_port, "close", None)
        if close is not None:
            close()


def open_serial_gimbal(
    *,
    port: str = DEFAULT_SERIAL_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    x_id: int = DEFAULT_X_ID,
    y_id: int = DEFAULT_Y_ID,
    speed_rpm: int = DEFAULT_SPEED_RPM,
    startup_delay: float = DEFAULT_STARTUP_DELAY,
    command_interval: float = DEFAULT_COMMAND_INTERVAL,
    enable_settle_delay: float = DEFAULT_ENABLE_SETTLE_DELAY,
    debug_frames: bool = False,
) -> F32CGimbal:
    try:
        import serial
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyserial is required for F32C serial control. Run: poetry install") from exc

    try:
        serial_port = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.001,
            write_timeout=1.0,
        )
    except serial.SerialException as exc:
        if getattr(exc, "errno", None) == errno.EACCES:
            raise RuntimeError(
                f"Permission denied opening {port}. "
                f"Run: sudo chgrp dialout {port} && sudo chmod 660 {port}"
            ) from exc
        raise
    return F32CGimbal(
        serial_port,
        x_id=x_id,
        y_id=y_id,
        speed_rpm=speed_rpm,
        startup_delay=startup_delay,
        command_interval=command_interval,
        enable_settle_delay=enable_settle_delay,
        debug_frames=debug_frames,
    )
