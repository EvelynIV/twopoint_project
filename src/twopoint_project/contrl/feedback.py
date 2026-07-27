from __future__ import annotations

from typing import Any


class GimbalFeedbackReader:
    """Read real encoder angles and report outages without command fallback."""

    def __init__(self, gimbal: Any, timeout: float) -> None:
        self.gimbal = gimbal
        self.timeout = timeout
        self.feedback_unavailable = False
        self.recovered_this_read = False

    def initialize(self) -> Any | None:
        angles = self.read()
        if angles is not None:
            self.gimbal.sync_commanded_angles(angles)
        return angles

    def read(self) -> Any | None:
        self.recovered_this_read = False
        try:
            angles = self.gimbal.read_angles(timeout=self.timeout)
        except (OSError, TimeoutError, ValueError) as exc:
            if not self.feedback_unavailable:
                print(f"gimbal feedback unavailable; pausing angle control: {exc}")
            self.feedback_unavailable = True
            return None
        if not angles.feedback_valid:
            if not self.feedback_unavailable:
                print("gimbal feedback unavailable; refusing non-encoder angle data")
            self.feedback_unavailable = True
            return None
        if self.feedback_unavailable:
            print("gimbal feedback recovered; waiting for a fresh visual target")
            self.recovered_this_read = True
        self.feedback_unavailable = False
        return angles

