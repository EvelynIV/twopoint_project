"""Temporal filtering for the normalized target-center observation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence

from twopoint_project.vision.inferencer import PointPrediction


TARGET_CENTER_LABEL = "target_center"


@dataclass(frozen=True)
class TargetCenterFilterConfig:
    """Configuration for target-center jump rejection followed by an EMA."""

    enabled: bool = False
    ema_alpha: float = 0.35
    max_jump: float = 0.08
    jump_confirm_frames: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("center.target_filter.ema_alpha must be in (0, 1]")
        if not isfinite(self.max_jump) or self.max_jump <= 0.0:
            raise ValueError("center.target_filter.max_jump must be greater than 0")
        if self.jump_confirm_frames <= 0:
            raise ValueError(
                "center.target_filter.jump_confirm_frames must be greater than 0"
            )


class TargetCenterFilter:
    """Reject isolated jumps, then smooth accepted observations with an EMA.

    A jump is measured from the last accepted raw observation. A stable cluster
    at a new location is accepted after ``jump_confirm_frames`` observations so
    that a real target relocation cannot remain rejected indefinitely.
    """

    def __init__(self, config: TargetCenterFilterConfig) -> None:
        self.config = config
        self._accepted_raw: tuple[float, float] | None = None
        self._filtered: tuple[float, float] | None = None
        self._pending_jump: tuple[float, float] | None = None
        self._pending_count = 0

    def reset(self) -> None:
        self._accepted_raw = None
        self._filtered = None
        self._pending_jump = None
        self._pending_count = 0

    def filter_points(
        self,
        points: Sequence[PointPrediction],
    ) -> list[PointPrediction]:
        copied = [dict(point) for point in points]
        if not self.config.enabled:
            return copied

        candidates = [
            (index, point)
            for index, point in enumerate(copied)
            if point["label"] == TARGET_CENTER_LABEL
        ]
        if not candidates:
            self.reset()
            return copied

        selected_index, selected = max(
            candidates,
            key=lambda item: float(item[1]["confidence"]),
        )
        x = float(selected["x"])
        y = float(selected["y"])
        confidence = float(selected["confidence"])
        if (
            not isfinite(confidence)
            or not isfinite(x)
            or not isfinite(y)
        ):
            self.reset()
            return copied

        accepted = self._accept_or_confirm_jump(x, y)
        without_centers = [
            point for point in copied if point["label"] != TARGET_CENTER_LABEL
        ]
        if not accepted:
            return without_centers

        if self._filtered is None:
            filtered_x, filtered_y = x, y
        else:
            alpha = self.config.ema_alpha
            filtered_x = alpha * x + (1.0 - alpha) * self._filtered[0]
            filtered_y = alpha * y + (1.0 - alpha) * self._filtered[1]
        self._filtered = (filtered_x, filtered_y)
        filtered = dict(selected)
        filtered["x"] = filtered_x
        filtered["y"] = filtered_y

        insertion_index = min(selected_index, len(without_centers))
        without_centers.insert(insertion_index, filtered)
        return without_centers

    def _accept_or_confirm_jump(self, x: float, y: float) -> bool:
        if self._accepted_raw is None:
            self._accept_raw(x, y)
            return True

        jump = hypot(x - self._accepted_raw[0], y - self._accepted_raw[1])
        if jump <= self.config.max_jump:
            self._accept_raw(x, y)
            return True

        if self.config.jump_confirm_frames == 1:
            self._accept_raw(x, y)
            return True

        pending = self._pending_jump
        if pending is not None and hypot(x - pending[0], y - pending[1]) <= self.config.max_jump:
            self._pending_count += 1
            self._pending_jump = (x, y)
        else:
            self._pending_jump = (x, y)
            self._pending_count = 1

        if self._pending_count < self.config.jump_confirm_frames:
            return False

        self._accept_raw(x, y)
        return True

    def _accept_raw(self, x: float, y: float) -> None:
        self._accepted_raw = (x, y)
        self._pending_jump = None
        self._pending_count = 0
