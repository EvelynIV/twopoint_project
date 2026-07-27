from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable


@dataclass(frozen=True)
class LoopTiming:
    elapsed_seconds: float
    overrun_seconds: float
    overrun_count: int
    skipped_deadlines: int

    @property
    def overran(self) -> bool:
        return self.overrun_seconds > 0.0


class FixedDeadlineScheduler:
    """Keep a fixed monotonic deadline grid without burst catch-up."""

    def __init__(
        self,
        loop_hz: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.period_seconds = 0.0 if loop_hz <= 0 else 1.0 / loop_hz
        self._clock = clock
        self._sleeper = sleeper
        self._next_deadline = (
            None if self.period_seconds <= 0 else self._clock() + self.period_seconds
        )
        self.overrun_count = 0

    def wait_for_next(self, iteration_started_at: float) -> LoopTiming:
        now = self._clock()
        elapsed = max(now - iteration_started_at, 0.0)
        deadline = self._next_deadline
        if deadline is None:
            return LoopTiming(elapsed, 0.0, self.overrun_count, 0)

        if now <= deadline:
            remaining = deadline - now
            if remaining > 0:
                self._sleeper(remaining)
            self._next_deadline = deadline + self.period_seconds
            return LoopTiming(elapsed, 0.0, self.overrun_count, 0)

        overrun = now - deadline
        skipped = int(overrun // self.period_seconds) + 1
        self._next_deadline = deadline + skipped * self.period_seconds
        self.overrun_count += 1
        return LoopTiming(elapsed, overrun, self.overrun_count, skipped)


def wait_for_motor_deadline(
    scheduler: FixedDeadlineScheduler,
    iteration_started_at: float,
    *,
    phase: str,
) -> None:
    timing = scheduler.wait_for_next(iteration_started_at)
    if timing.overran:
        print(
            f"{phase}: motor loop overrun #{timing.overrun_count} "
            f"elapsed={timing.elapsed_seconds * 1000.0:.1f}ms "
            f"late={timing.overrun_seconds * 1000.0:.1f}ms "
            f"skipped_deadlines={timing.skipped_deadlines}"
        )

