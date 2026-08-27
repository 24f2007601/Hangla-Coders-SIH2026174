"""Performance timing helpers (measured values only, never invented)."""

from __future__ import annotations

import time


class FPSMeter:
    """Exponential-moving-average frames-per-second meter."""

    def __init__(self, alpha: float = 0.9) -> None:
        self._alpha = alpha
        self._fps = 0.0
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self._fps = (
                    self._alpha * self._fps + (1 - self._alpha) * inst if self._fps else inst
                )
        self._last = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self) -> None:
        self._fps = 0.0
        self._last = None


class LatencyMeter:
    """Exponential-moving-average latency (milliseconds) meter."""

    def __init__(self, alpha: float = 0.9) -> None:
        self._alpha = alpha
        self._ms = 0.0

    def update(self, seconds: float) -> float:
        ms = seconds * 1000.0
        self._ms = self._alpha * self._ms + (1 - self._alpha) * ms if self._ms else ms
        return self._ms

    @property
    def milliseconds(self) -> float:
        return self._ms

    def reset(self) -> None:
        self._ms = 0.0


__all__ = ["FPSMeter", "LatencyMeter"]
