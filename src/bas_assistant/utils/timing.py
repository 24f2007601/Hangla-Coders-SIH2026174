"""Performance timing helpers (measured values only, never invented)."""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager


class FPSMeter:
    """Exponential-moving-average frames-per-second meter."""

    def __init__(self, alpha: float = 0.9) -> None:
        self._alpha = alpha
        self._fps = 0.0
        self._last: float | None = None

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


class Metrics:
    """Small thread-safe timing/throughput collector for debug instrumentation.

    Records per-key durations (seconds) into a fixed-size window and exposes
    windowed mean / last value / inferred FPS plus plain counters. Overhead is a
    deque append and a lock — negligible, so it can stay enabled for diagnostics.
    """

    def __init__(self, window: int = 120) -> None:
        self._window = window
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = {}
        self._counters: dict[str, int] = {}

    def record(self, key: str, seconds: float) -> None:
        with self._lock:
            samples = self._samples.setdefault(key, deque(maxlen=self._window))
            samples.append(seconds * 1000.0)

    def counter(self, key: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + delta

    def snapshot(self) -> dict:
        with self._lock:
            out: dict[str, object] = {}
            for key, samples in self._samples.items():
                values = list(samples)
                mean_ms = statistics.fmean(values)
                fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
                out[key] = {
                    "count": len(values),
                    "mean_ms": round(mean_ms, 2),
                    "last_ms": round(values[-1], 2),
                    "fps": round(fps, 2),
                }
            out["counters"] = dict(self._counters)
            return out

    def format(self) -> str:
        snap = self.snapshot()
        lines: list[str] = []
        for key in sorted(snap):
            if key == "counters":
                continue
            value = snap[key]
            assert isinstance(value, dict)
            lines.append(
                f"{key}: {value['mean_ms']:.1f} ms avg, {value['last_ms']:.1f} ms last, "
                f"{value['fps']:.1f} fps, n={value['count']}"
            )
        counters = snap.get("counters")
        assert isinstance(counters, dict)
        if counters:
            lines.append("counters: " + ", ".join(f"{k}={v}" for k, v in counters.items()))
        return "\n".join(lines) or "(no samples)"


@contextmanager
def timed(key: str, metrics: Metrics) -> Iterator[None]:
    """Record the wall-clock duration of a block into `metrics` under `key`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.record(key, time.perf_counter() - start)


__all__ = ["FPSMeter", "LatencyMeter", "Metrics", "timed"]
