"""Unit tests for the reusable timing/metrics utilities (no camera/hardware)."""

from __future__ import annotations

import time

import pytest

from bas_assistant.utils.timing import FPSMeter, LatencyMeter, Metrics, timed


def test_fps_meter_returns_nonzero_after_two_ticks() -> None:
    meter = FPSMeter()
    meter.tick()
    assert meter.fps == 0.0
    meter.tick()
    assert meter.fps > 0.0


def test_fps_meter_reset() -> None:
    meter = FPSMeter()
    meter.tick()
    meter.tick()
    meter.reset()
    assert meter.fps == 0.0


def test_latency_meter_updates_and_resets() -> None:
    meter = LatencyMeter()
    assert meter.milliseconds == 0.0
    meter.update(0.010)
    assert meter.milliseconds == pytest.approx(10.0)
    meter.reset()
    assert meter.milliseconds == 0.0


def test_metrics_record_snapshot_mean_and_last() -> None:
    metrics = Metrics()
    metrics.record("pose", 0.010)
    metrics.record("pose", 0.030)
    snap = metrics.snapshot()
    pose = snap["pose"]
    assert pose["count"] == 2
    assert pose["mean_ms"] == pytest.approx(20.0)
    assert pose["last_ms"] == pytest.approx(30.0)


def test_metrics_counter() -> None:
    metrics = Metrics()
    metrics.counter("frames_processed", 2)
    metrics.counter("frames_processed")
    assert metrics.snapshot()["counters"]["frames_processed"] == 3


def test_metrics_format_contains_stage_and_counters() -> None:
    metrics = Metrics()
    metrics.record("vision_total", 0.012)
    metrics.counter("frames_processed")
    text = metrics.format()
    assert "vision_total" in text
    assert "frames_processed=1" in text


def test_timed_context_manager_records_duration() -> None:
    metrics = Metrics()
    with timed("capture", metrics):
        time.sleep(0.001)
    snap = metrics.snapshot()
    assert snap["capture"]["count"] == 1
    assert snap["capture"]["mean_ms"] > 0.0
