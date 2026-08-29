"""Unit tests for pose estimation helpers (hand extraction, drawing)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from bas_assistant.pose.estimation import MediaPipePoseEstimator


class _Category:
    def __init__(self, name: str, score: float) -> None:
        self.category_name = name
        self.score = score


class _Classifications:
    def __init__(self, categories: list) -> None:
        self.categories = categories


class _HandsOut:
    """Mimics the mediapipe Tasks API HandLandmarkerResult shape."""

    def __init__(self, handedness) -> None:
        self.hand_landmarks = [[type("L", (), {"x": 0.1, "y": 0.2})()] * 21]
        self.handedness = handedness


def test_extract_hands_tasks_api_list_of_categories() -> None:
    """Tasks API (mediapipe >= 1.0) hands: handedness is a list of Category."""
    out = _HandsOut(handedness=[[_Category("Left", 0.93)]])
    hands = MediaPipePoseEstimator._extract_hands(out, width=100, height=100)
    assert len(hands) == 1
    assert hands[0]["handedness"] == "Left"
    assert hands[0]["confidence"] == pytest.approx(0.93)
    assert len(hands[0]["keypoints"]) == 21
    assert hands[0]["keypoints"][0] == {"x": 10.0, "y": 20.0}


def test_extract_hands_legacy_classifications_shape() -> None:
    """Older mediapipe shape: handedness is a Classifications wrapper."""
    out = _HandsOut(handedness=[_Classifications([_Category("Right", 0.88)])])
    hands = MediaPipePoseEstimator._extract_hands(out, width=200, height=100)
    assert hands[0]["handedness"] == "Right"
    assert hands[0]["confidence"] == pytest.approx(0.88)


def test_extract_hands_empty_result() -> None:
    out = _HandsOut(handedness=[])
    assert MediaPipePoseEstimator._extract_hands(out, width=100, height=100) == []
    assert MediaPipePoseEstimator._extract_hands(None, width=100, height=100) == []


def test_extract_hands_missing_handedness_falls_back() -> None:
    out = _HandsOut(handedness=[[]])
    hands = MediaPipePoseEstimator._extract_hands(out, width=100, height=100)
    assert hands[0]["handedness"] == "Unknown"
    assert hands[0]["confidence"] == 1.0


def test_draw_hands_is_noop_without_metadata() -> None:
    from bas_assistant.pose.estimation import DummyPoseEstimator
    from bas_assistant.utils.visualization import draw_hands

    pose = DummyPoseEstimator().estimate(np.zeros((240, 320, 3), dtype=np.uint8))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_hands(frame, pose)  # should not raise
    assert frame.sum() == 0


def test_next_timestamp_ms_is_strictly_increasing(monkeypatch) -> None:
    est = object.__new__(MediaPipePoseEstimator)
    est._last_ts_ms = 0

    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    first = est._next_timestamp_ms(est._last_ts_ms)
    est._last_ts_ms = first
    assert first == 100_000

    monkeypatch.setattr(time, "monotonic", lambda: 100.5)
    second = est._next_timestamp_ms(est._last_ts_ms)
    est._last_ts_ms = second
    assert second == 100_500

    assert second > first


def test_next_timestamp_ms_never_repeats_on_same_clock_tick(monkeypatch) -> None:
    est = object.__new__(MediaPipePoseEstimator)
    est._last_ts_ms = 0

    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    first = est._next_timestamp_ms(est._last_ts_ms)
    second = est._next_timestamp_ms(first)  # same tick, must not collide

    assert second == first + 1


def _hold_estimator() -> MediaPipePoseEstimator:
    est = object.__new__(MediaPipePoseEstimator)
    est._last_detected = []
    est._last_detected_ts = 0.0
    est._hand_hold_seconds = 0.5
    return est


def test_exposed_hands_returns_detection_and_refreshes_last() -> None:
    est = _hold_estimator()
    out = _HandsOut(handedness=[[_Category("Left", 0.93)]])
    hands = est._exposed_hands(out, width=100, height=100)
    assert len(hands) == 1
    assert hands == est._last_detected


def test_exposed_hands_holds_last_detection_across_a_miss(monkeypatch) -> None:
    est = _hold_estimator()
    est._last_detected = [{"handedness": "Right", "confidence": 0.9, "keypoints": []}]
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    est._last_detected_ts = 99.9  # 0.1s ago, inside hold window
    hands = est._exposed_hands(None, width=100, height=100)
    assert hands == est._last_detected


def test_exposed_hands_clears_after_hold_expires(monkeypatch) -> None:
    est = _hold_estimator()
    est._last_detected = [{"handedness": "Right", "confidence": 0.9, "keypoints": []}]
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    est._last_detected_ts = 90.0  # 10s ago, outside hold window
    hands = est._exposed_hands(None, width=100, height=100)
    assert hands == []


def test_exposed_hands_empty_when_nothing_detected(monkeypatch) -> None:
    est = _hold_estimator()
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    est._last_detected_ts = 100.0  # recent but nothing held
    hands = est._exposed_hands(None, width=100, height=100)
    assert hands == []
