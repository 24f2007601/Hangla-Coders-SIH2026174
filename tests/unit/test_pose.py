"""Unit tests for pose estimation helpers (hand extraction, drawing)."""

from __future__ import annotations

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
