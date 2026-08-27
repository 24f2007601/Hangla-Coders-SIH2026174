"""Shared test fixtures and import bootstrap (works without an editable install)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bas_assistant.models import Keypoint, PoseResult  # noqa: E402
from bas_assistant.pose.estimation import DummyPoseEstimator  # noqa: E402


@pytest.fixture
def make_pose():
    """Build a PoseResult from a list of (x, y) tuples (all keypoints visible)."""

    def _make(
        xy: list[tuple[float, float]], person_id: int = 1, confidence: float = 1.0
    ) -> PoseResult:
        keypoints = [Keypoint(x=float(x), y=float(y), confidence=confidence) for x, y in xy]
        return PoseResult(
            timestamp=0.0,
            person_id=person_id,
            keypoints=keypoints,
            confidence=confidence,
        )

    return _make


@pytest.fixture
def dummy_pose():
    """A deterministic synthetic standing pose (via DummyPoseEstimator)."""
    estimator = DummyPoseEstimator(motion=0.0)

    def _make(width: int = 320, height: int = 240) -> PoseResult:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        pose = estimator.estimate(frame)
        assert pose is not None
        return pose

    return _make
