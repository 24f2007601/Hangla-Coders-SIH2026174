"""Unit tests for feature extraction (spatial + temporal window features)."""

from __future__ import annotations

import numpy as np
import pytest

from bas_assistant.features.extractor import (
    FEATURE_VECTOR_SIZE,
    SPATIAL_FEATURES,
    TEMPORAL_FEATURES,
    PoseFeatureExtractor,
)
from bas_assistant.features.window import FeatureWindow


def test_feature_vector_size_matches_schema() -> None:
    assert 2 * len(SPATIAL_FEATURES) + len(TEMPORAL_FEATURES) == FEATURE_VECTOR_SIZE


def test_feature_window_full_transition() -> None:
    window = FeatureWindow[int](3)
    assert window.push(1) is False
    assert window.push(2) is False
    assert window.push(3) is True
    assert window.is_full
    # Window keeps sliding, stays full; push does not re-report full.
    assert window.push(4) is False
    assert window.items() == [2, 3, 4]
    window.clear()
    assert not window.is_full


def test_feature_window_requires_positive_size() -> None:
    with pytest.raises(ValueError):
        FeatureWindow[int](0)


def test_extractor_not_ready_until_window_full(dummy_pose) -> None:
    extractor = PoseFeatureExtractor(sequence_length=5)
    assert not extractor.is_ready
    pose = dummy_pose()
    for _ in range(4):
        assert extractor.push(pose) is False
        assert not extractor.is_ready
    assert extractor.push(pose) is True
    assert extractor.is_ready


def test_features_shape_and_determinism(dummy_pose) -> None:
    extractor = PoseFeatureExtractor(sequence_length=8)
    pose = dummy_pose()
    for _ in range(8):
        extractor.push(pose)
    first = extractor.features()
    assert first.shape == (FEATURE_VECTOR_SIZE,)

    extractor2 = PoseFeatureExtractor(sequence_length=8)
    for _ in range(8):
        extractor2.push(pose)
    np.testing.assert_allclose(first, extractor2.features())


def test_features_raise_before_window_full(dummy_pose) -> None:
    extractor = PoseFeatureExtractor(sequence_length=8)
    pose = dummy_pose()
    for _ in range(4):
        extractor.push(pose)
    with pytest.raises(ValueError):
        extractor.features()


def test_extractor_reset(dummy_pose) -> None:
    extractor = PoseFeatureExtractor(sequence_length=4)
    pose = dummy_pose()
    for _ in range(4):
        extractor.push(pose)
    assert extractor.is_ready
    extractor.reset()
    assert not extractor.is_ready


def test_extractor_ignores_invalid_pose(make_pose) -> None:
    extractor = PoseFeatureExtractor(sequence_length=4)
    invalid = make_pose([(0.0, 0.0)] * 3)  # too few keypoints to normalize
    assert extractor.push(invalid) is False
    assert not extractor.is_ready


def test_temporal_velocity_features_respond_to_motion(dummy_pose) -> None:
    from bas_assistant.pose.estimation import DummyPoseEstimator

    still = PoseFeatureExtractor(sequence_length=10)
    still_pose = DummyPoseEstimator(motion=0.0).estimate(np.zeros((240, 320, 3), dtype=np.uint8))
    assert still_pose is not None
    for _ in range(10):
        still.push(still_pose)

    moving = PoseFeatureExtractor(sequence_length=10)
    moving_estimator = DummyPoseEstimator(motion=1.0)
    for _ in range(10):
        moving.push(moving_estimator.estimate(np.zeros((240, 320, 3), dtype=np.uint8)))

    # Temporal features sit at the tail of the vector.
    still_temporal = still.features()[-len(TEMPORAL_FEATURES) :]
    moving_temporal = moving.features()[-len(TEMPORAL_FEATURES) :]
    assert np.linalg.norm(moving_temporal) > np.linalg.norm(still_temporal)
