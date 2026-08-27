"""Unit tests for pose normalization (translation + scale)."""

from __future__ import annotations

import numpy as np
import pytest

from bas_assistant.pose.landmarks import LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_HIP, RIGHT_SHOULDER
from bas_assistant.pose.normalization import NormalizedPose, normalize_pose


def _standing_pose(make_pose):
    """33-point pose with hips at y=200 (centre x=100), shoulders at y=150."""
    pts = [(0.0, 0.0)] * 33
    pts[LEFT_HIP] = (90.0, 200.0)
    pts[RIGHT_HIP] = (110.0, 200.0)
    pts[LEFT_SHOULDER] = (80.0, 150.0)
    pts[RIGHT_SHOULDER] = (120.0, 150.0)
    pts[NOSE] = (100.0, 120.0)
    return make_pose(pts)


def test_normalize_translates_hip_center_to_origin(make_pose) -> None:
    norm = normalize_pose(_standing_pose(make_pose))
    assert norm is not None
    hip_center = (norm.keypoints[LEFT_HIP] + norm.keypoints[RIGHT_HIP]) / 2
    assert hip_center[0] == pytest.approx(0.0, abs=1e-9)
    assert hip_center[1] == pytest.approx(0.0, abs=1e-9)


def test_normalize_scales_by_torso_length(make_pose) -> None:
    norm = normalize_pose(_standing_pose(make_pose))
    assert norm is not None
    # Torso length is 50 px; shoulder-centre is 50 px above hip-centre.
    assert norm.scale == pytest.approx(50.0)
    shoulder_center = (norm.keypoints[LEFT_SHOULDER] + norm.keypoints[RIGHT_SHOULDER]) / 2
    assert shoulder_center[0] == pytest.approx(0.0, abs=1e-9)
    assert shoulder_center[1] == pytest.approx(-1.0, abs=1e-9)


def test_normalize_is_scale_and_translation_invariant(make_pose) -> None:
    base = _standing_pose(make_pose)
    moved = make_pose([(p.x + 300.0, p.y - 40.0) for p in base.keypoints])
    norm_base = normalize_pose(base)
    norm_moved = normalize_pose(moved)
    assert norm_base is not None and norm_moved is not None
    np.testing.assert_allclose(norm_base.keypoints, norm_moved.keypoints, atol=1e-9)


def test_normalize_returns_none_when_too_few_visible(make_pose) -> None:
    pose = make_pose([(0.0, 0.0)] * 5)
    assert normalize_pose(pose) is None


def test_normalize_returns_none_on_zero_scale(make_pose) -> None:
    pts = [(10.0, 10.0)] * 33  # all points identical -> zero torso length
    assert normalize_pose(make_pose(pts)) is None


def test_normalize_falls_back_to_nose_when_hips_hidden(make_pose) -> None:
    pts = [(0.0, 0.0)] * 33
    pts[LEFT_HIP] = (0.0, 0.0)
    pts[RIGHT_HIP] = (0.0, 0.0)
    pts[LEFT_SHOULDER] = (0.0, 0.0)
    pts[RIGHT_SHOULDER] = (0.0, 0.0)
    pts[NOSE] = (100.0, 200.0)
    pose = make_pose(pts)
    pose.keypoints[LEFT_HIP].confidence = 0.0
    pose.keypoints[RIGHT_HIP].confidence = 0.0
    pose.keypoints[LEFT_SHOULDER].confidence = 0.0
    pose.keypoints[RIGHT_SHOULDER].confidence = 0.0
    norm = normalize_pose(pose)
    assert norm is not None
    np.testing.assert_allclose(norm.keypoints[NOSE], [0.0, 0.0], atol=1e-9)


def test_normalized_pose_as_array_length() -> None:
    np_normalized = NormalizedPose(
        keypoints=np.zeros((33, 2)),
        confidence=np.ones(33),
        valid=True,
        origin=(0.0, 0.0),
        scale=1.0,
    )
    assert np_normalized.as_array().shape == (66,)
