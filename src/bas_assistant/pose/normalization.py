"""Pose normalization: translation + scale only (PoC scope).

This module implements basic normalization (translation to a body origin and scaling
by torso length). Advanced techniques — body-centric coordinates, rotation/gamma
normalization, camera-coordinate normalization, 2D/3D — are intentionally NOT claimed
as implemented; see `docs/architecture.md` for the future orientation-agnostic path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bas_assistant.models import PoseResult
from bas_assistant.pose.landmarks import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

MIN_VISIBLE_KEYPOINTS = 8


@dataclass(slots=True)
class NormalizedPose:
    """Translation- and scale-normalized keypoints.

    Attributes:
        keypoints: (N, 2) array of normalized coordinates, origin- and scale-free.
        confidence: (N,) per-keypoint confidence.
        valid: True if normalization succeeded (enough visible keypoints).
        origin: (x, y) pixel origin used for translation.
        scale: pixel scale factor used for normalization.
    """

    keypoints: np.ndarray
    confidence: np.ndarray
    valid: bool
    origin: tuple[float, float]
    scale: float
    metadata: dict = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        """Concatenated (N*2,) vector of x,y coordinates (skips confidence)."""
        return self.keypoints.reshape(-1)


def _to_arrays(pose: PoseResult) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array([[k.x, k.y] for k in pose.keypoints], dtype=float)
    conf = np.array([k.confidence for k in pose.keypoints], dtype=float)
    return coords, conf


def _visible_mask(confidence: np.ndarray, min_conf: float = 0.5) -> np.ndarray:
    return confidence >= min_conf


def _valid_index(indices: list[int], visible: np.ndarray) -> int | None:
    """Return the first visible index, or None when none of the indices are visible."""
    for idx in indices:
        if visible[idx]:
            return idx
    return None


def normalize_pose(
    pose: PoseResult,
    min_confidence: float = 0.5,
    min_visible: int = MIN_VISIBLE_KEYPOINTS,
) -> NormalizedPose | None:
    """Translate so the hip-centre is at the origin and scale by torso length.

    Falls back to the nose as origin when the hips are not visible (scale then uses
    the shoulder-centre-to-nose distance). Returns None when too few keypoints are
    visible or the resulting scale is degenerate.
    """
    coords, conf = _to_arrays(pose)
    visible = _visible_mask(conf, min_confidence)
    if int(visible.sum()) < min_visible:
        return None

    hip_pts = np.array([coords[i] for i in (LEFT_HIP, RIGHT_HIP) if visible[i]])
    shoulder_pts = np.array([coords[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER) if visible[i]])
    nose_idx = _valid_index([NOSE], visible)

    if len(hip_pts):
        origin = hip_pts.mean(axis=0)
    elif nose_idx is not None:
        origin = coords[nose_idx]
    else:
        origin = coords[visible].mean(axis=0)

    if len(hip_pts) and len(shoulder_pts):
        scale = float(np.linalg.norm(shoulder_pts.mean(axis=0) - hip_pts.mean(axis=0)))
    elif len(shoulder_pts):
        # No hips: fall back to the shoulder-centre as the scale reference.
        reference = coords[nose_idx] if nose_idx is not None else origin
        scale = float(np.linalg.norm(shoulder_pts.mean(axis=0) - reference))
    elif nose_idx is not None and not np.allclose(coords[nose_idx], origin):
        scale = float(np.linalg.norm(coords[nose_idx] - origin))
    else:
        scale = float(np.std(coords[visible], axis=0).mean())

    if scale <= 1e-6:
        return None

    normalized = (coords - origin) / scale
    return NormalizedPose(
        keypoints=normalized,
        confidence=conf,
        valid=True,
        origin=(float(origin[0]), float(origin[1])),
        scale=scale,
        metadata=pose.metadata,
    )


def wrist_to_torso_vector(normalized: NormalizedPose, which: str = "right") -> np.ndarray:
    """Normalized vector from torso centre to the requested wrist (for features)."""
    torso = (normalized.keypoints[LEFT_SHOULDER] + normalized.keypoints[RIGHT_SHOULDER]) / 2
    wrist = normalized.keypoints[RIGHT_WRIST if which == "right" else LEFT_WRIST]
    return wrist - torso


__all__ = ["MIN_VISIBLE_KEYPOINTS", "NormalizedPose", "normalize_pose", "wrist_to_torso_vector"]
