"""Spatial + temporal feature extraction from normalized poses.

The extractor converts a sliding window of normalized poses into a fixed-length
feature vector consumed by the step classifier. Per-frame features are simple,
deterministic geometric statistics (distances, angles, coordinates); the window
aggregator reduces them to mean/std statistics plus mean joint velocities.

The output is a plain 1-D float array with a documented feature order — the step
classifier never sees raw keypoints or MediaPipe internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bas_assistant.features.window import FeatureWindow
from bas_assistant.features.hands import palm_center
from bas_assistant.models import PoseResult
from bas_assistant.pose.landmarks import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from bas_assistant.pose.normalization import NormalizedPose, normalize_pose

# Per-frame spatial feature names (in output order). All in normalized, scale-free units.
SPATIAL_FEATURES: tuple[str, ...] = (
    "right_wrist_left_shoulder",
    "right_wrist_right_shoulder",
    "left_wrist_left_shoulder",
    "left_wrist_right_shoulder",
    "right_wrist_nose",
    "left_wrist_nose",
    "right_wrist_hip_center",
    "left_wrist_hip_center",
    "wrist_wrist",
    "right_elbow_angle",
    "left_elbow_angle",
    "nose_x",
    "nose_y",
    "wrist_confidence",
)

HAND_FEATURES: tuple[str, ...] = (
    "left_palm_x",
    "left_palm_y",
    "right_palm_x",
    "right_palm_y",
    "left_hand_present",
    "right_hand_present",
    "left_hand_to_torso_x",
    "left_hand_to_torso_y",
    "right_hand_to_torso_x",
    "right_hand_to_torso_y",
)

# Per-frame reference joint coordinates (normalized space), used for velocities.
TRACKED_FEATURES: tuple[str, ...] = (
    "nose_x",
    "nose_y",
    "right_wrist_x",
    "right_wrist_y",
    "left_wrist_x",
    "left_wrist_y",
)

# Per-frame temporal feature names (velocity of reference joints).
TEMPORAL_FEATURES: tuple[str, ...] = (
    "nose_vx",
    "nose_vy",
    "right_wrist_vx",
    "right_wrist_vy",
    "left_wrist_vx",
    "left_wrist_vy",
)

WINDOW_STATS = ("mean", "std")

# Fixed feature-vector length emitted by PoseFeatureExtractor.features().
FEATURE_VECTOR_SIZE = (
    2 * len(SPATIAL_FEATURES)
    + len(TEMPORAL_FEATURES)
    + len(HAND_FEATURES)
)

_NUM_SPATIAL = len(SPATIAL_FEATURES)
_NUM_TRACKED = len(TRACKED_FEATURES)


@dataclass(slots=True)
class FrameFeatures:
    spatial: np.ndarray  # (len(SPATIAL_FEATURES),)
    tracked: np.ndarray  # (len(TRACKED_FEATURES),)
    hands: np.ndarray | None = None

def _angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle (radians) at `vertex` between vectors (a - vertex) and (b - vertex)."""
    v1 = a - vertex
    v2 = b - vertex
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))
    return float(np.arccos(cos))


def extract_frame_features(normalized: NormalizedPose) -> FrameFeatures:
    """Compute spatial + tracked features for one normalized pose."""
    kp = normalized.keypoints
    conf = normalized.confidence

    hands = normalized.metadata.get("hands", [])

    left_palm = np.array([0.0, 0.0], dtype=float)
    right_palm = np.array([0.0, 0.0], dtype=float)
    left_present = 0.0
    right_present = 0.0

    for hand in hands:
        keypoints = hand.get("keypoints", [])
        handedness = hand.get("handedness", "").lower()

        if len(keypoints) != 21:
            continue

        center = palm_center(keypoints)

        if handedness == "left":
            left_palm = center
            left_present = 1.0
        elif handedness == "right":
            right_palm = center
            right_present = 1.0

    visible = {i for i in range(len(kp)) if conf[i] >= 0.5}

    l_sh, r_sh = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]
    l_el, r_el = kp[LEFT_ELBOW], kp[RIGHT_ELBOW]
    l_w, r_w = kp[LEFT_WRIST], kp[RIGHT_WRIST]
    nose = kp[NOSE]

    hip_idx = [i for i in (LEFT_HIP, RIGHT_HIP) if i in visible]
    hip_center = (
        np.mean([kp[i] for i in hip_idx], axis=0) if hip_idx else (kp[LEFT_HIP] + kp[RIGHT_HIP]) / 2
    )

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    spatial = np.array(
        [
            dist(r_w, l_sh),
            dist(r_w, r_sh),
            dist(l_w, l_sh),
            dist(l_w, r_sh),
            dist(r_w, nose),
            dist(l_w, nose),
            dist(r_w, hip_center),
            dist(l_w, hip_center),
            dist(r_w, l_w),
            _angle(r_w, r_el, r_sh),
            _angle(l_w, l_el, l_sh),
            nose[0],
            nose[1],
            float(min(conf[RIGHT_WRIST], conf[LEFT_WRIST])),
        ],
        dtype=float,
    )
    tracked = np.array(
        [nose[0], nose[1], r_w[0], r_w[1], l_w[0], l_w[1]],
        dtype=float,
    )
    torso = (
        normalized.keypoints[LEFT_SHOULDER]
        + normalized.keypoints[RIGHT_SHOULDER]
    ) / 2

    left_relative = left_palm - torso
    right_relative = right_palm - torso

    hands_array = np.array(
        [
            left_palm[0],
            left_palm[1],
            right_palm[0],
            right_palm[1],
            left_present,
            right_present,
            left_relative[0],
            left_relative[1],
            right_relative[0],
            right_relative[1],
        ],
        dtype=float,
    )

    return FrameFeatures(
        spatial=spatial,
        tracked=tracked,
        hands=hands_array,
    )


class PoseFeatureExtractor:
    """Accumulates normalized poses and emits aggregated window feature vectors.

    The returned vector has length ``2 * len(SPATIAL_FEATURES) + len(TEMPORAL_FEATURES)``
    (34 for the default feature set): mean and std of each spatial feature over the
    window, followed by the mean velocity of each tracked joint over the window.
    """

    def __init__(self, sequence_length: int = 30) -> None:
        self._window = FeatureWindow[np.ndarray](sequence_length)
        self._last_tracked: np.ndarray | None = None
        self._ready = False

    @property
    def sequence_length(self) -> int:
        return self._window.size

    @property
    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._window.clear()
        self._last_tracked = None
        self._ready = False

    def push(self, pose: PoseResult) -> bool:
        normalized = normalize_pose(pose)
        if normalized is None:
            return False
        frame = extract_frame_features(normalized)
        temporal = np.zeros(len(TEMPORAL_FEATURES), dtype=float)
        if self._last_tracked is not None:
            temporal = frame.tracked - self._last_tracked
        self._last_tracked = frame.tracked
        raw = np.concatenate([frame.spatial, temporal, frame.hands])
        self._window.push(raw)
        if self._window.is_full:
            self._ready = True
        return self._window.is_full

    def features(self) -> np.ndarray:
        """Aggregated (34,) vector: mean/std of spatial + mean of temporal features."""
        if not self._ready:
            raise ValueError("feature window not full; call push() until it returns True")
        items = np.asarray(self._window.items(), dtype=float)  # (W, 26)
        spatial = items[:, :_NUM_SPATIAL]
        temporal = items[:, _NUM_SPATIAL:_NUM_SPATIAL + len(TEMPORAL_FEATURES)]
        hands = items[:, _NUM_SPATIAL + len(TEMPORAL_FEATURES):]
        return np.concatenate(
            [
                spatial.mean(axis=0),
                spatial.std(axis=0),
                temporal.mean(axis=0),
                hands.mean(axis=0),
            ]
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *[f"{name}_{stat}" for stat in WINDOW_STATS for name in SPATIAL_FEATURES],
            *TEMPORAL_FEATURES,
            *HAND_FEATURES,
        )


__all__ = [
    "SPATIAL_FEATURES",
    "TEMPORAL_FEATURES",
    "TRACKED_FEATURES",
    "FEATURE_VECTOR_SIZE",
    "PoseFeatureExtractor",
    "extract_frame_features",
]
