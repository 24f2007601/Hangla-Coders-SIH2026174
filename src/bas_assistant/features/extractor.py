"""Spatial + temporal feature extraction from normalized poses.

The extractor converts a sliding window of normalized poses into a fixed-length
feature vector consumed by the step classifier.

Features include:
- Spatial distances and angles
- Hand position relative to the torso
- X/Y velocity
- Motion speed
- Motion variability
- Total motion over the window
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bas_assistant.features.hands import palm_center
from bas_assistant.features.window import FeatureWindow
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

MIN_VISIBLE_KEYPOINTS = 8


# ---------------------------------------------------------
# Spatial features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Hand features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Tracked joints
# ---------------------------------------------------------

TRACKED_FEATURES: tuple[str, ...] = (
    "nose_x",
    "nose_y",
    "right_wrist_x",
    "right_wrist_y",
    "left_wrist_x",
    "left_wrist_y",
)


# ---------------------------------------------------------
# Temporal features
# ---------------------------------------------------------

TEMPORAL_FEATURES: tuple[str, ...] = (
    "nose_vx",
    "nose_vy",
    "right_wrist_vx",
    "right_wrist_vy",
    "left_wrist_vx",
    "left_wrist_vy",
    "nose_speed",
    "right_wrist_speed",
    "left_wrist_speed",
    "nose_speed_std",
    "right_wrist_speed_std",
    "left_wrist_speed_std",
    "nose_total_motion",
    "right_wrist_total_motion",
    "left_wrist_total_motion",
)


WINDOW_STATS = ("mean", "std")

_NUM_SPATIAL = len(SPATIAL_FEATURES)
_NUM_TEMPORAL = len(TEMPORAL_FEATURES)


# ---------------------------------------------------------
# Frame container
# ---------------------------------------------------------


@dataclass(slots=True)
class FrameFeatures:
    spatial: np.ndarray
    tracked: np.ndarray
    hands: np.ndarray | None = None


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------


def _angle(
    a: np.ndarray,
    vertex: np.ndarray,
    b: np.ndarray,
) -> float:
    """Angle in radians at vertex between a and b."""

    v1 = a - vertex
    v2 = b - vertex

    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm < 1e-9:
        return 0.0

    cos = float(
        np.clip(
            np.dot(v1, v2) / norm,
            -1.0,
            1.0,
        )
    )

    return float(np.arccos(cos))


# ---------------------------------------------------------
# Per-frame spatial extraction
# ---------------------------------------------------------


def extract_frame_features(
    normalized: NormalizedPose,
) -> FrameFeatures:

    kp = normalized.keypoints
    conf = normalized.confidence

    hands = normalized.metadata.get("hands", [])

    left_palm = np.array(
        [0.0, 0.0],
        dtype=float,
    )

    right_palm = np.array(
        [0.0, 0.0],
        dtype=float,
    )

    left_present = 0.0
    right_present = 0.0

    for hand in hands:
        keypoints = hand.get(
            "keypoints",
            [],
        )

        handedness = hand.get(
            "handedness",
            "",
        ).lower()

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

    l_sh = kp[LEFT_SHOULDER]
    r_sh = kp[RIGHT_SHOULDER]

    l_el = kp[LEFT_ELBOW]
    r_el = kp[RIGHT_ELBOW]

    l_w = kp[LEFT_WRIST]
    r_w = kp[RIGHT_WRIST]

    nose = kp[NOSE]

    hip_idx = [
        i
        for i in (
            LEFT_HIP,
            RIGHT_HIP,
        )
        if i in visible
    ]

    if hip_idx:
        hip_center = np.mean(
            [kp[i] for i in hip_idx],
            axis=0,
        )
    else:
        hip_center = (kp[LEFT_HIP] + kp[RIGHT_HIP]) / 2

    def dist(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
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
            float(
                min(
                    conf[RIGHT_WRIST],
                    conf[LEFT_WRIST],
                )
            ),
        ],
        dtype=float,
    )

    tracked = np.array(
        [
            nose[0],
            nose[1],
            r_w[0],
            r_w[1],
            l_w[0],
            l_w[1],
        ],
        dtype=float,
    )

    torso = (normalized.keypoints[LEFT_SHOULDER] + normalized.keypoints[RIGHT_SHOULDER]) / 2

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


# ---------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------


class PoseFeatureExtractor:
    """Accumulate a pose window and produce classifier features."""

    def __init__(
        self,
        sequence_length: int = 30,
    ) -> None:

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

    def push(
        self,
        pose: PoseResult,
    ) -> bool:

        normalized = normalize_pose(pose)

        if normalized is None:
            return False

        frame = extract_frame_features(normalized)

        # -------------------------------------------------
        # Calculate velocity.
        # -------------------------------------------------

        if self._last_tracked is None:
            velocity = np.zeros(
                len(TRACKED_FEATURES),
                dtype=float,
            )

        else:
            velocity = frame.tracked - self._last_tracked

        self._last_tracked = frame.tracked.copy()

        # -------------------------------------------------
        # Calculate speed for each tracked joint.
        #
        # nose       -> indices 0,1
        # right wrist -> indices 2,3
        # left wrist  -> indices 4,5
        # -------------------------------------------------

        nose_speed = float(np.linalg.norm(velocity[0:2]))

        right_wrist_speed = float(np.linalg.norm(velocity[2:4]))

        left_wrist_speed = float(np.linalg.norm(velocity[4:6]))

        # -------------------------------------------------
        # Speed starts as one value per frame.
        # The extractor later calculates std/total motion.
        # -------------------------------------------------

        temporal = np.array(
            [
                velocity[0],
                velocity[1],
                velocity[2],
                velocity[3],
                velocity[4],
                velocity[5],
                nose_speed,
                right_wrist_speed,
                left_wrist_speed,
                # These are populated later using the
                # window-level calculation.
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        raw = np.concatenate(
            [
                frame.spatial,
                temporal,
                frame.hands,
            ]
        )

        self._window.push(raw)

        if self._window.is_full:
            self._ready = True

        return self._window.is_full

    def features(self) -> np.ndarray:
        """Return the aggregated feature vector."""

        if not self._ready:
            raise ValueError("feature window not full; call push() until it returns True")

        items = np.asarray(
            self._window.items(),
            dtype=float,
        )

        # -------------------------------------------------
        # Split stored frame features.
        # -------------------------------------------------

        spatial = items[
            :,
            :_NUM_SPATIAL,
        ]

        temporal = items[
            :,
            _NUM_SPATIAL : _NUM_SPATIAL + _NUM_TEMPORAL,
        ]

        hands = items[:, _NUM_SPATIAL + _NUM_TEMPORAL :]

        # -------------------------------------------------
        # Spatial statistics.
        # -------------------------------------------------

        spatial_mean = spatial.mean(axis=0)

        spatial_std = spatial.std(axis=0)

        # -------------------------------------------------
        # Signed velocity.
        #
        # Useful for direction.
        # -------------------------------------------------

        velocity_mean = temporal[
            :,
            0:6,
        ].mean(axis=0)

        # -------------------------------------------------
        # Speed statistics.
        #
        # Unlike signed velocity, speed cannot cancel
        # when the hand changes direction.
        # -------------------------------------------------

        speed = temporal[
            :,
            6:9,
        ]

        speed_mean = speed.mean(axis=0)

        speed_std = speed.std(axis=0)

        speed_total = speed.sum(axis=0)

        # -------------------------------------------------
        # Hand statistics.
        # -------------------------------------------------

        hands_mean = hands.mean(axis=0)

        return np.concatenate(
            [
                spatial_mean,
                spatial_std,
                velocity_mean,
                speed_mean,
                speed_std,
                speed_total,
                hands_mean,
            ]
        )

    @property
    def feature_names(self) -> tuple[str, ...]:

        return (
            *[f"{name}_{stat}" for stat in WINDOW_STATS for name in SPATIAL_FEATURES],
            *TEMPORAL_FEATURES,
            *HAND_FEATURES,
        )


# ---------------------------------------------------------
# Feature vector size
# ---------------------------------------------------------

FEATURE_VECTOR_SIZE = 2 * len(SPATIAL_FEATURES) + 6 + 3 + 3 + 3 + len(HAND_FEATURES)


__all__ = [
    "SPATIAL_FEATURES",
    "TEMPORAL_FEATURES",
    "TRACKED_FEATURES",
    "HAND_FEATURES",
    "FEATURE_VECTOR_SIZE",
    "PoseFeatureExtractor",
    "extract_frame_features",
]
