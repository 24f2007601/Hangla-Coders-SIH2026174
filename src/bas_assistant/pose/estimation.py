"""Pose estimation backends: MediaPipe (pretrained, frozen) and a deterministic dummy."""

from __future__ import annotations

import logging
import time

import numpy as np

from bas_assistant.models import BoundingBox, Keypoint, PoseResult
from bas_assistant.pose.landmarks import NUM_KEYPOINTS

logger = logging.getLogger(__name__)


class PoseEstimatorUnavailableError(RuntimeError):
    """Raised when the requested pose backend cannot be initialized."""


class MediaPipePoseEstimator:
    """MediaPipe Pose (+ optional Hands), pretrained and frozen.

    Keypoints are converted to pixel coordinates and stored in the canonical
    MediaPipe 33-point order (see `pose.landmarks`). Hand landmarks (21 per hand)
    are attached to `PoseResult.metadata["hands"]` so feature extraction can use
    hand-object interaction signals without seeing MediaPipe types.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        with_hands: bool = True,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on install
            raise PoseEstimatorUnavailableError(
                "mediapipe is not installed. Install the 'ml' extra or set pose.model=dummy."
            ) from exc

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = (
            mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            if with_hands
            else None
        )
        logger.info("Initialized MediaPipe pose estimator (hands=%s)", with_hands)

    def estimate(self, frame: np.ndarray) -> PoseResult | None:
        rgb = cv2_cvt_bgr_to_rgb(frame)
        ts = time.time()

        pose_out = self._pose.process(rgb)
        if pose_out is None or pose_out.pose_landmarks is None:
            return None

        height, width = frame.shape[:2]
        keypoints: list[Keypoint] = []
        for lm in pose_out.pose_landmarks.landmark:
            keypoints.append(Keypoint(x=lm.x * width, y=lm.y * height, confidence=lm.visibility))

        x_vals = [k.x for k in keypoints]
        y_vals = [k.y for k in keypoints]
        bbox = BoundingBox(
            x_min=max(int(min(x_vals)), 0),
            y_min=max(int(min(y_vals)), 0),
            x_max=min(int(max(x_vals)), width - 1),
            y_max=min(int(max(y_vals)), height - 1),
        )
        confidence = float(np.mean([k.confidence for k in keypoints]))

        metadata: dict = {}
        if self._hands is not None:
            hands_out = self._hands.process(rgb)
            metadata["hands"] = self._extract_hands(hands_out, width, height)

        return PoseResult(
            timestamp=ts,
            person_id=1,
            keypoints=keypoints,
            confidence=confidence,
            bounding_box=bbox,
            metadata=metadata,
        )

    @staticmethod
    def _extract_hands(hands_out, width: int, height: int) -> list[dict]:
        if hands_out is None or not hands_out.multi_hand_landmarks:
            return []
        hands: list[dict] = []
        for hand_landmarks, handedness in zip(
            hands_out.multi_hand_landmarks, hands_out.multi_handedness, strict=False
        ):
            if handedness.classification:
                label = str(handedness.classification[0].label)
                score = float(handedness.classification[0].score)
            else:
                label = "Unknown"
                score = 1.0
            hands.append(
                {
                    "handedness": label,
                    "confidence": score,
                    "keypoints": [
                        {"x": lm.x * width, "y": lm.y * height} for lm in hand_landmarks.landmark
                    ],
                }
            )
        return hands


class DummyPoseEstimator:
    """Deterministic synthetic pose so the pipeline runs with no MediaPipe install.

    Emits a plausible standing figure (canonical 33-point order, pixel coordinates)
    scaled to the frame. `motion` adds a gentle sinusoid to wrist/nose positions so
    temporal features are non-trivial in demos; leave it 0 for exact tests.
    """

    def __init__(self, motion: float = 0.0) -> None:
        self._motion = motion
        self._phase = 0.0

    def estimate(self, frame: np.ndarray) -> PoseResult | None:
        if frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        ts = time.time()
        cx = width / 2
        cy = height * 0.6
        u = height / 100.0  # unit = 1% of frame height
        m = self._motion * u * np.sin(self._phase)
        self._phase += 0.1

        pts: list[tuple[float, float]] = [
            (cx, cy - 42 * u),  # 0 nose
            (cx - 1.5 * u, cy - 44 * u),
            (cx - 2.5 * u, cy - 44 * u),
            (cx - 4 * u, cy - 44 * u),
            (cx + 1.5 * u, cy - 44 * u),
            (cx + 2.5 * u, cy - 44 * u),
            (cx + 4 * u, cy - 44 * u),
            (cx - 5 * u, cy - 43 * u),
            (cx + 5 * u, cy - 43 * u),
            (cx - 2 * u, cy - 40 * u),
            (cx + 2 * u, cy - 40 * u),
            (cx - 10 * u, cy - 32 * u),
            (cx + 10 * u, cy - 32 * u),
            (cx - 16 * u, cy - 18 * u),
            (cx + 16 * u, cy - 18 * u),
            (cx - 17 * u + m, cy - 4 * u),
            (cx + 17 * u + m, cy - 4 * u),
            (cx - 17 * u + m, cy - 1 * u),
            (cx + 17 * u + m, cy - 1 * u),
            (cx - 17 * u + m, cy - 1 * u),
            (cx + 17 * u + m, cy - 1 * u),
            (cx - 17 * u + m, cy - 3 * u),
            (cx + 17 * u + m, cy - 3 * u),
            (cx - 8 * u, cy),
            (cx + 8 * u, cy),
            (cx - 9 * u, cy + 22 * u),
            (cx + 9 * u, cy + 22 * u),
            (cx - 8 * u + m, cy + 42 * u),
            (cx + 8 * u + m, cy + 42 * u),
            (cx - 7 * u, cy + 44 * u),
            (cx + 7 * u, cy + 44 * u),
            (cx - 8 * u, cy + 45 * u),
            (cx + 8 * u, cy + 45 * u),
        ]
        assert len(pts) == NUM_KEYPOINTS
        keypoints = [Keypoint(x=x, y=y, confidence=1.0) for x, y in pts]
        return PoseResult(
            timestamp=ts,
            person_id=1,
            keypoints=keypoints,
            confidence=1.0,
            bounding_box=BoundingBox(0, 0, width - 1, height - 1),
            metadata={},
        )


def cv2_cvt_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


__all__ = [
    "DummyPoseEstimator",
    "MediaPipePoseEstimator",
    "PoseEstimatorUnavailableError",
    "cv2_cvt_bgr_to_rgb",
]
