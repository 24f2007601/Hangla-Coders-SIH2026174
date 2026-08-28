"""Pose estimation backends: MediaPipe (pretrained, frozen) and a deterministic dummy.

MediaPipe is used through the current **Tasks API** (`PoseLandmarker` +
`HandLandmarker`), not the removed legacy `mp.solutions` API. Tasks API models are
bundled as `.task` files — see `scripts/download_mediapipe_models.py`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from bas_assistant.models import BoundingBox, Keypoint, PoseResult
from bas_assistant.pose.landmarks import NUM_KEYPOINTS

logger = logging.getLogger(__name__)


class PoseEstimatorUnavailableError(RuntimeError):
    """Raised when the requested pose backend cannot be initialized."""


class MediaPipePoseEstimator:
    """MediaPipe Pose (+ optional Hands), pretrained and frozen.

    Backed by the Tasks API `PoseLandmarker` / `HandLandmarker` running in VIDEO
    mode. Keypoints are converted to pixel coordinates and stored in the canonical
    MediaPipe 33-point order (see `pose.landmarks`). Hand landmarks (21 per hand)
    are attached to `PoseResult.metadata["hands"]` so feature extraction can use
    hand-object interaction signals without seeing MediaPipe types.
    """

    def __init__(
        self,
        pose_model_path: Path = Path("models/pose_landmarker_lite.task"),
        hand_model_path: Path = Path("models/hand_landmarker.task"),
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        with_hands: bool = True,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depends on install
            raise PoseEstimatorUnavailableError(
                "mediapipe is not installed. Install the 'ml' extra or set pose.model=dummy."
            ) from exc

        if not Path(pose_model_path).exists():
            raise PoseEstimatorUnavailableError(
                f"MediaPipe pose model not found at {pose_model_path}. Run "
                "`python scripts/download_mediapipe_models.py` or set pose.model=dummy."
            )

        self._mp = mp
        self._vision = vision
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self._hands = None
        if with_hands:
            if not Path(hand_model_path).exists():
                raise PoseEstimatorUnavailableError(
                    f"MediaPipe hand model not found at {hand_model_path}. Run "
                    "`python scripts/download_mediapipe_models.py` or set pose.with_hands=false."
                )
            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            )
        self._frame_ts_ms = 0
        logger.info("Initialized MediaPipe pose estimator (hands=%s)", with_hands)

    def estimate(self, frame: np.ndarray) -> PoseResult | None:
        rgb = cv2_cvt_bgr_to_rgb(frame)
        ts = time.time()
        self._frame_ts_ms += 33  # strictly increasing timestamp required by VIDEO mode
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        pose_out = self._pose.detect_for_video(image, self._frame_ts_ms)
        if pose_out is None or not pose_out.pose_landmarks:
            return None

        height, width = frame.shape[:2]
        keypoints: list[Keypoint] = []
        for lm in pose_out.pose_landmarks[0]:
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
            hands_out = self._hands.detect_for_video(image, self._frame_ts_ms)
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
        if hands_out is None or not hands_out.hand_landmarks:
            return []
        hands: list[dict] = []
        for hand_landmarks, handedness in zip(
            hands_out.hand_landmarks, hands_out.handedness, strict=False
        ):
            categories = handedness if isinstance(handedness, list) else handedness.categories
            if categories:
                label = str(categories[0].category_name)
                score = float(categories[0].score)
            else:
                label = "Unknown"
                score = 1.0
            hands.append(
                {
                    "handedness": label,
                    "confidence": score,
                    "keypoints": [{"x": lm.x * width, "y": lm.y * height} for lm in hand_landmarks],
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
