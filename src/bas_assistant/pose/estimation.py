"""Per-frame perception backends: MediaPipe hand tracking (pretrained, frozen) and a
deterministic dummy.

The microphone-protocol pipeline has no use for body-pose landmarks: the fused
XGBoost features are built from **hand landmarks + YOLO object detections**, and
the receiver-LED verification gates are driven by YOLO + pixel analysis. Body
pose therefore was removed — the MediaPipe backend now runs only the
**HandLandmarker** (Tasks API, not the removed legacy `mp.solutions` API). Its
`.task` model is bundled on disk — see `scripts/download_mediapipe_models.py`.

The returned `PoseResult` remains the model-independent per-frame carrier:
`keypoints` is always empty, hands live in `metadata["hands"]`, and YOLO
detections / LED state are attached downstream by the pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from bas_assistant.models import BoundingBox, Keypoint, PoseResult
from bas_assistant.pose.landmarks import NUM_KEYPOINTS
from bas_assistant.utils.timing import Metrics

logger = logging.getLogger(__name__)


class PoseEstimatorUnavailableError(RuntimeError):
    """Raised when the requested perception backend cannot be initialized."""


class MediaPipeHandEstimator:
    """MediaPipe Hands only (pretrained, frozen) — no body-pose model.

    Hand inference runs on a dedicated worker thread in VIDEO mode so its ROI
    tracking — which carries a detected hand through frames where the palm
    detector momentarily fails — stays responsive. The main loop always hands
    the worker the newest frame (stale frames are replaced) and reads back the
    worker's latest result. A short output hold debounces residual one-off
    misses so markers do not flicker.

    Hand landmarks (21 per hand, pixel coordinates) are attached to
    `PoseResult.metadata["hands"]` so feature extraction can use hand-object
    interaction signals without seeing MediaPipe types. The result is the
    per-frame carrier for everything downstream (YOLO objects, LED state), so
    `estimate()` returns a PoseResult for every valid frame.
    """

    def __init__(
        self,
        hand_model_path: Path = Path("models/hand_landmarker.task"),
        min_hand_detection_confidence: float = 0.3,
        min_hand_presence_confidence: float = 0.4,
        min_hand_tracking_confidence: float = 0.4,
        hand_hold_seconds: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depends on install
            raise PoseEstimatorUnavailableError(
                "mediapipe is not installed. Install the 'ml' extra or set pose.model=dummy."
            ) from exc

        if not Path(hand_model_path).exists():
            raise PoseEstimatorUnavailableError(
                f"MediaPipe hand model not found at {hand_model_path}. Run "
                "`python scripts/download_mediapipe_models.py` or set pose.model=dummy."
            )

        self._mp = mp
        self._vision = vision
        self._metrics = Metrics()
        self._hands_slot: np.ndarray | None = None
        self._hands_slot_lock = threading.Lock()
        self._hands_pending = threading.Event()
        self._hands_thread: threading.Thread | None = None
        self._hands_cache: object | None = None
        self._hands_lock = threading.Lock()
        self._hands_stop = threading.Event()
        self._last_hands_ts_ms = 0
        self._hand_hold_seconds = max(hand_hold_seconds, 0.0)
        self._last_detected: list[dict] = []
        self._last_detected_ts = 0.0
        self._hands_detected = 0
        self._hands_held = 0
        self._hands_missed = 0
        self._hands_frames_dropped = 0
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_hand_tracking_confidence,
            )
        )
        self._hands_thread = threading.Thread(
            target=self._hand_worker, name="hand-landmarker", daemon=True
        )
        self._hands_thread.start()
        logger.info("Initialized MediaPipe hand estimator (hold=%.1fs)", self._hand_hold_seconds)

    def close(self) -> None:
        """Signal the hand worker thread to stop and join it briefly."""
        if self._hands_thread is not None:
            self._hands_stop.set()
            self._hands_pending.set()
            self._hands_thread.join(timeout=2.0)
            self._hands_thread = None

    @property
    def metrics(self) -> Metrics:
        """Timing metrics for hand inference (diagnostics)."""
        return self._metrics

    def hands_diagnostics(self) -> dict:
        """Counts separating detector misses from stale/held landmarks and worker lag.

        - ``detected``: frames with a fresh hand detection.
        - ``held_stale``: frames where the previous detection was shown because the
          hold window had not expired (landmarks can lag the video hand here).
        - ``missed``: frames with no hand shown (genuine detector failure).
        - ``worker_frames_dropped``: frames skipped because the hand worker thread
          was still busy with an older frame (hand results lag the frame).
        """
        return {
            "detected": self._hands_detected,
            "held_stale": self._hands_held,
            "missed": self._hands_missed,
            "worker_frames_dropped": self._hands_frames_dropped,
        }

    def _next_timestamp_ms(self, last: int) -> int:
        """Strictly-increasing monotonic timestamp (ms) for VIDEO-mode detect calls."""
        now = int(time.monotonic() * 1000)
        if now <= last:
            now = last + 1
        return now

    def _next_hands_timestamp_ms(self) -> int:
        self._last_hands_ts_ms = self._next_timestamp_ms(self._last_hands_ts_ms)
        return self._last_hands_ts_ms

    def _hand_worker(self) -> None:
        """Consume the newest frame and cache the latest hand-landmark result."""
        while not self._hands_stop.is_set():
            if not self._hands_pending.wait(timeout=0.1):
                continue
            with self._hands_slot_lock:
                rgb = self._hands_slot
                self._hands_slot = None
                self._hands_pending.clear()
            if rgb is None:
                continue
            try:
                image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                start = time.perf_counter()
                out = self._hands.detect_for_video(image, self._next_hands_timestamp_ms())
                self._metrics.record("hands", time.perf_counter() - start)
                with self._hands_lock:
                    self._hands_cache = out
            except Exception:  # noqa: BLE001 - log and keep the worker alive
                logger.exception("Hand landmarker detection failed")

    def _exposed_hands(self, hands_out, width: int, height: int) -> list[dict]:
        """Hand landmarks with a debounce hold against one-off detection misses.

        Each call is tallied as one of: fresh ``detected``, ``held`` (stale
        landmarks shown during a miss), or ``missed`` (nothing shown). Combined
        with the worker-frame-drop counter this separates genuine detector
        misses from stale-landmark rendering artifacts.
        """
        current = self._extract_hands(hands_out, width, height)
        now = time.monotonic()
        if current:
            if not self._last_detected:
                logger.debug("hands: detection resumed (%d hand(s))", len(current))
            self._hands_detected += 1
            self._last_detected = current
            self._last_detected_ts = now
            return current
        if self._last_detected and now - self._last_detected_ts < self._hand_hold_seconds:
            self._hands_held += 1
            return self._last_detected
        if self._last_detected:
            logger.debug("hands: hold expired after %.2fs (miss)", now - self._last_detected_ts)
            self._last_detected = []
        self._hands_missed += 1
        return []

    def estimate(self, frame: np.ndarray) -> PoseResult | None:
        if frame is None or frame.size == 0:
            return None

        rgb = cv2_cvt_bgr_to_rgb(frame)
        ts = time.time()

        with self._hands_slot_lock:
            if self._hands_slot is not None:
                # Worker still busy with an older frame -> it will skip this one.
                self._hands_frames_dropped += 1
            self._hands_slot = rgb
            self._hands_pending.set()
        with self._hands_lock:
            hands_out = self._hands_cache

        height, width = frame.shape[:2]
        hands = self._exposed_hands(hands_out, width, height)

        confidences = [float(hand.get("confidence", 0.0)) for hand in hands]
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return PoseResult(
            timestamp=ts,
            person_id=1,
            # No body-pose model: hands (metadata) are the perception signal.
            keypoints=[],
            confidence=confidence,
            bounding_box=BoundingBox(0, 0, width - 1, height - 1),
            metadata={"hands": hands},
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
    "MediaPipeHandEstimator",
    "PoseEstimatorUnavailableError",
    "cv2_cvt_bgr_to_rgb",
]
