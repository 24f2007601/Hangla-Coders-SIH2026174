"""Timestamped receiver LED-state estimation for G1/G2 verification."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class LEDObservation:
    """The latest localized receiver and its two independent LED states."""

    left: str = "unknown"
    right: str = "unknown"
    left_score: float = 0.0
    right_score: float = 0.0
    receiver_detected: bool = False
    receiver_bbox: tuple[float, float, float, float] | None = None
    receiver_confidence: float = 0.0
    history_seconds: float = 0.0
    sample_count: int = 0
    ready: bool = False
    g1_passed: bool = False
    g2_passed: bool = False


class LEDStateEstimator:
    """Classify the two receiver LEDs from YOLO-localized video frames.

    The receiver in the recorded experiment is horizontal. The two normalized
    search bands were calibrated from its actual YOLO crop: the left LED lies
    close to the crop's left edge and the right LED near its centre-right.
    Scores use the strongest blue pixels, which survives low LED resolution.
    """

    WINDOW_SECONDS = 2.0
    MIN_HISTORY_SECONDS = 0.6
    MIN_SAMPLES = 12
    MAX_LOCALIZATION_GAP_SECONDS = 0.30
    MIN_RECEIVER_CONFIDENCE = 0.25

    # Calibrated against re_total_experiment_v5.mp4 receiver detections.
    LEFT_ROI = (0.03, 0.28, 0.30, 0.78)
    RIGHT_ROI = (0.43, 0.28, 0.74, 0.78)

    ON_THRESHOLD = 0.18
    STEADY_ON_FRACTION = 0.78
    BLINK_MIN_ON_FRACTION = 0.18
    BLINK_MAX_ON_FRACTION = 0.72
    MIN_BLINK_TRANSITIONS = 2

    def __init__(self) -> None:
        self._history: deque[tuple[float, float, float]] = deque()
        self._last_sample_timestamp: float | None = None
        self._last_observation = LEDObservation()

    @property
    def state(self) -> dict[str, str]:
        return {"left": self._last_observation.left, "right": self._last_observation.right}

    @property
    def scores(self) -> dict[str, float]:
        return {
            "left": self._last_observation.left_score,
            "right": self._last_observation.right_score,
        }

    @property
    def g1_passed(self) -> bool:
        return self._last_observation.g1_passed

    @property
    def g2_passed(self) -> bool:
        return self._last_observation.g2_passed

    @property
    def observation(self) -> LEDObservation:
        return self._last_observation

    def reset(self) -> None:
        self.reset_history()

    def reset_history(self) -> None:
        """Discard evidence so a new verification gate starts cleanly."""
        self._history.clear()
        self._last_sample_timestamp = None
        self._last_observation = LEDObservation()

    def update(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
        timestamp: float | None = None,
    ) -> dict[str, object]:
        """Ingest one video frame using its source timestamp, never wall time."""
        sample_timestamp = time.monotonic() if timestamp is None else float(timestamp)
        receiver = self._best_receiver(objects)

        if receiver is None:
            self._trim_history(sample_timestamp)
            self._last_observation = LEDObservation(
                history_seconds=self._history_seconds(),
                sample_count=len(self._history),
            )
            return self._as_dict()

        bbox = self._valid_bbox(receiver, frame)
        if bbox is None:
            self._trim_history(sample_timestamp)
            self._last_observation = LEDObservation(
                history_seconds=self._history_seconds(),
                sample_count=len(self._history),
            )
            return self._as_dict()

        has_timestamp_gap = (
            self._last_sample_timestamp is not None
            and sample_timestamp - self._last_sample_timestamp > self.MAX_LOCALIZATION_GAP_SECONDS
        )
        has_non_monotonic_timestamp = (
            self._last_sample_timestamp is not None
            and sample_timestamp <= self._last_sample_timestamp
        )
        if has_timestamp_gap or has_non_monotonic_timestamp:
            self._history.clear()

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        left_score = self._blue_score(crop, self.LEFT_ROI)
        right_score = self._blue_score(crop, self.RIGHT_ROI)

        self._history.append((sample_timestamp, left_score, right_score))
        self._last_sample_timestamp = sample_timestamp
        self._trim_history(sample_timestamp)

        history_seconds = self._history_seconds()
        ready = (
            len(self._history) >= self.MIN_SAMPLES and history_seconds >= self.MIN_HISTORY_SECONDS
        )
        left_state, right_state = ("unknown", "unknown")
        if ready:
            left_values = np.asarray([row[1] for row in self._history], dtype=float)
            right_values = np.asarray([row[2] for row in self._history], dtype=float)
            left_state = self._classify(left_values)
            right_state = self._classify(right_values)

        g1_passed = left_state == "blinking" and right_state == "blinking"
        g2_passed = {left_state, right_state} == {"steady", "blinking"}
        self._last_observation = LEDObservation(
            left=left_state,
            right=right_state,
            left_score=left_score,
            right_score=right_score,
            receiver_detected=True,
            receiver_bbox=tuple(float(value) for value in receiver["xyxy"]),
            receiver_confidence=float(receiver["confidence"]),
            history_seconds=history_seconds,
            sample_count=len(self._history),
            ready=ready,
            g1_passed=g1_passed,
            g2_passed=g2_passed,
        )
        return self._as_dict()

    def _trim_history(self, timestamp: float) -> None:
        cutoff = timestamp - self.WINDOW_SECONDS
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _history_seconds(self) -> float:
        if len(self._history) < 2:
            return 0.0
        return max(0.0, self._history[-1][0] - self._history[0][0])

    def _as_dict(self) -> dict[str, object]:
        observation = self._last_observation
        return {
            "receiver_detected": observation.receiver_detected,
            "receiver_bbox": list(observation.receiver_bbox) if observation.receiver_bbox else None,
            "receiver_confidence": round(observation.receiver_confidence, 4),
            "left": observation.left,
            "right": observation.right,
            "left_score": round(observation.left_score, 4),
            "right_score": round(observation.right_score, 4),
            "history_seconds": round(observation.history_seconds, 3),
            "sample_count": observation.sample_count,
            "ready": observation.ready,
            "g1_passed": observation.g1_passed,
            "g2_passed": observation.g2_passed,
        }

    @classmethod
    def _best_receiver(cls, objects: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for obj in objects:
            if str(obj.get("name", "")) != "receiver":
                continue
            try:
                confidence = float(obj.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            xyxy = obj.get("xyxy")
            if (
                confidence >= cls.MIN_RECEIVER_CONFIDENCE
                and isinstance(xyxy, (list, tuple))
                and len(xyxy) == 4
            ):
                candidates.append({**obj, "confidence": confidence})
        return max(candidates, key=lambda item: float(item["confidence"]), default=None)

    @staticmethod
    def _valid_bbox(
        receiver: dict[str, Any], frame: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        try:
            x1, y1, x2, y2 = (float(value) for value in receiver["xyxy"])
        except (KeyError, TypeError, ValueError):
            return None
        height, width = frame.shape[:2]
        x1_i = max(0, min(width - 1, round(x1)))
        y1_i = max(0, min(height - 1, round(y1)))
        x2_i = max(x1_i + 1, min(width, round(x2)))
        y2_i = max(y1_i + 1, min(height, round(y2)))
        if x2_i - x1_i < 20 or y2_i - y1_i < 12:
            return None
        return x1_i, y1_i, x2_i, y2_i

    @classmethod
    def _blue_score(cls, crop: np.ndarray, roi: tuple[float, float, float, float]) -> float:
        height, width = crop.shape[:2]
        rx1, ry1, rx2, ry2 = roi
        x1 = max(0, min(width - 1, round(rx1 * width)))
        y1 = max(0, min(height - 1, round(ry1 * height)))
        x2 = max(x1 + 1, min(width, round(rx2 * width)))
        y2 = max(y1 + 1, min(height, round(ry2 * height)))
        patch = crop[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0

        bgr = patch.astype(np.float32)
        blue = bgr[:, :, 0] / 255.0
        non_blue = np.maximum(bgr[:, :, 1], bgr[:, :, 2]) / 255.0
        evidence = np.clip(blue - non_blue, 0.0, 1.0) * blue
        return float(np.percentile(evidence, 99.0))

    @classmethod
    def _classify(cls, values: np.ndarray) -> str:
        on = values >= cls.ON_THRESHOLD
        on_fraction = float(np.mean(on))
        if on_fraction < cls.BLINK_MIN_ON_FRACTION:
            return "off"
        transitions = int(np.count_nonzero(on[1:] != on[:-1]))
        if (
            cls.BLINK_MIN_ON_FRACTION <= on_fraction <= cls.BLINK_MAX_ON_FRACTION
            and transitions >= cls.MIN_BLINK_TRANSITIONS
        ):
            return "blinking"
        if on_fraction >= cls.STEADY_ON_FRACTION:
            return "steady"
        return "unknown"


__all__ = ["LEDObservation", "LEDStateEstimator"]
