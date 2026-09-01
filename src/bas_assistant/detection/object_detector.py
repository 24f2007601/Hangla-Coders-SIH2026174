"""YOLO object detector for microphone-protocol payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLASS_NAMES = {
    0: "microphone",
    1: "microphone_case_closed",
    2: "microphone_case_open",
    3: "phone_screen_on",
    4: "receiver",
}


class YOLOMicrophoneDetector:
    """Runs the trained YOLO model and returns normalized detection records."""

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.25,
        device: int | str = 0,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed.") from exc

        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._device = device

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        result = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )[0]

        if result.boxes is None:
            return []

        detections: list[dict] = []

        for box in result.boxes:
            class_id = int(box.cls.item())

            if class_id not in CLASS_NAMES:
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "name": CLASS_NAMES[class_id],
                    "confidence": float(box.conf.item()),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                }
            )

        return detections


__all__ = [
    "YOLOMicrophoneDetector",
]
