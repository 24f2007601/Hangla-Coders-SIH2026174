"""XGBoost step classifier wrapper.

Gracefully handles a missing or unloadable model: prediction falls back to
("unknown", 0.0) and a warning is logged. The model is never trained here —
training happens in `scripts/` / notebooks with the recorded dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bas_assistant.classification.models import UNKNOWN_STEP, ClassificationResult

logger = logging.getLogger(__name__)


class XGBoostStepClassifier:
    """Loads a trained XGBoost model from disk and predicts step ids.

    Expected model output classes are the protocol step ids (+ "background"),
    either stored in the model's ``classes_`` or in a sidecar
    ``<model_path>.labels.json`` next to the model file.
    """

    def __init__(self, model_path: Path, expected_feature_size: int | None = None) -> None:
        self._model = None
        self._classes: list[str] = []
        self._feature_size = expected_feature_size

        if not model_path.exists():
            logger.warning(
                "XGBoost model not found at %s; falling back to unknown predictions", model_path
            )
            return

        try:
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            self._classes = self._load_classes(model_path, model)
            self._model = model
            logger.info(
                "Loaded XGBoost step classifier from %s (%d classes)",
                model_path,
                len(self._classes),
            )
        except Exception as exc:  # noqa: BLE001 - report and fall back
            logger.error("Failed to load XGBoost model %s: %s", model_path, exc)
            self._model = None

    @staticmethod
    def _load_classes(model_path: Path, model) -> list[str]:
        labels_path = model_path.with_suffix(model_path.suffix + ".labels.json")
        if labels_path.exists():
            try:
                data = json.loads(labels_path.read_text(encoding="utf-8"))
                classes = [str(c) for c in data["classes"]]
                if classes:
                    return classes
            except (ValueError, KeyError) as exc:
                logger.warning("Could not read labels file %s: %s", labels_path, exc)
        return [str(c) for c in model.classes_]

    def classify(self, features: np.ndarray) -> ClassificationResult:
        if self._model is None:
            return ClassificationResult(step=UNKNOWN_STEP, confidence=0.0)
        if self._feature_size is not None and features.shape[0] != self._feature_size:
            logger.error(
                "Feature size mismatch: model expects %d, got %d",
                self._feature_size,
                features.shape[0],
            )
            return ClassificationResult(step=UNKNOWN_STEP, confidence=0.0)
        proba = self._model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        return ClassificationResult(step=self._classes[idx], confidence=float(proba[idx]))


__all__ = ["XGBoostStepClassifier"]
