"""Dummy step classifier so the pipeline runs with no trained model."""

from __future__ import annotations

import numpy as np

from bas_assistant.classification.models import UNKNOWN_STEP, ClassificationResult


class DummyClassifier:
    """Always predicts ("unknown", 0.0) — keeps the pipeline end-to-end runnable."""

    def classify(self, features: np.ndarray) -> ClassificationResult:
        return ClassificationResult(step=UNKNOWN_STEP, confidence=0.0)


__all__ = ["DummyClassifier"]
