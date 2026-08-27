"""Unit tests for the step classifier (dummy fallback, XGBoost wrapper, smoothing)."""

from __future__ import annotations

import numpy as np

from bas_assistant.classification.classifier import DummyClassifier
from bas_assistant.classification.models import BACKGROUND_STEP, UNKNOWN_STEP, ClassificationResult
from bas_assistant.classification.smoothing import majority_vote
from bas_assistant.classification.xgboost_classifier import XGBoostStepClassifier


def test_dummy_classifier_returns_unknown() -> None:
    result = DummyClassifier().classify(np.zeros(34))
    assert result.step == UNKNOWN_STEP
    assert result.confidence == 0.0
    assert result.is_background


def test_majority_vote_empty() -> None:
    assert majority_vote([], min_confidence=0.5) is None


def test_majority_vote_ignores_background_and_unknown() -> None:
    recent = [(BACKGROUND_STEP, 0.9), (UNKNOWN_STEP, 0.0), (BACKGROUND_STEP, 0.8)]
    assert majority_vote(recent, min_confidence=0.5) is None


def test_majority_vote_returns_clear_winner() -> None:
    recent = [("S1", 0.9), ("S1", 0.8), ("S1", 0.7), ("S2", 0.9), ("S2", 0.8)]
    assert majority_vote(recent, min_confidence=0.5) == "S1"


def test_majority_vote_no_majority() -> None:
    recent = [("S1", 0.9), ("S1", 0.8), ("S2", 0.9), ("S2", 0.8), ("S3", 0.7)]
    assert majority_vote(recent, min_confidence=0.5) is None


def test_majority_vote_applies_min_confidence() -> None:
    recent = [("S1", 0.4), ("S1", 0.4), ("S1", 0.4), ("S1", 0.4), ("S1", 0.4)]
    assert majority_vote(recent, min_confidence=0.5) is None


def test_xgboost_missing_model_falls_back(tmp_path) -> None:
    classifier = XGBoostStepClassifier(tmp_path / "missing_model.json", expected_feature_size=34)
    result = classifier.classify(np.zeros(34))
    assert result.step == UNKNOWN_STEP
    assert result.confidence == 0.0


def test_xgboost_feature_size_mismatch_returns_unknown(tmp_path) -> None:
    classifier = XGBoostStepClassifier(tmp_path / "missing_model.json", expected_feature_size=34)
    result = classifier.classify(np.zeros(10))
    assert result.step == UNKNOWN_STEP


def test_classification_result_helpers() -> None:
    assert ClassificationResult(step="S1", confidence=0.8).is_background is False
    assert ClassificationResult(step=BACKGROUND_STEP, confidence=0.9).is_background is True
    assert ClassificationResult(step=UNKNOWN_STEP, confidence=0.0).is_background is True
