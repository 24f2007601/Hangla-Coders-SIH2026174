"""Step classification: interface implementations."""

from bas_assistant.classification.classifier import DummyClassifier
from bas_assistant.classification.models import (
    BACKGROUND_STEP,
    UNKNOWN_STEP,
    ClassificationResult,
)
from bas_assistant.classification.smoothing import majority_vote
from bas_assistant.classification.xgboost_classifier import XGBoostStepClassifier

__all__ = [
    "BACKGROUND_STEP",
    "UNKNOWN_STEP",
    "ClassificationResult",
    "DummyClassifier",
    "XGBoostStepClassifier",
    "majority_vote",
]
