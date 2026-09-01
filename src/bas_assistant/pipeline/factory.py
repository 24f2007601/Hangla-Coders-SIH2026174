"""Composes pipeline components from `Settings`.

Keeps dependency wiring in one place so scripts and the GUI build the same pipeline.
"""

from __future__ import annotations

from bas_assistant.classification.classifier import DummyClassifier
from bas_assistant.classification.xgboost_classifier import XGBoostStepClassifier
from bas_assistant.config.settings import Settings
from bas_assistant.detection.detector import DummyPersonDetector
from bas_assistant.events.manager import EventManager
from bas_assistant.features.extractor import FEATURE_VECTOR_SIZE, PoseFeatureExtractor
from bas_assistant.pipeline.pipeline import ExperimentPipeline
from bas_assistant.pose.estimation import DummyPoseEstimator, MediaPipePoseEstimator
from bas_assistant.storage.repository import JsonResultRepository
from bas_assistant.tracking.tracker import DummyPersonTracker
from bas_assistant.validation.fsm import ExperimentFSM
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL


def build_pipeline(settings: Settings) -> ExperimentPipeline:
    """Assemble an ExperimentPipeline from typed settings (all real PoC components)."""
    if settings.pose.model == "mediapipe":
        pose_estimator = MediaPipePoseEstimator(
            pose_model_path=settings.pose.pose_model_path,
            hand_model_path=settings.pose.hand_model_path,
            min_detection_confidence=settings.pose.min_detection_confidence,
            min_tracking_confidence=settings.pose.min_tracking_confidence,
            min_hand_detection_confidence=settings.pose.min_hand_detection_confidence,
            min_hand_presence_confidence=settings.pose.min_hand_presence_confidence,
            min_hand_tracking_confidence=settings.pose.min_hand_tracking_confidence,
            hand_hold_seconds=settings.pose.hand_hold_seconds,
            with_hands=settings.pose.with_hands,
        )
    else:
        pose_estimator = DummyPoseEstimator()

    if settings.classifier.model_type == "xgboost":
        classifier = XGBoostStepClassifier(
            model_path=settings.classifier.model_path,
            expected_feature_size=FEATURE_VECTOR_SIZE,
        )
    else:
        classifier = DummyClassifier()

    return ExperimentPipeline(
        settings=settings,
        detector=DummyPersonDetector(),
        tracker=DummyPersonTracker(),
        pose_estimator=pose_estimator,
        feature_extractor=PoseFeatureExtractor(settings.pipeline.sequence_length),
        classifier=classifier,
        validator=ExperimentFSM(DEFAULT_MICROPHONE_PROTOCOL),
        event_manager=EventManager(),
        repository=JsonResultRepository(settings.database.output_dir),
    )


__all__ = ["build_pipeline"]
