"""Composes pipeline components from `Settings`.

Keeps dependency wiring in one place so scripts and the GUI build the same pipeline.
"""

from __future__ import annotations

from pathlib import Path

from bas_assistant.classification.classifier import DummyClassifier
from bas_assistant.classification.xgboost_classifier import XGBoostStepClassifier
from bas_assistant.config.settings import Settings
from bas_assistant.detection.detector import DummyPersonDetector
from bas_assistant.detection.object_detector import YOLOMicrophoneDetector
from bas_assistant.events.manager import EventManager
from bas_assistant.features.microphone import (
    MICROPHONE_FEATURE_VECTOR_SIZE,
    MicrophoneFeatureExtractor,
)
from bas_assistant.pipeline.pipeline import ExperimentPipeline
from bas_assistant.pose.estimation import DummyPoseEstimator, MediaPipeHandEstimator
from bas_assistant.storage.repository import JsonResultRepository
from bas_assistant.tracking.tracker import DummyPersonTracker
from bas_assistant.validation.fsm import ExperimentFSM
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL


def build_pipeline(settings: Settings) -> ExperimentPipeline:
    """Assemble an ExperimentPipeline from typed settings (all real PoC components)."""
    if settings.pose.model == "mediapipe":
        pose_estimator = MediaPipeHandEstimator(
            hand_model_path=settings.pose.hand_model_path,
            min_hand_detection_confidence=settings.pose.min_hand_detection_confidence,
            min_hand_presence_confidence=settings.pose.min_hand_presence_confidence,
            min_hand_tracking_confidence=settings.pose.min_hand_tracking_confidence,
            hand_hold_seconds=settings.pose.hand_hold_seconds,
        )
    else:
        pose_estimator = DummyPoseEstimator()

    if settings.classifier.model_type == "xgboost":
        classifier = XGBoostStepClassifier(
            model_path=settings.classifier.model_path,
            expected_feature_size=MICROPHONE_FEATURE_VECTOR_SIZE,
        )
    else:
        classifier = DummyClassifier()

    # Prefer CUDA when available; fall back to CPU so the pipeline still runs
    # on machines without a GPU (device selection only — model/weights unchanged).
    import torch

    yolo_device = 0 if torch.cuda.is_available() else "cpu"

    object_detector = YOLOMicrophoneDetector(
        model_path=Path("runs/detect/runs/microphone_yolo/baseline-2/weights/best.pt"),
        confidence=0.25,
        device=yolo_device,
    )

    return ExperimentPipeline(
        settings=settings,
        detector=DummyPersonDetector(),
        tracker=DummyPersonTracker(),
        pose_estimator=pose_estimator,
        feature_extractor=MicrophoneFeatureExtractor(settings.pipeline.sequence_length),
        classifier=classifier,
        validator=ExperimentFSM(DEFAULT_MICROPHONE_PROTOCOL),
        event_manager=EventManager(),
        repository=JsonResultRepository(settings.database.output_dir),
        object_detector=object_detector,
    )


__all__ = ["build_pipeline"]
