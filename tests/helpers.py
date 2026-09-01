"""Shared helpers for pipeline-level tests (scripted classifier, pipeline builder)."""

from __future__ import annotations

from pathlib import Path

from bas_assistant.classification.classifier import DummyClassifier
from bas_assistant.classification.models import BACKGROUND_STEP, ClassificationResult
from bas_assistant.config.settings import Settings
from bas_assistant.detection.detector import DummyPersonDetector
from bas_assistant.events.manager import EventManager
from bas_assistant.features.extractor import PoseFeatureExtractor
from bas_assistant.pipeline.pipeline import ExperimentPipeline
from bas_assistant.pose.estimation import DummyPoseEstimator
from bas_assistant.storage.repository import JsonResultRepository
from bas_assistant.tracking.tracker import DummyPersonTracker
from bas_assistant.validation.fsm import ExperimentFSM
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL


class ScriptedClassifier:
    """Deterministic classifier emitting a scripted step sequence for tests.

    `segments` is a list of (step_id, num_classify_calls); each step is returned for
    `num_classify_calls` consecutive calls so the pipeline's smoothing vote stabilizes.
    After the script is exhausted it reports background.
    """

    def __init__(self, segments: list[tuple[str, int]], confidence: float = 0.9) -> None:
        self._segments = segments
        self._confidence = confidence
        self._segment_idx = 0
        self._calls = 0

    def classify(self, features) -> ClassificationResult:
        if self._segment_idx >= len(self._segments):
            return ClassificationResult(step=BACKGROUND_STEP, confidence=0.0)
        step, hold = self._segments[self._segment_idx]
        self._calls += 1
        if self._calls >= hold:
            self._segment_idx += 1
            self._calls = 0
        return ClassificationResult(step=step, confidence=self._confidence)


def build_test_pipeline(
    output_dir: Path,
    *,
    sequence_length: int = 30,
    hop: int = 5,
    smoothing: int = 5,
    classifier: ScriptedClassifier | None = None,
) -> ExperimentPipeline:
    """Compose a pipeline from dummy components + (optionally) a scripted classifier."""
    settings = Settings()
    settings.pipeline.sequence_length = sequence_length
    settings.pipeline.classify_hop = hop
    settings.pipeline.smoothing_window = smoothing
    return ExperimentPipeline(
        settings=settings,
        detector=DummyPersonDetector(),
        tracker=DummyPersonTracker(),
        pose_estimator=DummyPoseEstimator(motion=0.0),
        feature_extractor=PoseFeatureExtractor(sequence_length),
        classifier=classifier or DummyClassifier(),
        validator=ExperimentFSM(DEFAULT_MICROPHONE_PROTOCOL),
        event_manager=EventManager(),
        repository=JsonResultRepository(output_dir),
    )
