"""Pipeline orchestrator: frame -> detection -> tracking -> pose -> features -> FSM.

The pipeline is deliberately decoupled from concrete implementations via the
protocols in `bas_assistant.protocols`. It tolerates a single failing frame
(catches, logs, and continues) but never silently swallows errors.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from bas_assistant.classification.models import BACKGROUND_STEP, ClassificationResult
from bas_assistant.classification.smoothing import majority_vote
from bas_assistant.config.settings import Settings
from bas_assistant.events.models import EVENT_SESSION_STARTED, Event
from bas_assistant.models import PoseResult
from bas_assistant.utils.timing import FPSMeter, LatencyMeter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameResult:
    """Everything the pipeline produced for one frame (read by the UI layer)."""

    timestamp: float
    frame_number: int
    person_id: int | None
    pose: PoseResult | None
    classification: ClassificationResult | None
    new_events: list[Event]
    fps: float
    inference_latency_ms: float
    error: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


class ExperimentPipeline:
    """Composes the pipeline stages. One pipeline serves one session/astronaut."""

    def __init__(
        self,
        settings: Settings,
        detector,
        tracker,
        pose_estimator,
        feature_extractor,
        classifier,
        validator,
        event_manager,
        repository,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._tracker = tracker
        self._pose_estimator = pose_estimator
        self._feature_extractor = feature_extractor
        self._classifier = classifier
        self._validator = validator
        self._event_manager = event_manager
        self._repository = repository

        self._hop = settings.pipeline.classify_hop
        self._smoothing_window = settings.pipeline.smoothing_window
        self._confidence_threshold = settings.classifier.confidence_threshold

        self._fps_meter = FPSMeter()
        self._latency_meter = LatencyMeter()
        self._frame_number = 0
        self._hop_counter = 0
        self._recent_labels: deque[tuple[str, float]] = deque(maxlen=self._smoothing_window)
        self._emitted_step: str | None = None
        self._current_classification: ClassificationResult | None = None
        self._session_started = False

    # -- Session lifecycle --------------------------------------------------

    def start_session(self) -> str:
        self._reset_runtime_state()
        session_id = self._repository.start_session()
        event = Event(
            timestamp=time.time(),
            type=EVENT_SESSION_STARTED,
            message=f"Session {session_id} started",
        )
        self._event_manager.publish(event)
        self._repository.record_event(event)
        self._session_started = True
        logger.info("Started pipeline session %s", session_id)
        return session_id

    def end_session(self) -> None:
        if not self._session_started:
            return
        self._repository.end_session(
            {
                "steps_completed": self._validator.done_steps,
                "frames_processed": self._frame_number,
            }
        )
        self._session_started = False
        logger.info("Ended pipeline session")

    # -- Frame processing ---------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        ts = time.time()
        frame_number = self._frame_number
        started = time.perf_counter()
        pose: PoseResult | None = None
        person_id: int | None = None
        classification: ClassificationResult | None = None
        new_events: list[Event] = []
        error: str | None = None

        try:
            detections = self._detector.detect(frame)
            tracked = self._tracker.update(detections)
            if tracked:
                person_id = tracked[0].person_id
                pose = self._pose_estimator.estimate(frame)

            if pose is not None:
                classification, new_events = self._step_pipeline(pose, ts)

            self._record_observation(frame_number, person_id, pose, classification, error)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the pipeline
            logger.error("Frame %d failed: %s: %s", frame_number, type(exc).__name__, exc)
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = self._latency_meter.update(time.perf_counter() - started)
        fps = self._fps_meter.tick()
        result = FrameResult(
            timestamp=ts,
            frame_number=frame_number,
            person_id=person_id,
            pose=pose,
            classification=classification or self._current_classification,
            new_events=new_events,
            fps=fps,
            inference_latency_ms=latency_ms,
            error=error,
        )
        self._frame_number += 1
        return result

    # -- Step recognition + validation --------------------------------------

    def _step_pipeline(
        self, pose: PoseResult, ts: float
    ) -> tuple[ClassificationResult | None, list[Event]]:
        events: list[Event] = []
        ready = self._feature_extractor.push(pose)
        if not ready:
            return self._current_classification, events

        self._hop_counter += 1
        if self._hop_counter % self._hop != 0:
            return self._current_classification, events

        features = self._feature_extractor.features()
        raw = self._classifier.classify(features)
        self._recent_labels.append((raw.step, raw.confidence))

        smoothed = majority_vote(list(self._recent_labels), self._confidence_threshold)
        if smoothed is None:
            self._current_classification = ClassificationResult(
                step=BACKGROUND_STEP, confidence=raw.confidence
            )
            return self._current_classification, events

        classification = ClassificationResult(step=smoothed, confidence=raw.confidence)
        if smoothed != self._emitted_step:
            events = self._validator.on_step_confirmed(smoothed, ts, raw.confidence)
            self._emitted_step = smoothed
            for event in events:
                self._event_manager.publish(event)
                self._repository.record_event(event)
        self._current_classification = classification
        return classification, events

    # -- Helpers ------------------------------------------------------------

    def _record_observation(
        self,
        frame_number: int,
        person_id: int | None,
        pose: PoseResult | None,
        classification: ClassificationResult | None,
        error: str | None,
    ) -> None:
        observation: dict[str, Any] = {
            "frame": frame_number,
            "timestamp": time.time(),
            "person_id": person_id,
        }
        if pose is not None:
            observation["pose"] = pose.to_dict()
        if classification is not None:
            observation["step"] = classification.step
            observation["confidence"] = classification.confidence
        if error is not None:
            observation["error"] = error
        self._repository.record_observation(observation)

    def _reset_runtime_state(self) -> None:
        self._frame_number = 0
        self._hop_counter = 0
        self._recent_labels.clear()
        self._emitted_step = None
        self._current_classification = None
        self._feature_extractor.reset()
        self._validator.reset()
        self._fps_meter.reset()
        self._latency_meter.reset()


__all__ = ["ExperimentPipeline", "FrameResult"]
