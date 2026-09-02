"""Pipeline orchestrator: frame -> detection -> tracking -> pose -> features -> FSM."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from bas_assistant.classification.models import ClassificationResult
from bas_assistant.config.settings import Settings
from bas_assistant.events.models import EVENT_SESSION_STARTED, Event
from bas_assistant.models import PoseResult
from bas_assistant.utils.timing import FPSMeter, LatencyMeter, Metrics, timed
from bas_assistant.validation.led_estimator import LEDStateEstimator
from bas_assistant.validation.protocol_evidence import confirm_step

logger = logging.getLogger(__name__)


# Runtime constants.
# These are intentionally a little more tolerant than the original
# configuration because the recorded demo footage is not perfectly clean.
MIN_STEP_CONFIDENCE = 0.25
STABLE_VOTES_REQUIRED = 2
STABLE_WINDOW = 3


@dataclass(slots=True)
class FrameResult:
    """Everything the pipeline produced for one frame."""

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
    """Composes the pipeline stages for one experiment session."""

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
        object_detector=None,
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
        self._object_detector = object_detector
        self._led_estimator = LEDStateEstimator()

        self._hop = max(1, settings.pipeline.classify_hop)

        self._fps_meter = FPSMeter()
        self._latency_meter = LatencyMeter()
        self._metrics = Metrics() if settings.pipeline.metrics_enabled else None

        self._frame_number = 0
        self._hop_counter = 0

        # Recent classifier outputs used only for stable expected-step voting.
        self._recent_step_votes: deque[str] = deque(maxlen=STABLE_WINDOW)

        # Last step that was actually accepted by the FSM.
        self._emitted_step: str | None = None

        # Verification-gate state.
        self._g1_passed = False
        self._m6_pending = False
        self._m5_pending = False
        self._pending_m5_confidence = 0.0
        self._g2_passed = False
        self._gate_status = "not_required"
        self._last_led_log_frame = -30

        self._current_classification: ClassificationResult | None = None
        self._session_started = False

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

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

        close = getattr(self._pose_estimator, "close", None)

        if callable(close):
            close()

        if self._metrics is not None:
            logger.info(
                "Vision metrics at end of session:\n%s",
                self._metrics.format(),
            )

        logger.info("Ended pipeline session")

    @property
    def metrics(self) -> Metrics | None:
        return self._metrics

    @property
    def gate_status(self) -> str:
        """Current verification-gate status."""
        return self._gate_status

    @property
    def led_state(self) -> dict[str, str]:
        """Latest left/right LED states."""
        return self._led_estimator.state

    def timing_report(self) -> str:
        parts: list[str] = []

        if self._metrics is not None:
            parts.append("pipeline:\n" + self._metrics.format())

        estimator_metrics = getattr(
            self._pose_estimator,
            "metrics",
            None,
        )

        if estimator_metrics is not None:
            parts.append("pose/hands:\n" + estimator_metrics.format())

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        source_timestamp: float | None = None,
    ) -> FrameResult:
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

                if self._metrics is not None:
                    with timed(
                        "pose_estimate",
                        self._metrics,
                    ):
                        pose = self._pose_estimator.estimate(frame)
                else:
                    pose = self._pose_estimator.estimate(frame)

            if pose is not None:
                objects: list[dict] = []

                if self._object_detector is not None:
                    objects = self._object_detector.detect(frame)

                    if getattr(pose, "metadata", None) is None:
                        pose.metadata = {}

                    pose.metadata["objects"] = objects
                    pose.metadata["frame_width"] = frame.shape[1]
                    pose.metadata["frame_height"] = frame.shape[0]

                # Use video-time rather than processing wall-clock time. This keeps
                # blink detection correct even when recorded video is processed
                # slightly faster/slower than real time.
                source_fps = max(1.0, float(getattr(self._settings.camera, "fps", 30.0)))
                led_time = (
                    float(source_timestamp)
                    if source_timestamp is not None
                    else frame_number / source_fps
                )
                led_state = self._led_estimator.update(
                    frame,
                    objects,
                    timestamp=led_time,
                )

                if (
                    self._emitted_step == "M4" or self._m6_pending
                ) and frame_number - self._last_led_log_frame >= 30:
                    self._last_led_log_frame = frame_number
                    logger.info(
                        "LED diagnostic frame=%d state=%s scores=%s ready=%s "
                        "G1=%s G2=%s receiver=%s",
                        frame_number,
                        led_state.get("left"),
                        {
                            "left": led_state.get("left_score"),
                            "right": led_state.get("right_score"),
                        },
                        led_state.get("ready"),
                        led_state.get("g1_passed"),
                        led_state.get("g2_passed"),
                        {
                            "detected": led_state.get("receiver_detected"),
                            "bbox": led_state.get("receiver_bbox"),
                            "confidence": led_state.get("receiver_confidence"),
                        },
                    )
                if getattr(pose, "metadata", None) is None:
                    pose.metadata = {}
                pose.metadata["led_state"] = led_state
                pose.metadata["gate_status"] = self._gate_status

                gate_events = self._process_verification_gates(
                    frame,
                    objects,
                    ts,
                )

                classification, step_events = self._step_pipeline(
                    pose,
                    ts,
                )

                new_events = gate_events + step_events

            self._record_observation(
                frame_number,
                person_id,
                pose,
                classification,
                error,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Frame %d failed: %s: %s",
                frame_number,
                type(exc).__name__,
                exc,
            )

            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started

        latency_ms = self._latency_meter.update(elapsed)
        fps = self._fps_meter.tick()

        if self._metrics is not None:
            self._metrics.record(
                "vision_total",
                elapsed,
            )
            self._metrics.counter("frames_processed")

        result = FrameResult(
            timestamp=ts,
            frame_number=frame_number,
            person_id=person_id,
            pose=pose,
            classification=(classification or self._current_classification),
            new_events=new_events,
            fps=fps,
            inference_latency_ms=latency_ms,
            error=error,
        )

        self._frame_number += 1

        return result

    # ------------------------------------------------------------------
    # Step recognition + validation
    # ------------------------------------------------------------------

    def _step_pipeline(
        self,
        pose: PoseResult,
        ts: float,
    ) -> tuple[
        ClassificationResult | None,
        list[Event],
    ]:
        events: list[Event] = []

        ready = self._feature_extractor.push(pose)
        if not ready:
            return self._current_classification, events

        self._hop_counter += 1
        if self._hop_counter % self._hop != 0:
            return self._current_classification, events

        features = self._feature_extractor.features()
        raw = self._classifier.classify(features)

        expected_index = self._validator.current_index + 1

        if expected_index > 6:
            self._current_classification = raw
            return raw, events

        expected_step = f"M{expected_index}"
        candidate = raw.step

        if self._object_detector is not None:
            objects: list[dict] = []
            if getattr(pose, "metadata", None):
                objects = pose.metadata.get("objects", [])

            confirmed_candidate = confirm_step(
                candidate=raw.step,
                confidence=raw.confidence,
                current_index=self._validator.current_index,
                objects=objects,
            )

            if confirmed_candidate is not None:
                candidate = confirmed_candidate

        self._current_classification = ClassificationResult(
            step=raw.step,
            confidence=raw.confidence,
        )

        if raw.confidence < MIN_STEP_CONFIDENCE:
            self._recent_step_votes.clear()
            return self._current_classification, events

        # G1 is logically between M4 and M5, but the supplied recording shows
        # the receiver LEDs becoming observable only after the receiver is being
        # connected to the phone. Therefore M5 is *observed* while G1 is pending
        # but held from the FSM until G1 passes. This preserves event order while
        # matching the actual video.
        if expected_step == "M5" and not self._g1_passed:
            self._gate_status = "G1_PENDING"
            if candidate != expected_step:
                self._recent_step_votes.clear()
                return self._current_classification, events

            self._recent_step_votes.append(candidate)
            expected_votes = sum(vote == expected_step for vote in self._recent_step_votes)

            if expected_votes >= STABLE_VOTES_REQUIRED:
                self._m5_pending = True
                self._pending_m5_confidence = max(
                    self._pending_m5_confidence,
                    float(raw.confidence),
                )
                self._recent_step_votes.clear()
            return self._current_classification, events

        # M6 is recognized, but cannot be committed until G2 passes.
        # Once M6 is pending, it must never fall through to the generic
        # confirmation block below. The ONLY legal M6 commit path is the
        # successful G2 branch in _process_verification_gates().
        if expected_step == "M6" and self._m6_pending:
            self._recent_step_votes.clear()
            return self._current_classification, events

        if expected_step == "M6" and self._g1_passed and not self._m6_pending:
            if candidate == expected_step:
                self._recent_step_votes.append(candidate)
            else:
                self._recent_step_votes.clear()
                return self._current_classification, events

            expected_votes = sum(vote == expected_step for vote in self._recent_step_votes)

            if expected_votes >= STABLE_VOTES_REQUIRED:
                self._m6_pending = True
                self._led_estimator.reset_history()
                self._gate_status = "G2_PENDING"
                self._recent_step_votes.clear()
                event = Event(
                    timestamp=ts,
                    type="gate_g2_pending",
                    message=(
                        "G2 pending: waiting for exactly one steady and one blinking receiver LED."
                    ),
                )
                self._event_manager.publish(event)
                self._repository.record_event(event)
                events.append(event)

            return self._current_classification, events

        if expected_step == "M5" and self._g1_passed and self._m5_pending:
            events = self._validator.on_step_confirmed(
                "M5",
                ts,
                max(MIN_STEP_CONFIDENCE, self._pending_m5_confidence),
            )
            self._emitted_step = "M5"
            self._m5_pending = False
            self._pending_m5_confidence = 0.0
            self._recent_step_votes.clear()
            for event in events:
                self._event_manager.publish(event)
                self._repository.record_event(event)
            return self._current_classification, events

        if candidate == expected_step:
            self._recent_step_votes.append(candidate)
        else:
            self._recent_step_votes.clear()
            return self._current_classification, events

        expected_votes = sum(vote == expected_step for vote in self._recent_step_votes)

        if expected_votes < STABLE_VOTES_REQUIRED:
            return self._current_classification, events

        if expected_step != self._emitted_step:
            events = self._validator.on_step_confirmed(
                expected_step,
                ts,
                raw.confidence,
            )

            self._emitted_step = expected_step

            for event in events:
                self._event_manager.publish(event)
                self._repository.record_event(event)

            self._recent_step_votes.clear()

            if expected_step == "M4":
                # G1 starts after M4. Do not let pre-M4 LED history contaminate
                # the blinking decision.
                self._led_estimator.reset_history()
                self._gate_status = "G1_PENDING"
                event = Event(
                    timestamp=ts,
                    type="gate_g1_pending",
                    message="G1 pending: waiting for both receiver LEDs to blink.",
                )
                self._event_manager.publish(event)
                self._repository.record_event(event)
                events.append(event)

        return self._current_classification, events

    def _process_verification_gates(
        self,
        frame: np.ndarray,
        objects: list[dict],
        ts: float,
    ) -> list[Event]:
        """Apply G1/G2 only after their preceding step is stable."""
        events: list[Event] = []

        # G1: after M4, require both LEDs to blink.
        if self._emitted_step == "M4" and not self._g1_passed and self._led_estimator.g1_passed:
            self._g1_passed = True
            self._gate_status = "G1_PASSED"

            event = Event(
                timestamp=ts,
                type="gate_g1_passed",
                message="G1 passed: both receiver LEDs are blinking.",
            )
            self._event_manager.publish(event)
            self._repository.record_event(event)
            events.append(event)

        # G2: after M6 has been recognized, require one steady + one blinking.
        if self._m6_pending and not self._g2_passed and self._led_estimator.g2_passed:
            self._g2_passed = True
            self._gate_status = "G2_PASSED"

            gate_event = Event(
                timestamp=ts,
                type="gate_g2_passed",
                message=(
                    "G2 passed: exactly one receiver LED is steady and the other is blinking."
                ),
            )
            self._event_manager.publish(gate_event)
            self._repository.record_event(gate_event)
            events.append(gate_event)

            step_events = self._validator.on_step_confirmed(
                "M6",
                ts,
                1.0,
            )

            self._emitted_step = "M6"
            self._m6_pending = False

            for event in step_events:
                self._event_manager.publish(event)
                self._repository.record_event(event)
            events.extend(step_events)

        return events

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

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

        observation["gate_status"] = self._gate_status
        observation["led_state"] = self._led_estimator.state

        if error is not None:
            observation["error"] = error

        self._repository.record_observation(observation)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_runtime_state(self) -> None:
        self._frame_number = 0
        self._hop_counter = 0

        self._recent_step_votes.clear()

        self._emitted_step = None
        self._current_classification = None

        self._g1_passed = False
        self._m6_pending = False
        self._m5_pending = False
        self._pending_m5_confidence = 0.0
        self._g2_passed = False
        self._gate_status = "not_required"
        self._last_led_log_frame = -30
        self._led_estimator.reset()

        self._feature_extractor.reset()
        self._validator.reset()

        self._fps_meter.reset()
        self._latency_meter.reset()

        if self._metrics is not None:
            self._metrics = Metrics()


__all__ = [
    "ExperimentPipeline",
    "FrameResult",
]
