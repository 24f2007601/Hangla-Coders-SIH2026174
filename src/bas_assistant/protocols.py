"""Abstract component contracts (PEP 544 Protocols).

Every replaceable pipeline stage is defined here. Implementations live in their own
modules and can be swapped without touching the pipeline orchestrator.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bas_assistant.classification.models import ClassificationResult
from bas_assistant.events.models import Event
from bas_assistant.models import BoundingBox, PoseResult, TrackedPerson


@runtime_checkable
class VideoSource(Protocol):
    """A frame producer (webcam index, video file, or synthetic source)."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def fps(self) -> float: ...

    def start(self) -> None: ...

    def read(self) -> np.ndarray | None: ...

    def stop(self) -> None: ...


@runtime_checkable
class PersonDetector(Protocol):
    """Detects persons in a frame; may be a stub for the PoC."""

    def detect(self, frame: np.ndarray) -> list[BoundingBox]: ...


@runtime_checkable
class PersonTracker(Protocol):
    """Assigns stable ids to detections; may be a stub for the PoC."""

    def update(self, detections: list[BoundingBox]) -> list[TrackedPerson]: ...


@runtime_checkable
class PoseEstimator(Protocol):
    """Estimates a standardized pose for a person in a frame."""

    def estimate(self, frame: np.ndarray) -> PoseResult | None: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Accumulates normalized poses into a temporal window and produces feature vectors."""

    def reset(self) -> None: ...

    def push(self, pose: PoseResult) -> bool:
        """Add one pose; returns True once a full window is available."""

    def features(self) -> np.ndarray: ...


@runtime_checkable
class ActivityClassifier(Protocol):
    """Maps a feature vector to a predicted Experiment Step (or background)."""

    def classify(self, features: np.ndarray) -> ClassificationResult: ...


@runtime_checkable
class ResultRepository(Protocol):
    """Persists observations and events for one session."""

    def start_session(self, session_id: str | None = None) -> str: ...

    def record_observation(self, observation: dict) -> None: ...

    def record_event(self, event: Event) -> None: ...

    def end_session(self, summary: dict | None = None) -> None: ...


__all__ = [
    "ActivityClassifier",
    "FeatureExtractor",
    "PersonDetector",
    "PersonTracker",
    "PoseEstimator",
    "ResultRepository",
    "VideoSource",
]
