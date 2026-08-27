"""Model-independent domain types shared across the pipeline.

`PoseResult` deliberately carries no MediaPipe (or any vendor) types: keypoints are a
plain list of `Keypoint` in a fixed, documented landmark order (MediaPipe 33-point by
default). Downstream stages must never depend on a specific pose backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Keypoint:
    """A single 2D landmark with an optional confidence.

    Coordinates are in pixel space unless a stage explicitly states otherwise.
    """

    x: float
    y: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned person box in pixel coordinates (inclusive)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class TrackedPerson:
    """A person with a stable id across frames, assigned by the tracker."""

    person_id: int
    bbox: BoundingBox


@dataclass(slots=True)
class PoseResult:
    """Standardized pose observation for one person in one frame.

    Attributes:
        timestamp: monotonic/UTC seconds at capture time.
        person_id: id assigned by the tracker.
        keypoints: body landmarks in fixed order (MediaPipe 33-point for the PoC).
        confidence: overall pose confidence (mean of keypoint confidences, or model output).
        bounding_box: person box if known.
        metadata: vendor-specific extras (e.g. hand landmarks) — never parsed by core stages.
    """

    timestamp: float
    person_id: int
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 0.0
    bounding_box: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["keypoints"] = [k.to_dict() for k in self.keypoints]
        return data
