"""Classification result types and canonical step labels."""

from __future__ import annotations

from dataclasses import dataclass

# Sentinel labels produced by the step classifier.
BACKGROUND_STEP = "background"  # idle / no-step
UNKNOWN_STEP = "unknown"  # classifier could not decide (dummy or failed model)


@dataclass(slots=True)
class ClassificationResult:
    """A predicted Experiment Step (id) with confidence.

    `step` is a protocol step id, or BACKGROUND_STEP / UNKNOWN_STEP.
    """

    step: str
    confidence: float

    @property
    def is_background(self) -> bool:
        return self.step in (BACKGROUND_STEP, UNKNOWN_STEP)


__all__ = ["BACKGROUND_STEP", "UNKNOWN_STEP", "ClassificationResult"]
