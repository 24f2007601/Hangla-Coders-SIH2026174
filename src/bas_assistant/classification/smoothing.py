"""Majority-vote smoothing over recent classifier outputs to avoid label flicker."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from bas_assistant.classification.models import BACKGROUND_STEP, UNKNOWN_STEP


def majority_vote(
    recent: Sequence[tuple[str, float]],
    min_confidence: float,
    min_share: float = 0.6,
) -> str | None:
    """Return the dominant non-background step if it holds a sufficient share.

    `recent` is a list of (step, confidence) tuples. Background/unknown labels are
    excluded from the vote. Returns None when no step commands a clear majority.
    """
    meaningful = [
        step
        for step, conf in recent
        if step not in (BACKGROUND_STEP, UNKNOWN_STEP) and conf >= min_confidence
    ]
    if not meaningful:
        return None
    counts = Counter(meaningful)
    top, top_count = counts.most_common(1)[0]
    if top_count / len(meaningful) >= min_share:
        return top
    return None


__all__ = ["majority_vote"]
