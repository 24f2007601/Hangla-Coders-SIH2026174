"""Person detection. The PoC uses a stub; YOLO fine-tuning is deferred (ADR-0001)."""

from __future__ import annotations

import numpy as np

from bas_assistant.models import BoundingBox


class DummyPersonDetector:
    """Assumes a single person occupying the full frame.

    Sufficient for the vertical slice; replaced by a YOLO fine-tune later.
    """

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        return [BoundingBox(x_min=0, y_min=0, x_max=width - 1, y_max=height - 1)]


__all__ = ["DummyPersonDetector"]
