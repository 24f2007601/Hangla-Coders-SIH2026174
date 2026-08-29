from __future__ import annotations

import numpy as np


def palm_center(keypoints: list[dict[str, float]]) -> np.ndarray:
    """Return the palm center from MediaPipe's 21 hand landmarks."""
    indices = (0, 5, 9, 13, 17)

    return np.mean(
        [[keypoints[i]["x"], keypoints[i]["y"]] for i in indices],
        axis=0,
        dtype=float,
    )
