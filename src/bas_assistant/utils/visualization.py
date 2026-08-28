"""OpenCV drawing helpers for the demo/GUI overlay (pure presentation)."""

from __future__ import annotations

import cv2
import numpy as np

from bas_assistant.models import PoseResult
from bas_assistant.pose.landmarks import POSE_CONNECTIONS

_STEP_COLORS = {
    "confirmed": (80, 200, 80),
    "out_of_sequence": (0, 80, 255),
    "skipped": (0, 165, 255),
}

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

def draw_pose(
    frame: np.ndarray, pose: PoseResult, color: tuple[int, int, int] = (80, 200, 80)
) -> None:
    """Draw the 33-point stick figure (mutates a copy provided by the caller)."""
    if pose is None or not pose.keypoints:
        return
    pts = np.array([[k.x, k.y] for k in pose.keypoints], dtype=int)
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    draw_hands(frame, pose)

def draw_hands(
    frame: np.ndarray,
    pose: PoseResult,
    color: tuple[int, int, int] = (255, 180, 0),
) -> None:
    """Draw MediaPipe Hands landmarks stored in PoseResult metadata."""
    if pose is None or not pose.metadata:
        return

    hands = pose.metadata.get("hands")
    if not hands:
        return

    for hand in hands:
        keypoints = hand.get("keypoints", [])

        if len(keypoints) < 21:
            continue

        pts = np.array(
            [[point["x"], point["y"]] for point in keypoints],
            dtype=int,
        )

        for a, b in _HAND_CONNECTIONS:
            cv2.line(
                frame,
                tuple(pts[a]),
                tuple(pts[b]),
                color,
                2,
            )

        for x, y in pts:
            cv2.circle(
                frame,
                (int(x), int(y)),
                3,
                color,
                -1,
            )

def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int] = (10, 30),
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    pose: PoseResult | None,
    step_label: str,
    confidence: float,
    fps: float,
    status: str = "monitoring",
) -> np.ndarray:
    """Return a new frame annotated with pose, current step, and FPS."""
    out = frame.copy()
    if pose is not None:
        draw_pose(out, pose)
        draw_hands(out, pose)
    color = _STEP_COLORS.get(status, (255, 255, 255))
    draw_label(out, f"Step: {step_label} ({confidence:.2f})", (10, 30), color)
    draw_label(out, f"Status: {status}", (10, 60))
    draw_label(out, f"FPS: {fps:.1f}", (10, 90))
    return out


__all__ = ["annotate_frame", "draw_hands", "draw_label", "draw_pose"]
