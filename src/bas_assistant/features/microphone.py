"""Microphone-protocol feature extraction from hands + YOLO objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bas_assistant.features.window import FeatureWindow
from bas_assistant.models import PoseResult

OBJECT_NAMES = (
    "microphone",
    "microphone_case_closed",
    "microphone_case_open",
    "phone_screen_on",
    "receiver",
)


@dataclass(slots=True)
class ObjectObservation:
    present: float = 0.0
    confidence: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0


FRAME_FEATURE_NAMES = (
    "left_hand_present",
    "right_hand_present",
    "left_hand_x",
    "left_hand_y",
    "right_hand_x",
    "right_hand_y",
    "hand_distance",
    "left_hand_vx",
    "left_hand_vy",
    "right_hand_vx",
    "right_hand_vy",
    "left_hand_speed",
    "right_hand_speed",
    "microphone_present",
    "microphone_confidence",
    "microphone_x",
    "microphone_y",
    "microphone_w",
    "microphone_h",
    "case_closed_present",
    "case_closed_confidence",
    "case_closed_x",
    "case_closed_y",
    "case_closed_w",
    "case_closed_h",
    "case_open_present",
    "case_open_confidence",
    "case_open_x",
    "case_open_y",
    "case_open_w",
    "case_open_h",
    "phone_present",
    "phone_confidence",
    "phone_x",
    "phone_y",
    "phone_w",
    "phone_h",
    "receiver_present",
    "receiver_confidence",
    "receiver_x",
    "receiver_y",
    "receiver_w",
    "receiver_h",
    "left_to_microphone",
    "right_to_microphone",
    "left_to_case",
    "right_to_case",
    "left_to_receiver",
    "right_to_receiver",
    "left_to_phone",
    "right_to_phone",
)


WINDOW_FEATURE_NAMES = (
    *(f"{name}_mean" for name in FRAME_FEATURE_NAMES),
    *(f"{name}_std" for name in FRAME_FEATURE_NAMES),
)


MICROPHONE_FEATURE_VECTOR_SIZE = len(WINDOW_FEATURE_NAMES)


def _normalise_point(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        float(x / max(width, 1)),
        float(y / max(height, 1)),
    )


def _distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _extract_hand_centers(
    pose: PoseResult,
    width: int,
    height: int,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    hands = pose.metadata.get("hands", [])

    left = None
    right = None

    for hand in hands:
        keypoints = hand.get("keypoints", [])

        if len(keypoints) != 21:
            continue

        # Palm center = average of wrist + MCP joints.
        # Indices: wrist=0, MCPs=5,9,13,17.
        selected = [
            keypoints[0],
            keypoints[5],
            keypoints[9],
            keypoints[13],
            keypoints[17],
        ]

        xs = []
        ys = []

        for point in selected:
            if isinstance(point, dict):
                xs.append(float(point["x"]))
                ys.append(float(point["y"]))
            else:
                xs.append(float(point[0]))
                ys.append(float(point[1]))

        center = _normalise_point(
            float(np.mean(xs)),
            float(np.mean(ys)),
            width,
            height,
        )

        handedness = str(hand.get("handedness", "")).lower()

        if handedness == "left":
            left = center
        elif handedness == "right":
            right = center

    return left, right


def _object_observation(
    object_name: str,
    detections: list[dict],
    width: int,
    height: int,
) -> ObjectObservation:
    candidates = [item for item in detections if item["name"] == object_name]

    if not candidates:
        return ObjectObservation()

    best = max(
        candidates,
        key=lambda item: item["confidence"],
    )

    x1, y1, x2, y2 = best["xyxy"]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return ObjectObservation(
        present=1.0,
        confidence=float(best["confidence"]),
        center_x=float(center_x / max(width, 1)),
        center_y=float(center_y / max(height, 1)),
        width=float((x2 - x1) / max(width, 1)),
        height=float((y2 - y1) / max(height, 1)),
    )


def frame_features(
    pose: PoseResult,
    detections: list[dict],
    width: int,
    height: int,
    previous_hands: (
        tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
        ]
        | None
    ),
) -> tuple[np.ndarray, tuple[tuple[float, float] | None, tuple[float, float] | None]]:

    left_hand, right_hand = _extract_hand_centers(
        pose,
        width,
        height,
    )

    previous_left = None
    previous_right = None

    if previous_hands is not None:
        previous_left, previous_right = previous_hands

    left_x, left_y = left_hand or (0.0, 0.0)
    right_x, right_y = right_hand or (0.0, 0.0)

    if left_hand is not None and previous_left is not None:
        left_vx = left_x - previous_left[0]
        left_vy = left_y - previous_left[1]
    else:
        left_vx = 0.0
        left_vy = 0.0

    if right_hand is not None and previous_right is not None:
        right_vx = right_x - previous_right[0]
        right_vy = right_y - previous_right[1]
    else:
        right_vx = 0.0
        right_vy = 0.0

    left_speed = float(np.hypot(left_vx, left_vy))

    right_speed = float(np.hypot(right_vx, right_vy))

    hand_distance = (
        _distance(left_hand, right_hand)
        if left_hand is not None and right_hand is not None
        else 0.0
    )

    microphone = _object_observation(
        "microphone",
        detections,
        width,
        height,
    )

    case_closed = _object_observation(
        "microphone_case_closed",
        detections,
        width,
        height,
    )

    case_open = _object_observation(
        "microphone_case_open",
        detections,
        width,
        height,
    )

    phone = _object_observation(
        "phone_screen_on",
        detections,
        width,
        height,
    )

    receiver = _object_observation(
        "receiver",
        detections,
        width,
        height,
    )

    microphone_point = (microphone.center_x, microphone.center_y) if microphone.present else None

    case_point = None

    if case_open.present:
        case_point = (
            case_open.center_x,
            case_open.center_y,
        )
    elif case_closed.present:
        case_point = (
            case_closed.center_x,
            case_closed.center_y,
        )

    phone_point = (phone.center_x, phone.center_y) if phone.present else None

    receiver_point = (receiver.center_x, receiver.center_y) if receiver.present else None

    values = [
        float(left_hand is not None),
        float(right_hand is not None),
        left_x,
        left_y,
        right_x,
        right_y,
        hand_distance,
        left_vx,
        left_vy,
        right_vx,
        right_vy,
        left_speed,
        right_speed,
        microphone.present,
        microphone.confidence,
        microphone.center_x,
        microphone.center_y,
        microphone.width,
        microphone.height,
        case_closed.present,
        case_closed.confidence,
        case_closed.center_x,
        case_closed.center_y,
        case_closed.width,
        case_closed.height,
        case_open.present,
        case_open.confidence,
        case_open.center_x,
        case_open.center_y,
        case_open.width,
        case_open.height,
        phone.present,
        phone.confidence,
        phone.center_x,
        phone.center_y,
        phone.width,
        phone.height,
        receiver.present,
        receiver.confidence,
        receiver.center_x,
        receiver.center_y,
        receiver.width,
        receiver.height,
        (
            _distance(left_hand, microphone_point)
            if left_hand is not None and microphone_point is not None
            else 0.0
        ),
        (
            _distance(right_hand, microphone_point)
            if right_hand is not None and microphone_point is not None
            else 0.0
        ),
        (
            _distance(left_hand, case_point)
            if left_hand is not None and case_point is not None
            else 0.0
        ),
        (
            _distance(right_hand, case_point)
            if right_hand is not None and case_point is not None
            else 0.0
        ),
        (
            _distance(left_hand, receiver_point)
            if left_hand is not None and receiver_point is not None
            else 0.0
        ),
        (
            _distance(right_hand, receiver_point)
            if right_hand is not None and receiver_point is not None
            else 0.0
        ),
        (
            _distance(left_hand, phone_point)
            if left_hand is not None and phone_point is not None
            else 0.0
        ),
        (
            _distance(right_hand, phone_point)
            if right_hand is not None and phone_point is not None
            else 0.0
        ),
    ]

    return np.asarray(values, dtype=float), (left_hand, right_hand)


def aggregate_window(
    frames: list[np.ndarray],
) -> np.ndarray:
    if not frames:
        raise ValueError("cannot aggregate empty feature window")

    matrix = np.stack(frames)

    return np.concatenate(
        [
            matrix.mean(axis=0),
            matrix.std(axis=0),
        ]
    )


__all__ = [
    "FRAME_FEATURE_NAMES",
    "WINDOW_FEATURE_NAMES",
    "MICROPHONE_FEATURE_VECTOR_SIZE",
    "frame_features",
    "aggregate_window",
]


class MicrophoneFeatureExtractor:
    """Runtime 102-feature extractor for the microphone protocol.

    YOLO detections are attached to PoseResult.metadata["objects"]
    by the pipeline before this extractor is called.
    """

    def __init__(self, sequence_length: int = 30) -> None:
        self._window = FeatureWindow[np.ndarray](sequence_length)
        self._last_hands = None
        self._ready = False

    @property
    def sequence_length(self) -> int:
        return self._window.size

    @property
    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._window.clear()
        self._last_hands = None
        self._ready = False

    def push(self, pose: PoseResult) -> bool:
        """Add one pose carrying YOLO detections in its metadata."""

        objects = pose.metadata.get(
            "objects",
            [],
        )

        if not isinstance(objects, list):
            objects = []

        width = 1
        height = 1

        if pose.bounding_box is not None:
            width = max(
                pose.bounding_box.x_max,
                1,
            )
            height = max(
                pose.bounding_box.y_max,
                1,
            )

        # Runtime frames normally contain the original frame dimensions
        # in metadata. Fall back safely if unavailable.
        width = int(
            pose.metadata.get(
                "frame_width",
                width,
            )
        )

        height = int(
            pose.metadata.get(
                "frame_height",
                height,
            )
        )

        vector, current_hands = frame_features(
            pose=pose,
            detections=objects,
            width=width,
            height=height,
            previous_hands=self._last_hands,
        )

        self._last_hands = current_hands
        self._window.push(vector)

        self._ready = self._window.is_full

        return self._ready

    def features(self) -> np.ndarray:
        if not self._ready:
            raise ValueError("feature window not full; " "call push() until it returns True")

        return aggregate_window(list(self._window.items()))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return WINDOW_FEATURE_NAMES


__all__.append("MicrophoneFeatureExtractor")
