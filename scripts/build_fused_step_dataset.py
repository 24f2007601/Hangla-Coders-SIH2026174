"""Build fused hand + YOLO temporal features for M0-M6 XGBoost."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from bas_assistant.features.microphone import (
    FRAME_FEATURE_NAMES,
    MICROPHONE_FEATURE_VECTOR_SIZE,
    aggregate_window,
    frame_features,
)
from bas_assistant.pose.estimation import MediaPipePoseEstimator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION_CSV = PROJECT_ROOT / "data" / "annotations" / "sessions.csv"

VIDEO_ROOT = Path(r"C:\Users\User\Documents\Project\BAS Assistant\Dataset\videos")

YOLO_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "microphone_yolo"
    / "baseline-2"
    / "weights"
    / "best.pt"
)

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"

YOLO_CONFIDENCE = 0.25
POSE_CONFIDENCE = 0.3

CLASS_NAMES = {
    0: "microphone",
    1: "microphone_case_closed",
    2: "microphone_case_open",
    3: "phone_screen_on",
    4: "receiver",
}


def load_sessions() -> list[dict[str, str]]:
    with SESSION_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def predict_objects(
    model: YOLO,
    frame,
) -> list[dict]:
    """Return normalized YOLO detections in a model-independent format."""

    result = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        device=0,
        verbose=False,
    )[0]

    detections: list[dict] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())

        if class_id not in CLASS_NAMES:
            continue

        detections.append(
            {
                "class_id": class_id,
                "name": CLASS_NAMES[class_id],
                "confidence": float(box.conf.item()),
                "xyxy": [float(value) for value in box.xyxy[0].tolist()],
            }
        )

    return detections


def process_session(
    session: dict[str, str],
    model: YOLO,
    sequence_length: int,
    hop: int,
) -> tuple[list[str], list[list[object]]]:

    session_id = session["session_id"]
    split = session["split"]
    scenario = session["scenario"]

    if not scenario.startswith("step_"):
        raise ValueError(f"Unexpected scenario for {session_id}: {scenario}")

    label = scenario.removeprefix("step_")

    video_name = Path(session["video_path"]).name

    video_path = VIDEO_ROOT / video_name

    if not video_path.exists():
        raise FileNotFoundError(f"Video missing for {session_id}: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    estimator = MediaPipePoseEstimator(
        min_detection_confidence=POSE_CONFIDENCE,
    )

    window: list = []
    rows: list[list[object]] = []

    frame_count = 0
    pose_frames = 0
    fused_frames = 0
    window_count = 0

    previous_hands = None

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_count += 1

            pose = estimator.estimate(frame)

            if pose is None:
                # A missing pose breaks temporal continuity.
                window.clear()
                previous_hands = None
                continue

            pose_frames += 1

            height, width = frame.shape[:2]

            detections = predict_objects(
                model,
                frame,
            )

            features, current_hands = frame_features(
                pose=pose,
                detections=detections,
                width=width,
                height=height,
                previous_hands=previous_hands,
            )

            previous_hands = current_hands

            window.append(features)
            fused_frames += 1

            if len(window) < sequence_length:
                continue

            window_count += 1

            if (window_count - 1) % hop == 0:

                aggregate = aggregate_window(window[-sequence_length:])

                if aggregate.shape[0] != MICROPHONE_FEATURE_VECTOR_SIZE:
                    raise RuntimeError(
                        "Unexpected microphone feature size: "
                        f"{aggregate.shape[0]} "
                        f"(expected "
                        f"{MICROPHONE_FEATURE_VECTOR_SIZE})"
                    )

                rows.append(
                    [
                        session_id,
                        split,
                        label,
                        frame_count - 1,
                        *aggregate.tolist(),
                    ]
                )

    finally:
        cap.release()

        close = getattr(
            estimator,
            "close",
            None,
        )

        if callable(close):
            close()

    feature_names = [f"{name}_mean" for name in FRAME_FEATURE_NAMES] + [
        f"{name}_std" for name in FRAME_FEATURE_NAMES
    ]

    print(
        f"{session_id}: "
        f"frames={frame_count}, "
        f"pose={pose_frames}, "
        f"fused={fused_frames}, "
        f"windows={len(rows)}"
    )

    return feature_names, rows


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("Build fused hand + YOLO temporal " "features for XGBoost.")
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--hop",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if not YOLO_MODEL.exists():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL}")

    sessions = load_sessions()

    step_sessions = [
        session
        for session in sessions
        if session["session_id"].startswith(
            (
                "M0_",
                "M1_",
                "M2_",
                "M3_",
                "M4_",
                "M5_",
                "M6_",
            )
        )
    ]

    if len(step_sessions) != 35:
        raise RuntimeError(f"Expected 35 step sessions, found {len(step_sessions)}")

    model = YOLO(str(YOLO_MODEL))

    grouped: dict[str, list[list[object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    feature_names: list[str] | None = None

    for session in step_sessions:

        names, rows = process_session(
            session=session,
            model=model,
            sequence_length=args.sequence_length,
            hop=args.hop,
        )

        if feature_names is None:
            feature_names = names

        elif feature_names != names:
            raise RuntimeError("Feature schema changed between sessions")

        grouped[session["split"]].extend(rows)

    assert feature_names is not None

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = [
        "session_id",
        "split",
        "label",
        "frame",
        *feature_names,
    ]

    for split, rows in grouped.items():

        output_path = OUTPUT_ROOT / f"fused_step_features_{split}.csv"

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

        print()
        print("=" * 60)
        print(split.upper())
        print("=" * 60)
        print(f"Sessions : " f"{len({row[0] for row in rows})}")
        print(f"Windows  : {len(rows)}")
        print(f"Features : " f"{len(feature_names)}")
        print(f"Output   : {output_path}")


if __name__ == "__main__":
    main()
