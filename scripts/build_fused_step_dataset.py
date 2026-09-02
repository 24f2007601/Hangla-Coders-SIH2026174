"""Build fused hand + YOLO temporal features for M0-M6 XGBoost."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from bas_assistant.features.microphone import (
    MICROPHONE_FEATURE_VECTOR_SIZE,
    aggregate_window,
    frame_features,
)
from bas_assistant.pose.estimation import MediaPipePoseEstimator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION_CSV = PROJECT_ROOT / "data" / "annotations" / "sessions.csv"
STEPS_CSV = PROJECT_ROOT / "data" / "annotations" / "steps.csv"

VIDEO_ROOT = Path(r"C:\Users\User\Documents\Project\BAS Assistant\Dataset\videos")
RESHOOT_ROOT = Path(r"C:\Users\User\Documents\Project\BAS Assistant\Dataset\reshoot")

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
MAX_POSE_GAP = 8

CLASS_NAMES = {
    0: "microphone",
    1: "microphone_case_closed",
    2: "microphone_case_open",
    3: "phone_screen_on",
    4: "receiver",
}


def load_sessions() -> list[dict[str, str]]:
    with SESSION_CSV.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_reshoot_labels() -> dict[str, list[dict[str, int | str]]]:
    labels: dict[str, list[dict[str, int | str]]] = {}

    with STEPS_CSV.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            session_id = row["session_id"]

            labels.setdefault(session_id, []).append(
                {
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "label": row["label"],
                }
            )

    return labels


def reshoot_split(session_id: str) -> str:
    if session_id.endswith("_v4"):
        return "val"
    if session_id.endswith("_v5"):
        return "test"
    return "train"


def predict_objects(model: YOLO, frame) -> list[dict]:
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
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            }
        )

    return detections


def feature_names() -> list[str]:
    from bas_assistant.features.microphone import WINDOW_FEATURE_NAMES

    return list(WINDOW_FEATURE_NAMES)


def process_video_segment(
    *,
    session_id: str,
    split: str,
    label: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    model: YOLO,
    sequence_length: int,
    hop: int,
) -> list[list[object]]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    estimator = MediaPipePoseEstimator(
        min_detection_confidence=POSE_CONFIDENCE,
    )

    window: list = []
    rows: list[list[object]] = []

    previous_hands = None
    last_valid_features = None
    pose_gap = 0
    window_count = 0

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame

        while current_frame <= end_frame:
            ok, frame = cap.read()

            if not ok:
                break

            pose = estimator.estimate(frame)

            # -------------------------------------------------
            # Short MediaPipe gap: reuse previous valid feature.
            # Long gap: reset temporal continuity.
            # -------------------------------------------------
            if pose is None:
                pose_gap += 1

                if last_valid_features is not None and pose_gap <= MAX_POSE_GAP:
                    window.append(last_valid_features)

                    if len(window) >= sequence_length:
                        if window_count % hop == 0:
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
                                    current_frame,
                                    *aggregate.tolist(),
                                ]
                            )

                        window_count += 1

                else:
                    window.clear()
                    previous_hands = None
                    last_valid_features = None
                    window_count = 0

                current_frame += 1
                continue

            # Valid pose recovered.
            pose_gap = 0

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
            last_valid_features = features
            window.append(features)

            if len(window) >= sequence_length:
                if window_count % hop == 0:
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
                            current_frame,
                            *aggregate.tolist(),
                        ]
                    )

                window_count += 1

            current_frame += 1

    finally:
        cap.release()

        close = getattr(estimator, "close", None)

        if callable(close):
            close()

    return rows


def process_old_session(
    session: dict[str, str],
    model: YOLO,
    sequence_length: int,
    hop: int,
) -> list[list[object]]:
    """Process original isolated step recordings."""
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    rows = process_video_segment(
        session_id=session_id,
        split=split,
        label=label,
        video_path=video_path,
        start_frame=0,
        end_frame=total_frames - 1,
        model=model,
        sequence_length=sequence_length,
        hop=hop,
    )

    print(f"{session_id}: label={label}, windows={len(rows)}")

    return rows


def process_reshoot(
    session_id: str,
    segments: list[dict[str, int | str]],
    model: YOLO,
    sequence_length: int,
    hop: int,
) -> tuple[str, list[list[object]]]:
    video_path = RESHOOT_ROOT / f"{session_id}.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Reshoot video missing: {video_path}")

    split = reshoot_split(session_id)
    rows: list[list[object]] = []

    for segment in segments:
        label = str(segment["label"])

        segment_rows = process_video_segment(
            session_id=session_id,
            split=split,
            label=label,
            video_path=video_path,
            start_frame=int(segment["start_frame"]),
            end_frame=int(segment["end_frame"]),
            model=model,
            sequence_length=sequence_length,
            hop=hop,
        )

        rows.extend(segment_rows)

        print(
            f"{session_id}: "
            f"{label} "
            f"frames={segment['start_frame']}-"
            f"{segment['end_frame']} "
            f"windows={len(segment_rows)}"
        )

    return split, rows


def main() -> None:
    parser = argparse.ArgumentParser()

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

    model = YOLO(str(YOLO_MODEL))

    grouped: dict[str, list[list[object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    names = feature_names()

    # Original isolated training clips.
    sessions = load_sessions()

    old_step_sessions = [
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
        and session["split"] == "train"
    ]

    for session in old_step_sessions:
        grouped["train"].extend(
            process_old_session(
                session=session,
                model=model,
                sequence_length=args.sequence_length,
                hop=args.hop,
            )
        )

    # New continuous recordings.
    reshoot_labels = load_reshoot_labels()

    for session_id, segments in sorted(reshoot_labels.items()):
        split, rows = process_reshoot(
            session_id=session_id,
            segments=segments,
            model=model,
            sequence_length=args.sequence_length,
            hop=args.hop,
        )

        grouped[split].extend(rows)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = [
        "session_id",
        "split",
        "label",
        "frame",
        *names,
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
        print(f"Sessions : {len({row[0] for row in rows})}")
        print(f"Windows  : {len(rows)}")
        print(f"Features : {len(names)}")
        print(f"Output   : {output_path}")


if __name__ == "__main__":
    main()
