"""Build a session-aware temporal feature dataset for XGBoost."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from bas_assistant.features.extractor import PoseFeatureExtractor
from bas_assistant.models import PoseResult
from bas_assistant.pose.estimation import MediaPipePoseEstimator

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION_CSV = PROJECT_ROOT / "data" / "annotations" / "sessions.csv"

VIDEO_ROOT = Path(r"C:\Users\User\Documents\Project\BAS Assistant\Dataset\videos")

OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"

POSE_CONFIDENCE = 0.4

# Only recover short gaps.
MAX_GAP_FRAMES = 5


def load_sessions() -> list[dict[str, str]]:
    with SESSION_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def interpolate_pose(
    previous: PoseResult,
    following: PoseResult,
    alpha: float,
) -> PoseResult:
    """Linearly interpolate a short missing pose gap."""

    keypoints = []

    for prev, next_ in zip(
        previous.keypoints,
        following.keypoints,
        strict=True,
    ):
        keypoints.append(
            type(prev)(
                x=prev.x + alpha * (next_.x - prev.x),
                y=prev.y + alpha * (next_.y - prev.y),
                confidence=min(prev.confidence, next_.confidence),
            )
        )

    bbox = None

    if previous.bounding_box and following.bounding_box:
        bbox = type(previous.bounding_box)(
            x_min=round(
                previous.bounding_box.x_min
                + alpha * (following.bounding_box.x_min - previous.bounding_box.x_min)
            ),
            y_min=round(
                previous.bounding_box.y_min
                + alpha * (following.bounding_box.y_min - previous.bounding_box.y_min)
            ),
            x_max=round(
                previous.bounding_box.x_max
                + alpha * (following.bounding_box.x_max - previous.bounding_box.x_max)
            ),
            y_max=round(
                previous.bounding_box.y_max
                + alpha * (following.bounding_box.y_max - previous.bounding_box.y_max)
            ),
        )

    return PoseResult(
        timestamp=previous.timestamp + alpha * (following.timestamp - previous.timestamp),
        person_id=previous.person_id,
        keypoints=keypoints,
        confidence=previous.confidence + alpha * (following.confidence - previous.confidence),
        bounding_box=bbox,
        metadata=previous.metadata,
    )


def collect_poses(
    video_path: Path,
    estimator: MediaPipePoseEstimator,
) -> tuple[list[PoseResult | None], int]:
    """Read every frame and collect the corresponding pose result."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    poses: list[PoseResult | None] = []
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_count += 1
            poses.append(estimator.estimate(frame))

    finally:
        cap.release()

    return poses, frame_count


def recover_short_gaps(
    poses: list[PoseResult | None],
    fps: float,
) -> tuple[list[PoseResult | None], int]:
    """Interpolate only short gaps surrounded by valid poses."""

    recovered = 0
    output = list(poses)

    index = 0

    while index < len(output):

        if output[index] is not None:
            index += 1
            continue

        gap_start = index

        while index < len(output) and output[index] is None:
            index += 1

        gap_end = index - 1
        gap_length = gap_end - gap_start + 1

        previous_index = gap_start - 1
        following_index = gap_end + 1

        if (
            gap_length <= MAX_GAP_FRAMES
            and previous_index >= 0
            and following_index < len(output)
            and output[previous_index] is not None
            and output[following_index] is not None
        ):
            previous = output[previous_index]
            following = output[following_index]

            assert previous is not None
            assert following is not None

            for gap_index in range(
                gap_start,
                gap_end + 1,
            ):
                # Fraction between surrounding valid frames.
                alpha = (gap_index - previous_index) / (following_index - previous_index)

                # Preserve real frame timing.
                interpolated = interpolate_pose(
                    previous,
                    following,
                    alpha,
                )

                interpolated.timestamp = gap_index / fps if fps > 0 else interpolated.timestamp

                output[gap_index] = interpolated
                recovered += 1

    return output, recovered


def extract_features(
    poses: list[PoseResult | None],
    sequence_length: int,
    hop: int,
) -> tuple[list[str], list[list[float]], int]:

    extractor = PoseFeatureExtractor(sequence_length=sequence_length)

    feature_names = list(extractor.feature_names)

    rows: list[list[float]] = []
    window_count = 0

    for pose in poses:

        if pose is None:
            # A long unrecoverable gap resets the temporal extractor.
            extractor.reset()
            continue

        ready = extractor.push(pose)

        if not ready:
            continue

        window_count += 1

        if (window_count - 1) % hop != 0:
            continue

        features = extractor.features()

        rows.append(features.tolist())

    return feature_names, rows, window_count


def process_session(
    session: dict[str, str],
    sequence_length: int,
    hop: int,
) -> tuple[list[str], list[list[object]], dict[str, int]]:

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

    estimator = MediaPipePoseEstimator(
        min_detection_confidence=POSE_CONFIDENCE,
    )

    try:
        poses, frame_count = collect_poses(
            video_path,
            estimator,
        )

        valid_before = sum(pose is not None for pose in poses)

        fps = 30.0

        cap = cv2.VideoCapture(str(video_path))

        if cap.isOpened():
            detected_fps = cap.get(cv2.CAP_PROP_FPS)

            if detected_fps > 0:
                fps = detected_fps

        cap.release()

        poses, recovered = recover_short_gaps(
            poses,
            fps,
        )

        valid_after = sum(pose is not None for pose in poses)

        feature_names, features, _ = extract_features(
            poses,
            sequence_length,
            hop,
        )

    finally:
        close = getattr(
            estimator,
            "close",
            None,
        )

        if callable(close):
            close()

    rows: list[list[object]] = []

    for index, feature_vector in enumerate(features):
        rows.append(
            [
                session_id,
                split,
                label,
                index,
                *feature_vector,
            ]
        )

    stats = {
        "frames": frame_count,
        "valid_before": valid_before,
        "recovered": recovered,
        "valid_after": valid_after,
        "windows": len(rows),
    }

    print(
        f"{session_id}: "
        f"frames={frame_count}, "
        f"valid={valid_before}, "
        f"recovered={recovered}, "
        f"windows={len(rows)}"
    )

    return feature_names, rows, stats


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("Build session-aware temporal features " "for M0-M6 XGBoost training.")
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

    if args.sequence_length <= 0:
        raise ValueError("sequence-length must be greater than 0")

    if args.hop <= 0:
        raise ValueError("hop must be greater than 0")

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

    grouped: dict[
        str,
        list[list[object]],
    ] = {
        "train": [],
        "val": [],
        "test": [],
    }

    feature_names: list[str] | None = None

    total_stats = {
        "train": {"sessions": 0, "windows": 0},
        "val": {"sessions": 0, "windows": 0},
        "test": {"sessions": 0, "windows": 0},
    }

    for session in step_sessions:

        split = session["split"]

        names, rows, stats = process_session(
            session,
            args.sequence_length,
            args.hop,
        )

        if feature_names is None:
            feature_names = names

        elif feature_names != names:
            raise RuntimeError("Feature schema changed between sessions")

        grouped[split].extend(rows)

        total_stats[split]["sessions"] += 1
        total_stats[split]["windows"] += len(rows)

    assert feature_names is not None

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = [
        "session_id",
        "split",
        "label",
        "window_id",
        *feature_names,
    ]

    for split, rows in grouped.items():

        output_path = OUTPUT_ROOT / f"step_features_{split}.csv"

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
        print(f"Sessions : " f"{total_stats[split]['sessions']}")
        print(f"Windows  : " f"{total_stats[split]['windows']}")
        print(f"Output   : {output_path}")


if __name__ == "__main__":
    main()
