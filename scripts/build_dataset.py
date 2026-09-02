"""Build a labelled feature dataset from an experiment video."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from bas_assistant.features.extractor import PoseFeatureExtractor
from bas_assistant.pose.estimation import MediaPipeHandEstimator


def build_dataset(
    video_path: Path,
    output_path: Path,
    label: str,
    sequence_length: int = 30,
    hop: int = 1,
) -> int:
    """Extract sliding-window feature vectors from a labelled video."""

    estimator = MediaPipeHandEstimator()
    extractor = PoseFeatureExtractor(sequence_length=sequence_length)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = extractor.feature_names
    rows: list[list[float | str]] = []
    frame_count = 0
    window_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            pose = estimator.estimate(frame)

            if pose is None:
                continue

            ready = extractor.push(pose)

            if ready:
                window_count += 1

                if (window_count - 1) % hop == 0:
                    features = extractor.features()
                    rows.append([*features.tolist(), label])
    finally:
        cap.release()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([*feature_names, "label"])
        writer.writerows(rows)

    print(f"Processed frames: {frame_count}")
    print(f"Feature vectors: {len(rows)}")
    print(f"Output: {output_path}")

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a labelled feature dataset from an experiment video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features.csv"),
    )
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hop", type=int, default=1)

    args = parser.parse_args()

    if args.sequence_length <= 0:
        raise ValueError("sequence-length must be greater than 0")

    if args.hop <= 0:
        raise ValueError("hop must be greater than 0")

    build_dataset(
        video_path=args.video,
        output_path=args.output,
        label=args.label,
        sequence_length=args.sequence_length,
        hop=args.hop,
    )


if __name__ == "__main__":
    main()
