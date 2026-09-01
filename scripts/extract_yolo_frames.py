"""Extract sampled frames from microphone Sessions for YOLO annotation."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

MANIFEST = Path("data/annotations/sessions.csv")
OUTPUT_ROOT = Path("data/yolo_microphone/images")

SAMPLE_EVERY_SECONDS = 0.5


def extract_frames() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Session manifest not found: {MANIFEST}")

    with MANIFEST.open("r", newline="", encoding="utf-8") as file:
        sessions = list(csv.DictReader(file))

    total = 0

    for session in sessions:
        session_id = session["session_id"]
        video_path = Path(session["video_path"])
        split = session["split"]

        # Your CSV contains repo-relative paths, while the videos are
        # currently stored outside the repository.
        video_path = Path(
            r"C:\Users\User\Documents\Project\BAS Assistant\Dataset\videos"
        ) / video_path.name

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid split {split!r} for session {session_id!r}"
            )

        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            continue

        output_dir = OUTPUT_ROOT / split
        output_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"[WARN] Could not open: {video_path}")
            continue

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_interval = max(int(round(fps * SAMPLE_EVERY_SECONDS)), 1)

        frame_index = 0
        saved = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_interval == 0:
                output_path = output_dir / f"{session_id}_f{frame_index:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved += 1

            frame_index += 1

        capture.release()

        print(f"{session_id}: {saved} frames → {split}")
        total += saved

    print(f"\nTotal frames extracted: {total}")


if __name__ == "__main__":
    extract_frames()