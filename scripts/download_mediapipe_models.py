"""Download the MediaPipe Tasks API model files needed by the real pose estimator.

The MediaPipe Tasks API (used since the legacy `mp.solutions` API was removed in
mediapipe 1.0) reads `.task` bundles from disk. This script fetches the pose and
hand landmarker models into `models/` (gitignored).

Usage:
    python scripts/download_mediapipe_models.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"), help="Where to write the .task files"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dest = args.output_dir / name
        print(f"Downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed allowlist of URLs
        except Exception as exc:  # noqa: BLE001 - surface any download failure clearly
            print(f"FAILED to download {name}: {exc}", file=sys.stderr)
            return 1
        print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")
    print("Done. Run the demo with pose.model=mediapipe (the default).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
