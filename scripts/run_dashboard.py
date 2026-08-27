"""Launch the PySide6 monitoring dashboard (requires the 'ui' extra).

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bas_assistant.config.settings import load_settings
from bas_assistant.pipeline.factory import build_pipeline
from bas_assistant.utils.logging import setup_logging
from bas_assistant.video.source import OpenCVVideoSource


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the BAS PySide6 dashboard.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a configs/*.yaml file")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument(
        "--pose", choices=["mediapipe", "dummy"], default=None, help="Pose backend override"
    )
    parser.add_argument(
        "--classifier", choices=["dummy", "xgboost"], default=None, help="Classifier override"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(logging.INFO)

    settings = load_settings(args.config)
    if args.pose is not None:
        settings.pose.model = args.pose  # type: ignore[assignment]
    if args.classifier is not None:
        settings.classifier.model_type = args.classifier  # type: ignore[assignment]

    source = OpenCVVideoSource(
        int(args.source) if args.source.isdigit() else args.source,
        width=settings.video.width,
        height=settings.video.height,
    )

    try:
        pipeline = build_pipeline(settings)
    except Exception as exc:  # noqa: BLE001 - surface component init failures clearly
        logging.getLogger(__name__).error("Pipeline build failed: %s", exc)
        return 1

    from bas_assistant.ui.dashboard import run_dashboard

    return run_dashboard(pipeline, source, max_fps=settings.pipeline.max_fps)


if __name__ == "__main__":
    sys.exit(main())
