"""Interactive demo: live annotated video window + step validation (no GUI app).

Opens a webcam (default) or `--source <path>` and displays the frame annotated with
the estimated pose, current step, status, and FPS. Press `q` or ESC to quit. The
session is also written to the JSONL log like `run_pipeline.py`.

If no webcam is available, a clear error is printed (use `--source` with a video file,
or `--source dummy` for a synthetic source).

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --source data/samples/x.mp4
    python scripts/run_demo.py --source dummy --pose dummy --max-frames 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bas_assistant.config.settings import load_settings
from bas_assistant.pipeline.factory import build_pipeline
from bas_assistant.utils.logging import setup_logging
from bas_assistant.utils.visualization import annotate_frame
from bas_assistant.validation.protocol import DEFAULT_TOY_PROTOCOL
from bas_assistant.video.source import DummyVideoSource, OpenCVVideoSource, VideoSourceError

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive BAS demo with a live annotated video window."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a configs/*.yaml file")
    parser.add_argument("--source", default="0", help="Camera index, video file path, or 'dummy'")
    parser.add_argument(
        "--pose", choices=["mediapipe", "dummy"], default=None, help="Pose backend override"
    )
    parser.add_argument(
        "--classifier", choices=["dummy", "xgboost"], default=None, help="Classifier override"
    )
    parser.add_argument(
        "--metrics", action="store_true", help="Enable debug timing instrumentation"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = run until source ends)"
    )
    return parser.parse_args()


def _step_name(step: str | None) -> str:
    if step is None:
        return "idle"
    proto_step = DEFAULT_TOY_PROTOCOL.step(step)
    return proto_step.name if proto_step else step


def main() -> int:
    args = _parse_args()
    setup_logging(logging.INFO)

    settings = load_settings(args.config)
    if args.pose is not None:
        settings.pose.model = args.pose  # type: ignore[assignment]
    if args.classifier is not None:
        settings.classifier.model_type = args.classifier  # type: ignore[assignment]
    if args.metrics:
        settings.pipeline.metrics_enabled = True

    try:
        pipeline = build_pipeline(settings)
    except Exception as exc:  # noqa: BLE001 - surface component init failures clearly
        logger.error("Pipeline build failed: %s", exc)
        return 1

    if args.source == "dummy":
        source: DummyVideoSource = DummyVideoSource(
            width=settings.camera.width, height=settings.camera.height, num_frames=1_000_000
        )
    elif args.source.isdigit():
        source = OpenCVVideoSource(
            int(args.source),
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
            format=settings.camera.format,
            backend=settings.camera.backend,
        )
    else:
        source = OpenCVVideoSource(
            args.source,
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
            format=settings.camera.format,
            backend=settings.camera.backend,
        )

    try:
        source.start()
    except VideoSourceError as exc:
        logger.error("%s", exc)
        logger.error(
            "No usable video source. Connect a webcam, pass --source <path/to/video.mp4>, "
            "or use --source dummy for a synthetic source."
        )
        return 1

    pipeline.start_session()
    frames_processed = 0
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            result = pipeline.process_frame(frame)
            frames_processed += 1
            step_name = _step_name(result.classification.step if result.classification else None)
            confidence = result.classification.confidence if result.classification else 0.0
            status = "error" if result.has_error else "monitoring"
            if result.new_events:
                status = result.new_events[-1].type.replace("step_", "").replace("_", " ")
                for event in result.new_events:
                    print(f"[{time.strftime('%H:%M:%S')}] {event.type:>18} | {event.message}")
            annotated = annotate_frame(
                frame, result.pose, step_name, confidence, result.fps, status
            )
            cv2.imshow("BAS Experiment Assistant - Demo", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.max_frames and frames_processed >= args.max_frames:
                break
    finally:
        pipeline.end_session()
        source.stop()
        cv2.destroyAllWindows()

    logger.info(
        "Demo finished after %d frames; JSONL session log written to data/processed/",
        frames_processed,
    )
    if pipeline.metrics is not None:
        logger.info("Camera diagnostics: %s", source.diagnostics())
        report = pipeline.timing_report()
        if report:
            logger.info("Vision metrics:\n%s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
