"""Headless pipeline runner: observe -> validate -> record with no GUI.

Processes a camera, a video file, or a synthetic dummy source end-to-end
(detection -> tracking -> pose -> features -> step classification -> FSM validation)
and writes the structured JSONL session log to `--out-dir`. Confirmed/validation
events are printed to stdout as they occur.

Usage:
    python scripts/run_pipeline.py --source 0                  # webcam
    python scripts/run_pipeline.py --source data/samples/x.mp4 # video file
    python scripts/run_pipeline.py --source dummy --max-frames 300
    python scripts/run_pipeline.py --pose dummy --classifier dummy
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bas_assistant.config.settings import load_settings
from bas_assistant.pipeline.factory import build_pipeline
from bas_assistant.utils.logging import setup_logging
from bas_assistant.video.source import DummyVideoSource, OpenCVVideoSource, VideoSourceError

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BAS pipeline headless and record a session."
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
        "--out-dir", type=Path, default=None, help="Where to write the JSONL session log"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = run until source ends)"
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parser.add_argument(
        "--metrics", action="store_true", help="Enable debug timing instrumentation"
    )
    return parser.parse_args()


def _make_source(source: str, settings) -> OpenCVVideoSource | DummyVideoSource:
    if source == "dummy":
        # Effectively unlimited synthetic frames; --max-frames caps the run.
        return DummyVideoSource(
            width=settings.camera.width, height=settings.camera.height, num_frames=1_000_000
        )
    if source.isdigit():
        return OpenCVVideoSource(
            int(source),
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
            format=settings.camera.format,
            backend=settings.camera.backend,
        )
    return OpenCVVideoSource(
        source,
        width=settings.camera.width,
        height=settings.camera.height,
        fps=settings.camera.fps,
        format=settings.camera.format,
        backend=settings.camera.backend,
    )


def main() -> int:
    args = _parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    settings = load_settings(args.config)
    if args.pose is not None:
        settings.pose.model = args.pose  # type: ignore[assignment]
    if args.classifier is not None:
        settings.classifier.model_type = args.classifier  # type: ignore[assignment]
    if args.out_dir is not None:
        settings.database.output_dir = args.out_dir
    if args.metrics:
        settings.pipeline.metrics_enabled = True

    try:
        pipeline = build_pipeline(settings)
    except Exception as exc:  # noqa: BLE001 - surface component init failures clearly
        logger.error("Pipeline build failed: %s", exc)
        return 1

    source = _make_source(args.source, settings)
    try:
        source.start()
    except VideoSourceError as exc:
        logger.error("%s", exc)
        return 1

    session_id = pipeline.start_session()
    logger.info(
        "Session %s started (source=%r, pose=%s, classifier=%s)",
        session_id,
        args.source,
        settings.pose.model,
        settings.classifier.model_type,
    )

    frames_processed = 0
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            result = pipeline.process_frame(frame)
            frames_processed += 1
            for event in result.new_events:
                print(f"[{time.strftime('%H:%M:%S')}] {event.type:>18} | {event.message}")
            if result.has_error:
                logger.warning("Frame %d error: %s", result.frame_number, result.error)
            if args.max_frames and frames_processed >= args.max_frames:
                break
    finally:
        pipeline.end_session()
        source.stop()

    logger.info(
        "Processed %d frames; session log written to %s",
        frames_processed,
        settings.database.output_dir / f"{session_id}.jsonl",
    )
    if pipeline.metrics is not None:
        logger.info("Camera diagnostics: %s", source.diagnostics())
        report = pipeline.timing_report()
        if report:
            logger.info("Vision metrics:\n%s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
