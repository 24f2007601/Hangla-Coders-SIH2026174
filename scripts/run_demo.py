"""Interactive BAS demo: annotated video + protocol step validation."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bas_assistant.config.settings import load_settings
from bas_assistant.pipeline.factory import build_pipeline
from bas_assistant.utils.logging import setup_logging
from bas_assistant.utils.visualization import annotate_frame
from bas_assistant.video.source import DummyVideoSource, OpenCVVideoSource, VideoSourceError

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive BAS experiment demo with annotated video."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a configs/*.yaml file",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video file path, or 'dummy'",
    )
    parser.add_argument(
        "--pose",
        choices=["mediapipe", "dummy"],
        default=None,
        help="Pose backend override",
    )
    parser.add_argument(
        "--classifier",
        choices=["dummy", "xgboost"],
        default=None,
        help="Classifier override",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Enable debug timing instrumentation",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = until source ends)",
    )
    return parser.parse_args()


def _step_name(step: str | None) -> str:
    """Convert M0-M6 into a readable protocol label."""
    labels = {
        "M0": "M0 • Verify phone powered on",
        "M1": "M1 • Move phone to working station",
        "M2": "M2 • Pick microphone case",
        "M3": "M3 • Open microphone case",
        "M4": "M4 • Remove receiver",
        "M5": "M5 • Connect receiver to phone",
        "M6": "M6 • Remove one microphone",
    }

    if not step:
        return "idle"

    return labels.get(step, step)


def _build_source(args: argparse.Namespace, settings):
    """Create the configured video source."""
    if args.source == "dummy":
        return DummyVideoSource(
            width=settings.camera.width,
            height=settings.camera.height,
            num_frames=max(args.max_frames, 1_000_000),
        )

    source_value = int(args.source) if args.source.isdigit() else args.source

    return OpenCVVideoSource(
        source_value,
        width=settings.camera.width,
        height=settings.camera.height,
        fps=settings.camera.fps,
        format=settings.camera.format,
        backend=settings.camera.backend,
    )


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
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline build failed: %s", exc)
        return 1

    source = _build_source(args, settings)

    try:
        source.start()
    except VideoSourceError as exc:
        logger.error("%s", exc)
        logger.error(
            "No usable video source. Pass --source <video.mp4>, "
            "connect a webcam, or use --source dummy."
        )
        return 1

    pipeline.start_session()

    frames_processed = 0

    try:
        while True:
            frame = source.read()

            if frame is None:
                break

            result = pipeline.process_frame(frame, source_timestamp=source.timestamp)
            frames_processed += 1

            classification = result.classification
            step = classification.step if classification else None
            confidence = classification.confidence if classification else 0.0

            status = "error" if result.has_error else "monitoring"

            if result.new_events:
                latest_event = result.new_events[-1]
                status = latest_event.type.replace("step_", "").replace("_", " ")

                for event in result.new_events:
                    print(f"[{time.strftime('%H:%M:%S')}] {event.type:>22} | {event.message}")

            annotated = annotate_frame(
                frame,
                result.pose,
                _step_name(step),
                confidence,
                result.fps,
                status,
            )

            cv2.imshow("BAS Experiment Assistant - Demo", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if args.max_frames and frames_processed >= args.max_frames:
                break

            # ExperimentPipeline exposes the validator internally as
            # _validator. Stop automatically once M0-M6 is complete.
            validator = getattr(pipeline, "_validator", None)
            if validator is not None and getattr(validator, "is_complete", False):
                logger.info("Protocol complete.")
                break

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")

    except Exception:  # noqa: BLE001
        logger.exception("Demo runtime failure.")
        return 1

    finally:
        try:
            pipeline.end_session()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to close pipeline session.")

        try:
            source.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to stop video source.")

        cv2.destroyAllWindows()

    logger.info(
        "Demo finished after %d frames; JSONL session log written to data/processed/.",
        frames_processed,
    )

    if pipeline.metrics is not None:
        try:
            logger.info("Camera diagnostics: %s", source.diagnostics())
            report = pipeline.timing_report()

            if report:
                logger.info("Vision metrics:\n%s", report)

        except Exception:  # noqa: BLE001
            logger.exception("Failed to print diagnostics.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
