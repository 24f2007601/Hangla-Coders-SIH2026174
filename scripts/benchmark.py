"""Micro-benchmark of per-frame pipeline latency and throughput.

Runs the pipeline over a synthetic source with dummy components (no GPU, no camera)
and reports measured FPS and per-stage latency. Only measured numbers are reported —
no invented performance claims.

Usage:
    python scripts/benchmark.py --max-frames 500
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bas_assistant.classification.classifier import DummyClassifier
from bas_assistant.config.settings import Settings
from bas_assistant.detection.detector import DummyPersonDetector
from bas_assistant.events.manager import EventManager
from bas_assistant.features.extractor import PoseFeatureExtractor
from bas_assistant.pipeline.pipeline import ExperimentPipeline
from bas_assistant.pose.estimation import DummyPoseEstimator
from bas_assistant.storage.repository import JsonResultRepository
from bas_assistant.tracking.tracker import DummyPersonTracker
from bas_assistant.utils.logging import setup_logging
from bas_assistant.validation.fsm import ExperimentFSM
from bas_assistant.validation.protocol import DEFAULT_TOY_PROTOCOL
from bas_assistant.video.source import DummyVideoSource

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-frame pipeline latency and FPS (dummy components)."
    )
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(logging.ERROR)

    settings = Settings()
    settings.pipeline.sequence_length = args.sequence_length
    settings.pipeline.classify_hop = 5
    settings.pipeline.smoothing_window = 5

    pipeline = ExperimentPipeline(
        settings=settings,
        detector=DummyPersonDetector(),
        tracker=DummyPersonTracker(),
        pose_estimator=DummyPoseEstimator(motion=0.0),
        feature_extractor=PoseFeatureExtractor(args.sequence_length),
        classifier=DummyClassifier(),
        validator=ExperimentFSM(DEFAULT_TOY_PROTOCOL),
        event_manager=EventManager(),
        repository=JsonResultRepository(Path("/tmp/bas_benchmark")),
    )
    source = DummyVideoSource(width=args.width, height=args.height, num_frames=args.max_frames)
    source.start()
    pipeline.start_session()

    latencies: list[float] = []
    fps_values: list[float] = []
    try:
        start = time.perf_counter()
        for _ in range(args.max_frames):
            frame = source.read()
            if frame is None:
                break
            result = pipeline.process_frame(frame)
            latencies.append(result.inference_latency_ms)
            fps_values.append(result.fps)
        elapsed = time.perf_counter() - start
    finally:
        pipeline.end_session()
        source.stop()

    n = len(latencies)
    if n == 0:
        logger.error("No frames processed")
        return 1

    print(f"Frames processed:        {n}")
    print(f"Total time:              {elapsed:.2f} s")
    print(f"Mean throughput:         {n / elapsed:.1f} fps")
    print(f"Median per-frame latency:{statistics.median(latencies):.2f} ms")
    print(f"Mean per-frame latency:  {statistics.mean(latencies):.2f} ms")
    print(f"P95 per-frame latency:   {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
    print(f"Reported FPS meter:      {statistics.mean(fps_values):.1f} fps")
    print("(Synthetic dummy components only; numbers are measured on this machine.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
