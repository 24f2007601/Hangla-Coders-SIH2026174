"""Unit tests for the pipeline orchestrator (dummy components, failure tolerance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bas_assistant.events.models import EVENT_CONFIRMED, EVENT_SESSION_STARTED
from bas_assistant.models import BoundingBox
from tests.helpers import ScriptedClassifier, build_test_pipeline


def _frames(n: int, width: int = 160, height: int = 120) -> list[np.ndarray]:
    return [np.full((height, width, 3), 90, dtype=np.uint8) for _ in range(n)]


def test_pipeline_runs_end_to_end_with_dummy_classifier(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    session_id = pipeline.start_session()

    results = [pipeline.process_frame(f) for f in _frames(120)]
    pipeline.end_session()

    assert session_id.startswith("session_")
    for result in results:
        assert result.has_error is False
        assert result.person_id == 1
        assert result.pose is not None
        assert result.fps >= 0.0

    # Session started event is always published first.
    events = pipeline._event_manager.events  # noqa: SLF001
    assert events[0].type == EVENT_SESSION_STARTED

    # The JSONL session log was written with observations.
    log = tmp_path / f"{session_id}.jsonl"
    assert log.exists()
    assert "observation" in log.read_text(encoding="utf-8")


def test_pipeline_publishes_confirmed_step_events(tmp_path: Path) -> None:
    classifier = ScriptedClassifier([("M0", 8), ("M1", 8), ("M2", 8)])
    pipeline = build_test_pipeline(tmp_path, classifier=classifier)
    pipeline.start_session()

    frames = _frames(400)
    for frame in frames:
        pipeline.process_frame(frame)
    pipeline.end_session()

    confirmed = [e for e in pipeline._event_manager.events if e.type == EVENT_CONFIRMED]
    assert [e.step for e in confirmed] == ["M0", "M1", "M2"]


def test_pipeline_tolerates_single_frame_failure(tmp_path: Path) -> None:
    class FlakyDetector:
        def __init__(self) -> None:
            self._calls = 0

        def detect(self, frame: np.ndarray) -> list[BoundingBox]:
            self._calls += 1
            if self._calls == 5:
                raise ValueError("boom")
            return [BoundingBox(0, 0, 159, 119)]

    pipeline = build_test_pipeline(tmp_path)
    pipeline._detector = FlakyDetector()  # noqa: SLF001
    pipeline.start_session()

    results = [pipeline.process_frame(f) for f in _frames(30)]
    pipeline.end_session()

    failed = [r for r in results if r.has_error]
    assert len(failed) == 1
    assert "ValueError" in failed[0].error
    assert sum(1 for r in results if not r.has_error) == 29
    # Frame failure is observable and does not kill the pipeline.
    assert results[-1].has_error is False


def test_pipeline_observations_before_session_are_errors(tmp_path: Path) -> None:
    """The pipeline tolerates a failing frame: it surfaces an error, never crashes."""
    pipeline = build_test_pipeline(tmp_path)
    result = pipeline.process_frame(_frames(1)[0])
    assert result.has_error is True
    assert "RuntimeError" in result.error
