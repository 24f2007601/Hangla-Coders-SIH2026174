"""Integration test: dummy frame -> pipeline -> step classification -> FSM -> JSON log.

No GPU and no camera required. This is the acceptance test for the vertical slice
(success-criteria #2/#3): the full Observe -> Validate -> Record loop with real
pipeline components and a scripted classifier standing in for the untrained XGBoost
model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.helpers import ScriptedClassifier, build_test_pipeline

STEPS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
HOLD = 8  # classify calls per step so the smoothing vote stabilizes


def _run_session(
    tmp_path: Path,
    classifier: ScriptedClassifier,
    num_frames: int = 400,
) -> Path:
    pipeline = build_test_pipeline(tmp_path, classifier=classifier)
    session_id = pipeline.start_session()
    frame = np.full((120, 160, 3), 90, dtype=np.uint8)

    for _ in range(num_frames):
        pipeline.process_frame(frame)

    pipeline.end_session()
    path = tmp_path / f"{session_id}.jsonl"
    assert path.exists()
    return path


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().splitlines()
    ]


def _events(records: list[dict]) -> list[dict]:
    return [record for record in records if record["kind"] == "event"]


def test_vertical_slice_correct_run(tmp_path: Path) -> None:
    classifier = ScriptedClassifier([(step, HOLD) for step in STEPS])
    path = _run_session(tmp_path, classifier)

    events = _events(_load(path))
    confirmed = [event for event in events if event["type"] == "step_confirmed"]

    assert [event["step"] for event in confirmed] == STEPS
    assert any(event["type"] == "protocol_complete" for event in events)
    assert not any(
        event["type"] in ("step_skipped", "out_of_sequence") for event in events
    )

    summary = [record for record in _load(path) if record["kind"] == "summary"][0]
    assert summary["steps_completed"] == STEPS


def test_vertical_slice_skip_step_scenario(tmp_path: Path) -> None:
    """Skip M4 (remove receiver) and jump directly to M5."""
    sequence = ["M0", "M1", "M2", "M3", "M5", "M6"]
    classifier = ScriptedClassifier([(step, HOLD) for step in sequence])
    path = _run_session(tmp_path, classifier)

    events = _events(_load(path))

    skipped = [event for event in events if event["type"] == "step_skipped"]
    assert [event["step"] for event in skipped] == ["M4"]

    out_of_seq = [event for event in events if event["type"] == "out_of_sequence"]
    assert [event["step"] for event in out_of_seq] == ["M5"]

    confirmed = [event for event in events if event["type"] == "step_confirmed"]
    assert [event["step"] for event in confirmed] == [
        "M0",
        "M1",
        "M2",
        "M3",
        "M6",
    ]

    assert any(event["type"] == "protocol_complete" for event in events)


def test_vertical_slice_dummy_classifier_records_unknown(tmp_path: Path) -> None:
    """With no trained model the pipeline still runs and records unknown steps."""
    path = _run_session(tmp_path, ScriptedClassifier([], confidence=0.0))

    events = _events(_load(path))
    assert not any(event["type"] == "step_confirmed" for event in events)

    observations = [
        record for record in _load(path) if record["kind"] == "observation"
    ]
    assert len(observations) > 0
    assert "person_id" in observations[0]