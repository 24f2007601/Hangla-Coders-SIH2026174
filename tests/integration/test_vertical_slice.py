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

STEPS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
HOLD = 8  # classify calls per step so the smoothing vote stabilizes


def _run_session(tmp_path: Path, classifier: ScriptedClassifier, num_frames: int = 400) -> Path:
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]


def _events(records: list[dict]) -> list[dict]:
    return [r for r in records if r["kind"] == "event"]


def test_vertical_slice_correct_run(tmp_path: Path) -> None:
    classifier = ScriptedClassifier([(step, HOLD) for step in STEPS])
    path = _run_session(tmp_path, classifier)

    events = _events(_load(path))
    confirmed = [e for e in events if e["type"] == "step_confirmed"]
    assert [e["step"] for e in confirmed] == STEPS

    assert any(e["type"] == "protocol_complete" for e in events)
    assert not any(e["type"] in ("step_skipped", "out_of_sequence") for e in events)

    summary = [r for r in _load(path) if r["kind"] == "summary"][0]
    assert summary["steps_completed"] == STEPS


def test_vertical_slice_skip_step_scenario(tmp_path: Path) -> None:
    """Astronaut skips S3 (place under scope): system flags skipped + out-of-sequence."""
    sequence = ["S0", "S1", "S2", "S4", "S5", "S6", "S7"]  # S3 missing
    classifier = ScriptedClassifier([(step, HOLD) for step in sequence])
    path = _run_session(tmp_path, classifier)

    events = _events(_load(path))
    skipped = [e for e in events if e["type"] == "step_skipped"]
    assert [e["step"] for e in skipped] == ["S3"]

    out_of_seq = [e for e in events if e["type"] == "out_of_sequence"]
    assert [e["step"] for e in out_of_seq] == ["S4"]

    confirmed = [e for e in events if e["type"] == "step_confirmed"]
    assert [e["step"] for e in confirmed] == ["S0", "S1", "S2", "S5", "S6", "S7"]
    assert any(e["type"] == "protocol_complete" for e in events)


def test_vertical_slice_dummy_classifier_records_unknown(tmp_path: Path) -> None:
    """With no trained model the pipeline still runs and records unknown steps."""
    path = _run_session(tmp_path, ScriptedClassifier([], confidence=0.0))

    events = _events(_load(path))
    assert not any(e["type"] == "step_confirmed" for e in events)

    observations = [r for r in _load(path) if r["kind"] == "observation"]
    assert len(observations) > 0
    assert "person_id" in observations[0]
