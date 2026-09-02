"""Integration tests for the BAS protocol vertical slice and LED gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.helpers import ScriptedClassifier, build_test_pipeline

STEPS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
HOLD = 8


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]


def _events(records: list[dict]) -> list[dict]:
    return [record for record in records if record["kind"] == "event"]


class _GateEstimator:
    """Deterministic LED gate signal for pipeline integration tests."""

    def __init__(self) -> None:
        self.g1_passed = False
        self.g2_passed = False
        self.reset_calls = 0

    @property
    def state(self) -> dict[str, str]:
        return {"left": "unknown", "right": "unknown"}

    def update(self, frame, objects, timestamp=None, now=None) -> dict[str, object]:
        return {
            "receiver_detected": True,
            "receiver_bbox": [0.0, 0.0, 1.0, 1.0],
            "receiver_confidence": 0.9,
            "left": "unknown",
            "right": "unknown",
            "left_score": 0.0,
            "right_score": 0.0,
            "history_seconds": 0.0,
            "sample_count": 0,
            "ready": False,
            "g1_passed": self.g1_passed,
            "g2_passed": self.g2_passed,
        }

    def reset(self) -> None:
        self.reset_history()

    def reset_history(self) -> None:
        self.reset_calls += 1


def _drive_until(
    pipeline,
    predicate,
    frame: np.ndarray,
    limit: int = 300,
) -> None:
    """Process frames until a state predicate becomes true."""
    for _ in range(limit):
        pipeline.process_frame(frame)
        if predicate():
            return
    raise AssertionError("Pipeline did not reach the expected state.")


def _confirmed_steps(pipeline) -> list[str]:
    return [
        event.message.replace("Step ", "").replace(" confirmed.", "")
        for event in pipeline._event_manager.events  # noqa: SLF001
        if event.type == "step_confirmed"
    ]


def test_vertical_slice_correct_run_requires_both_gates(tmp_path: Path) -> None:
    """A correct run must pass G1 before M5 and G2 before M6."""
    classifier = ScriptedClassifier(
        [
            ("M0", HOLD),
            ("M1", HOLD),
            ("M2", HOLD),
            ("M3", HOLD),
            ("M4", HOLD),
            ("M5", HOLD),
            ("M6", HOLD * 2),
        ]
    )
    pipeline = build_test_pipeline(tmp_path, classifier=classifier)
    gate_estimator = _GateEstimator()
    pipeline._led_estimator = gate_estimator  # noqa: SLF001
    pipeline.start_session()

    frame = np.full((120, 160, 3), 90, dtype=np.uint8)

    # Reach the point where M5 has been recognized but is still held behind G1.
    _drive_until(
        pipeline,
        lambda: pipeline._m5_pending,  # noqa: SLF001
        frame,
    )

    assert pipeline._validator.done_steps == [  # noqa: SLF001
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
    ]

    # Pass G1. Do not assume this happens on the same frame as the next
    # classifier call; the pipeline's feature/hop cadence may require more
    # than one processed frame.
    gate_estimator.g1_passed = True

    _drive_until(
        pipeline,
        lambda: pipeline._validator.done_steps[-1:] == ["M5"],  # noqa: SLF001
        frame,
    )

    event_types = [event.type for event in pipeline._event_manager.events]  # noqa: SLF001
    assert "gate_g1_passed" in event_types
    assert pipeline._m5_pending is False  # noqa: SLF001

    # Reach M6_PENDING while G2 is still false.
    _drive_until(
        pipeline,
        lambda: pipeline._m6_pending,  # noqa: SLF001
        frame,
    )

    assert pipeline._validator.done_steps == [  # noqa: SLF001
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ]
    assert pipeline._validator.is_complete is False  # noqa: SLF001

    # Keep feeding M6 evidence with G2 deliberately false.
    for _ in range(30):
        pipeline.process_frame(frame)

    assert pipeline._m6_pending is True  # noqa: SLF001
    assert pipeline._validator.done_steps[-1] == "M5"  # noqa: SLF001
    assert pipeline._validator.is_complete is False  # noqa: SLF001

    event_types = [event.type for event in pipeline._event_manager.events]  # noqa: SLF001
    assert "gate_g2_passed" not in event_types

    # Now pass G2 and let the pipeline finish.
    gate_estimator.g2_passed = True

    _drive_until(
        pipeline,
        lambda: pipeline._validator.is_complete,  # noqa: SLF001
        frame,
    )

    event_types = [event.type for event in pipeline._event_manager.events]  # noqa: SLF001

    assert "gate_g2_passed" in event_types
    assert "protocol_complete" in event_types

    g2_index = event_types.index("gate_g2_passed")
    complete_index = event_types.index("protocol_complete")
    assert g2_index < complete_index

    assert pipeline._validator.done_steps == STEPS  # noqa: SLF001


def test_vertical_slice_holds_m6_until_g2(tmp_path: Path) -> None:
    """Repeated M6 classifications cannot commit M6 while G2 is false."""
    classifier = ScriptedClassifier(
        [
            ("M0", HOLD),
            ("M1", HOLD),
            ("M2", HOLD),
            ("M3", HOLD),
            ("M4", HOLD),
            ("M5", HOLD),
            ("M6", HOLD * 3),
        ]
    )
    pipeline = build_test_pipeline(tmp_path, classifier=classifier)
    gate_estimator = _GateEstimator()
    pipeline._led_estimator = gate_estimator  # noqa: SLF001
    pipeline.start_session()

    frame = np.full((120, 160, 3), 90, dtype=np.uint8)

    _drive_until(
        pipeline,
        lambda: pipeline._m5_pending,  # noqa: SLF001
        frame,
    )

    gate_estimator.g1_passed = True

    # Wait for the actual M5 commit rather than assuming it occurs on the
    # first frame after G1 becomes true.
    _drive_until(
        pipeline,
        lambda: pipeline._validator.done_steps[-1:] == ["M5"],  # noqa: SLF001
        frame,
    )

    _drive_until(
        pipeline,
        lambda: pipeline._m6_pending,  # noqa: SLF001
        frame,
    )

    # This is the critical regression check: M6 must remain pending.
    for _ in range(50):
        pipeline.process_frame(frame)

    assert pipeline._m6_pending is True  # noqa: SLF001
    assert pipeline._validator.done_steps[-1] == "M5"  # noqa: SLF001
    assert pipeline._validator.is_complete is False  # noqa: SLF001

    event_types = [event.type for event in pipeline._event_manager.events]  # noqa: SLF001
    assert "gate_g2_passed" not in event_types

    # A G2 pass must be the event that unlocks M6.
    gate_estimator.g2_passed = True

    _drive_until(
        pipeline,
        lambda: pipeline._validator.is_complete,  # noqa: SLF001
        frame,
    )

    event_types = [event.type for event in pipeline._event_manager.events]  # noqa: SLF001
    assert "gate_g2_passed" in event_types
    assert pipeline._validator.done_steps[-1] == "M6"  # noqa: SLF001
    assert pipeline._validator.is_complete is True  # noqa: SLF001


def test_vertical_slice_dummy_classifier_records_unknown(tmp_path: Path) -> None:
    """With no trained model the pipeline still runs and records unknown steps."""
    path = _run_session(
        tmp_path,
        ScriptedClassifier([], confidence=0.0),
    )

    events = _events(_load(path))
    assert not any(event["type"] == "step_confirmed" for event in events)

    observations = [record for record in _load(path) if record["kind"] == "observation"]
    assert len(observations) > 0
    assert "person_id" in observations[0]
