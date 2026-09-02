"""Regression coverage for receiver LED verification gates."""

from __future__ import annotations

import numpy as np

from bas_assistant.validation.led_estimator import LEDStateEstimator
from tests.helpers import ScriptedClassifier, build_test_pipeline


def _receiver_frame(left_on: bool, right_on: bool) -> tuple[np.ndarray, list[dict]]:
    """Create a small horizontal receiver with two blue LED positions."""
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    frame[20:60, 30:130] = (20, 20, 20)
    if left_on:
        frame[35:43, 42:50] = (255, 40, 10)
    if right_on:
        frame[35:43, 88:96] = (255, 40, 10)
    return frame, [{"name": "receiver", "confidence": 0.9, "xyxy": [30, 20, 130, 60]}]


def test_led_estimator_uses_supplied_timestamps_and_resets_history() -> None:
    estimator = LEDStateEstimator()

    for frame_number in range(30):
        frame, objects = _receiver_frame(frame_number % 4 < 2, frame_number % 6 < 3)
        observation = estimator.update(frame, objects, timestamp=frame_number / 30.0)

    assert observation["ready"] is True
    assert observation["left"] == "blinking"
    assert observation["right"] == "blinking"
    assert observation["g1_passed"] is True

    estimator.reset_history()

    frame, objects = _receiver_frame(True, True)
    reset_observation = estimator.update(frame, objects, timestamp=2.0)
    assert reset_observation["ready"] is False
    assert reset_observation["g1_passed"] is False


def test_led_estimator_never_passes_a_gate_without_a_confident_receiver() -> None:
    estimator = LEDStateEstimator()
    frame, _ = _receiver_frame(True, True)

    observation = estimator.update(frame, [], timestamp=1.0)

    assert observation["receiver_detected"] is False
    assert observation["receiver_bbox"] is None
    assert observation["left"] == "unknown"
    assert observation["right"] == "unknown"
    assert observation["g1_passed"] is False
    assert observation["g2_passed"] is False


class _GateEstimator:
    """Deterministic gate signal used to exercise protocol ordering."""

    def __init__(self) -> None:
        self.g1_passed = False
        self.g2_passed = False
        self.reset_calls = 0

    @property
    def state(self) -> dict[str, str]:
        return {"left": "unknown", "right": "unknown"}

    def update(self, frame, objects, timestamp=None) -> dict[str, object]:
        return {
            "receiver_detected": True,
            "receiver_bbox": [0.0, 0.0, 1.0, 1.0],
            "left": "unknown",
            "right": "unknown",
            "left_score": 0.0,
            "right_score": 0.0,
            "ready": False,
            "g1_passed": self.g1_passed,
            "g2_passed": self.g2_passed,
        }

    def reset(self) -> None:
        self.reset_history()

    def reset_history(self) -> None:
        self.reset_calls += 1


def test_pipeline_holds_m5_and_m6_until_their_led_gates_pass(tmp_path) -> None:
    classifier = ScriptedClassifier(
        [
            ("M0", 2),
            ("M1", 2),
            ("M2", 2),
            ("M3", 2),
            ("M4", 2),
            ("M5", 3),
            ("M6", 12),
        ]
    )
    pipeline = build_test_pipeline(tmp_path, sequence_length=1, hop=1, classifier=classifier)
    gate_estimator = _GateEstimator()
    pipeline._led_estimator = gate_estimator  # noqa: SLF001
    pipeline.start_session()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    for _ in range(12):
        pipeline.process_frame(frame)

    assert pipeline._validator.done_steps == ["M0", "M1", "M2", "M3", "M4"]  # noqa: SLF001
    assert pipeline._m5_pending is True  # noqa: SLF001

    gate_estimator.g1_passed = True
    g1_result = pipeline.process_frame(frame)
    assert [event.type for event in g1_result.new_events] == ["gate_g1_passed", "step_confirmed"]
    assert pipeline._validator.done_steps[-1] == "M5"  # noqa: SLF001

    for _ in range(4):
        pipeline.process_frame(frame)
    assert pipeline._m6_pending is True  # noqa: SLF001
    assert [event.type for event in pipeline._event_manager.events].count("gate_g2_pending") == 1  # noqa: SLF001

    gate_estimator.g2_passed = True
    g2_result = pipeline.process_frame(frame)
    assert [event.type for event in g2_result.new_events] == [
        "gate_g2_passed",
        "step_confirmed",
        "protocol_complete",
    ]
    assert pipeline._validator.is_complete is True  # noqa: SLF001
