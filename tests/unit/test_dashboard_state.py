"""Unit tests for the dashboard live-state bridge (state snapshot + UI widgets).

The state snapshot tests are pure; the widget tests run offscreen so they pass
on headless CI machines.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bas_assistant.ui.state import (
    STANDBY_STEP_LABEL,
    build_status_snapshot,
    gate_badge_states,
    status_label,
    step_display_label,
)
from bas_assistant.validation.protocol import (
    DEFAULT_MICROPHONE_PROTOCOL,
    GATE_ONE_MICROPHONE_PAIRED,
    GATE_RECEIVER_CONNECTED,
)
from tests.helpers import ScriptedClassifier, build_test_pipeline

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _frames(n: int, width: int = 160, height: int = 120) -> list[np.ndarray]:
    return [np.full((height, width, 3), 90, dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# gate_badge_states
# ---------------------------------------------------------------------------


def test_gate_badge_states_not_required() -> None:
    states = gate_badge_states("not_required")
    assert states[GATE_RECEIVER_CONNECTED] == "NOT_REQUIRED"
    assert states[GATE_ONE_MICROPHONE_PAIRED] == "NOT_REQUIRED"


def test_gate_badge_states_g1_transitions() -> None:
    assert gate_badge_states("G1_PENDING")[GATE_RECEIVER_CONNECTED] == "PENDING"
    assert gate_badge_states("G1_PASSED")[GATE_RECEIVER_CONNECTED] == "PASSED"
    # G2 stays not required while G1 is in progress.
    assert gate_badge_states("G1_PENDING")[GATE_ONE_MICROPHONE_PAIRED] == "NOT_REQUIRED"


def test_gate_badge_states_g2_transitions() -> None:
    assert gate_badge_states("G2_PENDING")[GATE_ONE_MICROPHONE_PAIRED] == "PENDING"
    assert gate_badge_states("G2_PASSED")[GATE_ONE_MICROPHONE_PAIRED] == "PASSED"
    # G1 remains PASSED once the pipeline moved on to G2.
    assert gate_badge_states("G2_PASSED")[GATE_RECEIVER_CONNECTED] == "PASSED"


# ---------------------------------------------------------------------------
# step display labels / status label
# ---------------------------------------------------------------------------


def test_step_display_label_resolves_known_steps() -> None:
    step = DEFAULT_MICROPHONE_PROTOCOL.step("M4")
    assert step is not None
    assert step_display_label("M4", DEFAULT_MICROPHONE_PROTOCOL) == step.name
    assert step_display_label(None, DEFAULT_MICROPHONE_PROTOCOL) == STANDBY_STEP_LABEL


def test_step_display_label_falls_back_to_raw_id() -> None:
    assert step_display_label("background", DEFAULT_MICROPHONE_PROTOCOL) == "background"


class _ResultStub:
    """Minimal FrameResult stand-in for status_label tests."""

    def __init__(self, error: str | None = None, event_types: list[str] | None = None) -> None:
        self.error = error

        class _Event:
            def __init__(self, type: str) -> None:
                self.type = type

        self.new_events = [_Event(t) for t in (event_types or [])]

    @property
    def has_error(self) -> bool:
        return self.error is not None


def test_status_label_priorities() -> None:
    assert status_label(_ResultStub(error="boom")) == "error"
    assert status_label(_ResultStub(event_types=["step_confirmed"])) == "confirmed"
    assert status_label(_ResultStub()) == "monitoring"


# ---------------------------------------------------------------------------
# build_status_snapshot (public pipeline properties)
# ---------------------------------------------------------------------------


def test_pipeline_public_properties(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    assert pipeline.session_id is None
    assert pipeline.protocol_state["done_steps"] == []
    assert pipeline.protocol_state["is_complete"] is False

    session_id = pipeline.start_session()
    assert pipeline.session_id == session_id

    pipeline.process_frame(_frames(1)[0])
    assert pipeline.frame_number == 1

    led = pipeline.led_observation
    assert led.left == "unknown"
    assert led.right == "unknown"
    assert led.receiver_detected is False
    assert led.g1_passed is False
    assert led.g2_passed is False
    pipeline.end_session()


def test_protocol_state_reflects_confirmed_steps(tmp_path: Path) -> None:
    classifier = ScriptedClassifier([("M0", 8), ("M1", 8)])
    pipeline = build_test_pipeline(tmp_path, classifier=classifier)
    pipeline.start_session()
    for frame in _frames(200):
        pipeline.process_frame(frame)
    pipeline.end_session()

    state = pipeline.protocol_state
    assert state["done_steps"] == ["M0", "M1"]
    assert state["expected_next_id"] == "M2"
    assert state["is_complete"] is False


def test_build_status_snapshot_full_contract(tmp_path: Path) -> None:
    classifier = ScriptedClassifier([("M0", 8)])
    pipeline = build_test_pipeline(tmp_path, classifier=classifier)
    pipeline.start_session()

    result = pipeline.process_frame(_frames(1)[0])
    snapshot = build_status_snapshot(
        pipeline, result, DEFAULT_MICROPHONE_PROTOCOL, active_events=1, source_kind="SIMULATED"
    )
    pipeline.end_session()

    # System + video
    assert snapshot["status"] in {"monitoring", "confirmed", "error"}
    assert snapshot["fps"] >= 0.0
    assert snapshot["persons"] == 1
    assert snapshot["active_events"] == 1
    assert snapshot["source_kind"] == "SIMULATED"
    assert snapshot["session_id"] is not None

    # Classification + protocol (first frame may precede the first classify hop)
    assert snapshot["step_id"] in {None, "M0", "M1", "background", "unknown"}
    assert snapshot["step"] == step_display_label(snapshot["step_id"], DEFAULT_MICROPHONE_PROTOCOL)
    assert 0.0 <= snapshot["confidence"] <= 1.0
    assert isinstance(snapshot["done_steps"], list)
    assert snapshot["gate_status"] == "not_required"
    assert snapshot["is_complete"] is False
    # FrameResult carries the 0-based counter of the processed frame.
    assert snapshot["frame_number"] == 0
    assert pipeline.frame_number == 1

    # Gates / LEDs
    led = snapshot["led"]
    assert set(led) == {
        "left",
        "right",
        "receiver_detected",
        "receiver_confidence",
        "g1_passed",
        "g2_passed",
        "ready",
    }
    assert led["g1_passed"] is False
    assert led["g2_passed"] is False


def test_build_status_snapshot_reports_frame_error(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    result = pipeline.process_frame(_frames(1)[0])  # no session -> error
    assert result.has_error

    snapshot = build_status_snapshot(pipeline, result, DEFAULT_MICROPHONE_PROTOCOL)
    assert snapshot["status"] == "error"
    assert snapshot["frame_error"] is not None
    assert "RuntimeError" in snapshot["frame_error"]


# ---------------------------------------------------------------------------
# Qt widgets (offscreen)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_protocol_progress_widget_states(qapp) -> None:  # noqa: ARG001
    from bas_assistant.ui.widgets import ProtocolProgressWidget

    widget = ProtocolProgressWidget(protocol=DEFAULT_MICROPHONE_PROTOCOL)
    rows = {row.entry_id: row for row in widget._rows}  # noqa: SLF001

    assert set(rows) == {"M0", "M1", "M2", "M3", "M4", "G1", "M5", "M6", "G2"}

    # Mid-protocol: M0-M3 done, M4 active, G1 pending.
    widget.update_progress(
        done_steps=["M0", "M1", "M2", "M3"],
        expected_next_id="M4",
        gate_status="G1_PENDING",
    )
    assert rows["M0"]._state == "done"  # noqa: SLF001
    assert rows["M4"]._state == "active"  # noqa: SLF001
    assert rows["G1"]._state == "gate_pending"  # noqa: SLF001
    assert rows["G2"]._state == "gate_not_required"  # noqa: SLF001

    # Gates passed + protocol complete.
    widget.update_progress(
        done_steps=["M0", "M1", "M2", "M3", "M4", "M5", "M6"],
        expected_next_id=None,
        gate_status="G2_PASSED",
        is_complete=True,
    )
    assert rows["G1"]._state == "gate_passed"  # noqa: SLF001
    assert rows["G2"]._state == "gate_passed"  # noqa: SLF001
    assert rows["M6"]._state == "done"  # noqa: SLF001

    widget.reset_progress()
    assert rows["M0"]._state == "pending"  # noqa: SLF001
    assert rows["G1"]._state == "gate_not_required"  # noqa: SLF001


def test_verification_gates_widget_update(qapp) -> None:  # noqa: ARG001
    from bas_assistant.ui.widgets import VerificationGatesWidget

    widget = VerificationGatesWidget(protocol=DEFAULT_MICROPHONE_PROTOCOL)

    led = {
        "left": "blinking",
        "right": "blinking",
        "receiver_detected": True,
        "receiver_confidence": 0.77,
        "g1_passed": True,
        "g2_passed": False,
        "ready": True,
    }
    widget.update_gates("G1_PASSED", led)

    g1_status = widget._g1_badge._lbl_status.text()  # noqa: SLF001
    g2_status = widget._g2_badge._lbl_status.text()  # noqa: SLF001
    assert g1_status == "PASSED"
    assert g2_status == "NOT REQUIRED"
    assert "LEFT: BLINKING" in widget._led_labels["LEFT"].text()  # noqa: SLF001
    assert "DETECTED" in widget._receiver_label.text()  # noqa: SLF001

    widget.reset_gates()
    assert widget._g1_badge._lbl_status.text() == "NOT REQUIRED"  # noqa: SLF001
