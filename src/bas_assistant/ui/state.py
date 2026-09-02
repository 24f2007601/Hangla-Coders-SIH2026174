"""Qt-free state snapshot helpers shared by the dashboard worker and widgets.

`build_status_snapshot` assembles the full live-data contract from public
`ExperimentPipeline` properties plus one `FrameResult`. It contains no Qt or
business logic: the pipeline/FSM remains the single source of truth and this
module only reformats backend state for display.
"""

from __future__ import annotations

from typing import Any

from bas_assistant.pipeline.pipeline import ExperimentPipeline, FrameResult
from bas_assistant.validation.protocol import (
    DEFAULT_MICROPHONE_PROTOCOL,
    GATE_ONE_MICROPHONE_PAIRED,
    GATE_RECEIVER_CONNECTED,
    ExperimentProtocol,
)

STANDBY_STEP_LABEL = "Idle / Standby"


def gate_badge_states(gate_status: str) -> dict[str, str]:
    """Map a pipeline gate status string to per-gate badge phases.

    Returns ``{gate_id: "NOT_REQUIRED" | "PENDING" | "PASSED"}``. The gates run
    in order (G1 then G2), so once G2 is pending/passed, G1 has already passed.

    >>> gate_badge_states("not_required") == {
    ...     GATE_RECEIVER_CONNECTED: 'NOT_REQUIRED',
    ...     GATE_ONE_MICROPHONE_PAIRED: 'NOT_REQUIRED',
    ... }
    True
    >>> gate_badge_states("G1_PENDING")[GATE_RECEIVER_CONNECTED]
    'PENDING'
    >>> gate_badge_states("G2_PASSED")[GATE_RECEIVER_CONNECTED]
    'PASSED'
    """
    gate_status = gate_status or "not_required"

    g1_phase = "NOT_REQUIRED"
    if "G2_PENDING" in gate_status or "G2_PASSED" in gate_status or "G1_PASSED" in gate_status:
        g1_phase = "PASSED"
    elif "G1_PENDING" in gate_status:
        g1_phase = "PENDING"

    g2_phase = "NOT_REQUIRED"
    if "G2_PASSED" in gate_status:
        g2_phase = "PASSED"
    elif "G2_PENDING" in gate_status:
        g2_phase = "PENDING"

    return {
        GATE_RECEIVER_CONNECTED: g1_phase,
        GATE_ONE_MICROPHONE_PAIRED: g2_phase,
    }


def step_display_label(step_id: str | None, protocol: ExperimentProtocol) -> str:
    """Human-readable label for a classified step id."""
    if step_id is None:
        return STANDBY_STEP_LABEL
    step = protocol.step(step_id)
    return step.name if step else step_id


def status_label(result: FrameResult) -> str:
    """Short status string for the telemetry chip (last notable outcome)."""
    if result.has_error:
        return "error"
    if result.new_events:
        last = result.new_events[-1].type.replace("step_", "").replace("_", " ")
        return last
    return "monitoring"


def build_status_snapshot(
    pipeline: ExperimentPipeline,
    result: FrameResult,
    protocol: ExperimentProtocol = DEFAULT_MICROPHONE_PROTOCOL,
    *,
    active_events: int = 0,
    source_kind: str = "webcam",
) -> dict[str, Any]:
    """Assemble the dashboard status payload from public pipeline state.

    Reads only public read-only properties; never mutates the pipeline.
    """
    classification = result.classification
    step_id = classification.step if classification else None
    confidence = classification.confidence if classification else 0.0
    protocol_state = pipeline.protocol_state
    led = pipeline.led_observation

    return {
        "status": status_label(result),
        "fps": result.fps,
        "persons": 1 if result.person_id else 0,
        "active_events": active_events,
        "latency_ms": result.inference_latency_ms,
        "frame_number": result.frame_number,
        "session_id": pipeline.session_id,
        "source_kind": source_kind,
        "step": step_display_label(step_id, protocol),
        "step_id": step_id,
        "confidence": confidence,
        "done_steps": protocol_state["done_steps"],
        "expected_next_id": protocol_state["expected_next_id"],
        "expected_next_name": protocol_state["expected_next_name"],
        "is_complete": protocol_state["is_complete"],
        "gate_status": pipeline.gate_status,
        "led": {
            "left": led.left,
            "right": led.right,
            "receiver_detected": led.receiver_detected,
            "receiver_confidence": led.receiver_confidence,
            "g1_passed": led.g1_passed,
            "g2_passed": led.g2_passed,
            "ready": led.ready,
        },
        "frame_error": result.error,
    }


__all__ = [
    "STANDBY_STEP_LABEL",
    "build_status_snapshot",
    "gate_badge_states",
    "status_label",
    "step_display_label",
]
