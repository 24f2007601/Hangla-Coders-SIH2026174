"""Reusable visual UI components for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np

from bas_assistant.ui.state import gate_badge_states
from bas_assistant.ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from bas_assistant.utils.visualization import annotate_frame
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL


def annotate_pipeline_frame(frame: np.ndarray, res: Any) -> np.ndarray:
    """Annotate raw frame using pipeline output."""
    step = res.classification.step if res.classification else "None"
    confidence = res.classification.confidence if res.classification else 0.0
    status = "error" if res.has_error else "monitoring"
    if res.new_events:
        status = res.new_events[-1].type.replace("step_", "").replace("_", " ")

    return annotate_frame(
        frame=frame,
        pose=res.pose,
        step_label=str(step),
        confidence=float(confidence),
        fps=float(res.fps),
        status=status,
    )


def format_gate_badge_html(label: str, phase: str) -> str:
    """Format single gate badge HTML with matching state style."""
    css_class = {
        "PASSED": "gate-passed",
        "PENDING": "gate-pending",
    }.get(phase, "gate-not-required")
    return f"<span class='gate-badge {css_class}'>{label}: {phase}</span>"


def render_telemetry_bar_html(snap: dict[str, Any]) -> str:
    """Format the top-level telemetry metrics bar."""
    status_color = ACCENT_EMERALD if snap["status"] != "ERROR" else "#EF4444"
    status_text = snap["status"].upper()
    active_step = snap["step_id"] or "NONE"
    gate_st = snap["gate_status"]

    return f"""
    <div class='telemetry-row'>
        <div class='telemetry-chip'>
            <div class='telemetry-label'>System Status</div>
            <div class='telemetry-value' style='color: {status_color};'>{status_text}</div>
        </div>
        <div class='telemetry-chip'>
            <div class='telemetry-label'>Inference FPS</div>
            <div class='telemetry-value' style='color: {ACCENT_CYAN};'>{snap['fps']:.1f}</div>
        </div>
        <div class='telemetry-chip'>
            <div class='telemetry-label'>Latency</div>
            <div class='telemetry-value'>{snap['latency_ms']:.1f} ms</div>
        </div>
        <div class='telemetry-chip'>
            <div class='telemetry-label'>Active Step</div>
            <div class='telemetry-value' style='color: {ACCENT_BLUE};'>{active_step}</div>
        </div>
        <div class='telemetry-chip'>
            <div class='telemetry-label'>Confidence</div>
            <div class='telemetry-value'>{snap['confidence'] * 100:.0f}%</div>
        </div>
        <div class='telemetry-chip'>
            <div class='telemetry-label'>Gate Status</div>
            <div class='telemetry-value' style='color: {ACCENT_AMBER}; font-size: 13px;'>
                {gate_st}
            </div>
        </div>
    </div>
    """


def render_step_card_html(snap: dict[str, Any]) -> str:
    """Format the active step card HTML."""
    step_name = snap["step"]
    step_id = snap["step_id"] or "--"
    conf_pct = snap["confidence"] * 100
    return f"""
    <div class='mc-card'>
        <div class='mc-card-header'>Protocol Execution Status</div>
        <div style='font-size: 15px; font-weight: 700; color: {TEXT_PRIMARY};'>
            <span style='color: {ACCENT_CYAN};'>{step_id}</span>: {step_name}
        </div>
        <div style='margin-top: 6px;'>
            <span style='font-size: 11px; color: {TEXT_MUTED};'>Classification Confidence: </span>
            <b style='color: {ACCENT_EMERALD}; font-family: monospace;'>{conf_pct:.1f}%</b>
        </div>
    </div>
    """


def render_gates_panel_html(snap: dict[str, Any]) -> str:
    """Format the verification gate badges and optical indicators."""
    g_states = gate_badge_states(snap["gate_status"])
    g1_html = format_gate_badge_html("G1 Receiver Connected", g_states.get("G1", "NOT_REQUIRED"))
    g2_html = format_gate_badge_html("G2 Mic Paired", g_states.get("G2", "NOT_REQUIRED"))
    led_state = snap.get("led", {})
    led_info = f"Left: {led_state.get('left', 'off')} | Right: {led_state.get('right', 'off')}"

    return f"""
    <div class='mc-card'>
        <div class='mc-card-header'>Verification Gates & Optical Telemetry</div>
        <div style='display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;'>
            {g1_html} {g2_html}
        </div>
        <div style='font-size: 11px; color: {TEXT_MUTED};'>
            Receiver LEDs:
            <span style='color: {ACCENT_CYAN}; font-family: monospace;'>{led_info}</span>
        </div>
    </div>
    """


def render_stepper_html(done_steps: list[str], expected_next_id: str | None) -> str:
    """Format the protocol progression checklist."""
    rows: list[str] = []
    for step in DEFAULT_MICROPHONE_PROTOCOL.steps:
        if step.id in done_steps:
            state_cls = "step-done"
            status_tag = "<span style='color: #10B981;'>✓ COMPLETED</span>"
        elif step.id == expected_next_id:
            state_cls = "step-active"
            status_tag = "<span style='color: #00F0FF;'>▶ ACTIVE</span>"
        else:
            state_cls = "step-pending"
            status_tag = "<span>PENDING</span>"

        rows.append(
            f"<div class='step-row {state_cls}'>"
            f"<span><b>{step.id}</b>: {step.name}</span>{status_tag}"
            f"</div>"
        )

    rows_content = "".join(rows)
    return f"""
    <div class='mc-card'>
        <div class='mc-card-header'>Protocol Progression (Deterministic FSM)</div>
        {rows_content}
    </div>
    """


def render_event_stream_html(history: list[str]) -> str:
    """Format the scrollable activity event stream."""
    lines = [f"<div class='log-line'>{entry}</div>" for entry in history[-10:]]
    content = "".join(lines) or '<span style="color: #64748B;">No events recorded yet.</span>'
    return f"<div class='log-stream'>{content}</div>"


__all__ = [
    "annotate_pipeline_frame",
    "format_gate_badge_html",
    "render_event_stream_html",
    "render_gates_panel_html",
    "render_step_card_html",
    "render_stepper_html",
    "render_telemetry_bar_html",
]
