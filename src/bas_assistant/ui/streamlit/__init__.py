"""Streamlit mission-control web presentation package."""

from __future__ import annotations

from bas_assistant.ui.streamlit.components import (
    render_gates_panel,
    render_stepper,
    render_telemetry_bar,
)
from bas_assistant.ui.streamlit.tabs import (
    render_architecture_tab,
    render_live_tab,
    render_metrics_tab,
)
from bas_assistant.ui.streamlit.theme import get_streamlit_css

__all__ = [
    "get_streamlit_css",
    "render_architecture_tab",
    "render_gates_panel",
    "render_live_tab",
    "render_metrics_tab",
    "render_stepper",
    "render_telemetry_bar",
]
