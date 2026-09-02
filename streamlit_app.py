"""Streamlit Mission-Control Entrypoint for BAS Experiment Assistant."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import streamlit as st

from bas_assistant.config.settings import load_settings
from bas_assistant.pipeline.factory import build_pipeline
from bas_assistant.ui.streamlit import (
    get_streamlit_css,
    render_architecture_tab,
    render_live_tab,
    render_metrics_tab,
)
from bas_assistant.ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    TEXT_MUTED,
    TEXT_SECONDARY,
)
from bas_assistant.utils.logging import setup_logging
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL

st.set_page_config(
    page_title="ISRO BAS Experiment Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging(logging.INFO)
st.markdown(get_streamlit_css(), unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Build and cache the ML models and pipeline orchestrator with resilient fallback."""
    settings = load_settings()
    model_path = Path(settings.pose.hand_model_path)

    if settings.pose.model == "mediapipe" and not model_path.exists():
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        try:
            logging.info("Downloading missing MediaPipe hand model to %s...", model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(model_url, model_path)
            logging.info("MediaPipe model successfully downloaded.")
        except Exception as exc:
            logging.warning(
                "Could not download MediaPipe hand model (%s). Fallback to dummy.",
                exc,
            )
            settings.pose.model = "dummy"

    return build_pipeline(settings)


pipeline = get_pipeline()

if "event_history" not in st.session_state:
    st.session_state.event_history = []

# Sidebar Navigation & Protocol Guide
sub_title = "MISSION CONTROL v0.1"
st.sidebar.markdown(
    f"""
    <div style='text-align: center; margin-bottom: 18px;'>
        <h2 style='color: {ACCENT_CYAN}; margin-bottom: 2px;'>ISRO BAS</h2>
        <span style='color: {TEXT_MUTED}; font-size: 11px; letter-spacing: 1px;'>{sub_title}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

input_mode = st.sidebar.radio(
    "Select Video Intake Source",
    [
        "Demo Video (data/raw)",
        "Upload Video File",
        "Browser Webcam (WebRTC)",
        "Offline Dummy Source",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='mc-card-header'>Protocol Steps (M0–M6)</div>", unsafe_allow_html=True
)
for step in DEFAULT_MICROPHONE_PROTOCOL.steps:
    st.sidebar.markdown(
        f"<div style='font-size: 11px; color: {TEXT_SECONDARY}; margin-bottom: 3px;'>"
        f"<b style='color: {ACCENT_BLUE};'>{step.id}</b>: {step.name}</div>",
        unsafe_allow_html=True,
    )

if DEFAULT_MICROPHONE_PROTOCOL.gates:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div class='mc-card-header'>Verification Gates</div>", unsafe_allow_html=True
    )
    for gate in DEFAULT_MICROPHONE_PROTOCOL.gates:
        st.sidebar.markdown(
            f"<div style='font-size: 11px; color: {TEXT_SECONDARY}; margin-bottom: 3px;'>"
            f"<b style='color: {ACCENT_AMBER};'>{gate.id}</b>: {gate.name}</div>",
            unsafe_allow_html=True,
        )

# Main Navigation Tabs
tab_live, tab_metrics, tab_architecture = st.tabs(
    [
        "🚀 Live Mission Control",
        "📊 Model Performance & Curves",
        "🔀 Protocol FSM & Architecture Graph",
    ]
)

with tab_live:
    render_live_tab(pipeline, input_mode, st.session_state.event_history)

with tab_metrics:
    render_metrics_tab()

with tab_architecture:
    render_architecture_tab()
