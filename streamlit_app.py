"""Streamlit Space-Exploration Mission Control Dashboard for BAS Experiment Assistant.

Matches the PySide6 dashboard (ADR-0004) space-exploration theme, telemetry cards,
G1/G2 verification gate badges, protocol stepper progression, and activity logging.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from bas_assistant.config.settings import load_settings
from bas_assistant.pipeline.factory import build_pipeline
from bas_assistant.ui.state import build_status_snapshot, gate_badge_states, step_display_label
from bas_assistant.ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    ACCENT_PURPLE,
    ACCENT_RED,
    BG_BASE,
    BG_CARD,
    BG_INPUT,
    BG_VOID,
    BORDER_CARD,
    BORDER_MUTED,
    EVENT_TYPE_COLORS,
    LED_STATE_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from bas_assistant.utils.logging import setup_logging
from bas_assistant.utils.visualization import annotate_frame
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL

st.set_page_config(
    page_title="ISRO BAS Experiment Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging(logging.INFO)

# =============================================================
# SPACE EXPLORATION MISSION CONTROL STYLING (matching theme.py)
# =============================================================
CUSTOM_CSS = f"""
<style>
    /* Dark space theme overrides */
    .stApp {{
        background-color: {BG_VOID};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {BG_BASE} !important;
        border-right: 1px solid {BORDER_CARD};
    }}

    /* Mission Control Card Containers */
    .mc-card {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_CARD};
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }}
    .mc-card-header {{
        color: {ACCENT_CYAN};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
        border-bottom: 1px solid {BORDER_MUTED};
        padding-bottom: 4px;
    }}

    /* Telemetry Chips */
    .telemetry-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 10px;
        margin-bottom: 16px;
    }}
    .telemetry-chip {{
        background: linear-gradient(180deg, {BG_CARD} 0%, #0c1424 100%);
        border: 1px solid {BORDER_CARD};
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        transition: border-color 0.2s;
    }}
    .telemetry-chip:hover {{
        border-color: {ACCENT_CYAN};
    }}
    .telemetry-label {{
        font-size: 10px;
        font-weight: 700;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 4px;
    }}
    .telemetry-value {{
        font-size: 20px;
        font-weight: 800;
        font-family: 'Consolas', 'Courier New', monospace;
        color: {TEXT_PRIMARY};
    }}

    /* Verification Gate Badges */
    .gate-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.8px;
        font-family: 'Consolas', monospace;
    }}
    .gate-passed {{
        background-color: rgba(16, 185, 129, 0.2);
        color: {ACCENT_EMERALD};
        border: 1px solid {ACCENT_EMERALD};
    }}
    .gate-pending {{
        background-color: rgba(245, 158, 11, 0.2);
        color: {ACCENT_AMBER};
        border: 1px solid {ACCENT_AMBER};
    }}
    .gate-not-required {{
        background-color: rgba(100, 116, 139, 0.15);
        color: {TEXT_MUTED};
        border: 1px solid {BORDER_MUTED};
    }}

    /* Stepper Progression Table */
    .step-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 10px;
        border-radius: 5px;
        margin-bottom: 4px;
        font-size: 12px;
        border-left: 3px solid transparent;
        background-color: rgba(11, 17, 32, 0.6);
    }}
    .step-done {{
        border-left-color: {ACCENT_EMERALD};
        color: {TEXT_SECONDARY};
    }}
    .step-active {{
        border-left-color: {ACCENT_CYAN};
        background-color: rgba(0, 240, 255, 0.12);
        color: {ACCENT_CYAN};
        font-weight: bold;
    }}
    .step-pending {{
        border-left-color: {BORDER_MUTED};
        color: {TEXT_MUTED};
    }}

    /* Event Log stream */
    .log-stream {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER_MUTED};
        border-radius: 6px;
        padding: 10px;
        height: 220px;
        overflow-y: auto;
        font-family: 'Consolas', monospace;
        font-size: 11px;
        line-height: 1.6;
    }}
    .log-line {{
        margin-bottom: 4px;
        word-break: break-all;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Build and cache the ML models and pipeline orchestrator with resilient fallback."""
    import urllib.request

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
                "Could not download MediaPipe hand model (%s). Falling back to dummy pose estimator.",
                exc,
            )
            settings.pose.model = "dummy"

    return build_pipeline(settings)


pipeline = get_pipeline()

# Session State Initialization
if "event_history" not in st.session_state:
    st.session_state.event_history = []
if "session_active" not in st.session_state:
    st.session_state.session_active = False

# Sidebar Configuration
st.sidebar.markdown(
    f"""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: {ACCENT_CYAN}; margin-bottom: 2px;'>ISRO BAS</h2>
        <span style='color: {TEXT_MUTED}; font-size: 11px; letter-spacing: 1px;'>MISSION CONTROL v0.1</span>
    </div>
    """,
    unsafe_allow_html=True,
)

input_mode = st.sidebar.radio(
    "Select Video Intake Source",
    ["Demo Video (data/raw)", "Upload Video File", "Browser Webcam (WebRTC)", "Offline Dummy Source"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='mc-card-header'>Protocol Steps (M0–M6)</div>", unsafe_allow_html=True)
for step in DEFAULT_MICROPHONE_PROTOCOL.steps:
    st.sidebar.markdown(
        f"<div style='font-size: 11px; color: {TEXT_SECONDARY}; margin-bottom: 3px;'>"
        f"<b style='color: {ACCENT_BLUE};'>{step.id}</b>: {step.name}</div>",
        unsafe_allow_html=True,
    )

if DEFAULT_MICROPHONE_PROTOCOL.gates:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div class='mc-card-header'>Verification Gates</div>", unsafe_allow_html=True)
    for gate in DEFAULT_MICROPHONE_PROTOCOL.gates:
        st.sidebar.markdown(
            f"<div style='font-size: 11px; color: {TEXT_SECONDARY}; margin-bottom: 3px;'>"
            f"<b style='color: {ACCENT_AMBER};'>{gate.id}</b>: {gate.name}</div>",
            unsafe_allow_html=True,
        )

# =============================================================
# MAIN LAYOUT: 3 TABS (Live Assistant, Metrics, Architecture)
# =============================================================
tab_live, tab_metrics, tab_architecture = st.tabs([
    "🚀 Live Mission Control",
    "📊 Model Performance & Curves",
    "🔀 Protocol FSM & Architecture Graph",
])


def _annotate(frame: np.ndarray, res) -> np.ndarray:
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


def _render_gate_badge_html(label: str, phase: str) -> str:
    css_class = {
        "PASSED": "gate-passed",
        "PENDING": "gate-pending",
    }.get(phase, "gate-not-required")
    return f"<span class='gate-badge {css_class}'>{label}: {phase}</span>"


def _render_stepper_html(done_steps: list[str], expected_next_id: str | None) -> str:
    rows_html = []
    for step in DEFAULT_MICROPHONE_PROTOCOL.steps:
        if step.id in done_steps:
            state_class = "step-done"
            status_tag = "<span style='color: #10B981;'>✓ COMPLETED</span>"
        elif step.id == expected_next_id:
            state_class = "step-active"
            status_tag = "<span style='color: #00F0FF;'>▶ ACTIVE</span>"
        else:
            state_class = "step-pending"
            status_tag = "<span>PENDING</span>"

        rows_html.append(
            f"<div class='step-row {state_class}'>"
            f"<span><b>{step.id}</b>: {step.name}</span>{status_tag}"
            f"</div>"
        )
    return "".join(rows_html)


# =============================================================
# TAB 1: LIVE MISSION CONTROL
# =============================================================
with tab_live:
    # Telemetry Row (Top Bar)
    telemetry_placeholder = st.empty()

    def update_telemetry_bar(snap: dict):
        telemetry_placeholder.markdown(
            f"""
            <div class='telemetry-row'>
                <div class='telemetry-chip'>
                    <div class='telemetry-label'>System Status</div>
                    <div class='telemetry-value' style='color: {ACCENT_EMERALD};'>{snap['status'].upper()}</div>
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
                    <div class='telemetry-value' style='color: {ACCENT_BLUE};'>{snap['step_id'] or 'NONE'}</div>
                </div>
                <div class='telemetry-chip'>
                    <div class='telemetry-label'>Confidence</div>
                    <div class='telemetry-value'>{snap['confidence'] * 100:.0f}%</div>
                </div>
                <div class='telemetry-chip'>
                    <div class='telemetry-label'>Gate Status</div>
                    <div class='telemetry-value' style='color: {ACCENT_AMBER}; font-size: 14px;'>{snap['gate_status']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Initial Telemetry View
    update_telemetry_bar({
        "status": "READY",
        "fps": 0.0,
        "latency_ms": 0.0,
        "step_id": None,
        "confidence": 0.0,
        "gate_status": "not_required",
    })

    col_video, col_telemetry = st.columns([1.7, 1.3])

    with col_telemetry:
        # 1. Experiment Header & Active Step
        step_card = st.empty()

        # 2. Verification Gates Panel (G1 / G2 + LEDs)
        gates_card = st.empty()

        # 3. Protocol Progression (M0–M6 Stepper)
        stepper_card = st.empty()

        # 4. Color-coded Activity Log
        st.markdown("<div class='mc-card-header'>Activity Log & Sequence Alerts</div>", unsafe_allow_html=True)
        log_stream_placeholder = st.empty()

        def render_right_panels(snap: dict):
            # Render Active Step
            step_name = snap["step"]
            step_id = snap["step_id"] or "--"
            step_card.markdown(
                f"""
                <div class='mc-card'>
                    <div class='mc-card-header'>Protocol Execution Status</div>
                    <div style='font-size: 16px; font-weight: 700; color: {TEXT_PRIMARY};'>
                        <span style='color: {ACCENT_CYAN};'>{step_id}</span>: {step_name}
                    </div>
                    <div style='margin-top: 8px;'>
                        <span style='font-size: 11px; color: {TEXT_MUTED};'>Classification Confidence: </span>
                        <b style='color: {ACCENT_EMERALD}; font-family: monospace;'>{snap['confidence'] * 100:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Render Verification Gates
            g_states = gate_badge_states(snap["gate_status"])
            g1_html = _render_gate_badge_html("G1 Receiver Connected", g_states.get("G1", "NOT_REQUIRED"))
            g2_html = _render_gate_badge_html("G2 Mic Paired", g_states.get("G2", "NOT_REQUIRED"))
            led_state = snap.get("led", {})
            led_status_str = f"Left: {led_state.get('left', 'off')} | Right: {led_state.get('right', 'off')}"

            gates_card.markdown(
                f"""
                <div class='mc-card'>
                    <div class='mc-card-header'>Verification Gates & Optical Telemetry</div>
                    <div style='display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;'>
                        {g1_html} {g2_html}
                    </div>
                    <div style='font-size: 11px; color: {TEXT_MUTED};'>
                        Receiver LEDs: <span style='color: {ACCENT_CYAN}; font-family: monospace;'>{led_status_str}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Render Stepper
            stepper_card.markdown(
                f"""
                <div class='mc-card'>
                    <div class='mc-card-header'>Protocol Progression (Deterministic FSM)</div>
                    {_render_stepper_html(snap['done_steps'], snap['expected_next_id'])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Render Log
            log_lines = []
            for ev in st.session_state.event_history[-10:]:
                log_lines.append(f"<div class='log-line'>{ev}</div>")
            log_stream_placeholder.markdown(
                f"<div class='log-stream'>{''.join(log_lines) or '<span style=\"color: #64748B;\">No events recorded yet.</span>'}</div>",
                unsafe_allow_html=True,
            )

        render_right_panels({
            "step": "Idle / Standby",
            "step_id": None,
            "confidence": 0.0,
            "gate_status": "not_required",
            "done_steps": [],
            "expected_next_id": "M0",
            "led": {"left": "off", "right": "off"},
        })

    def _process_video_stream(video_path: str | Path, frame_placeholder, stop_requested_fn):
        cap = cv2.VideoCapture(str(video_path))
        pipeline.start_session()
        frame_idx = 0

        while cap.isOpened():
            if stop_requested_fn():
                break
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            res = pipeline.process_frame(frame)
            snap = build_status_snapshot(pipeline, res, DEFAULT_MICROPHONE_PROTOCOL)

            # Update video frame
            annotated = _annotate(frame, res)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update event log
            if res.new_events:
                for ev in res.new_events:
                    color_tag, badge, _ = EVENT_TYPE_COLORS.get(ev.type, ("#94A3B8", "INFO", ""))
                    st.session_state.event_history.append(
                        f"<span style='color: {color_tag}; font-weight: bold;'>[{badge}]</span> "
                        f"<b>{ev.step or ''}</b> {ev.message} "
                        f"<span style='color: {TEXT_MUTED};'>({ev.timestamp:.1f}s)</span>"
                    )

            # Update UI panels
            update_telemetry_bar(snap)
            render_right_panels(snap)

        cap.release()
        pipeline.end_session()

    # -------------------------------------------------------------
    # Mode 1: Demo Video from data/raw/
    # -------------------------------------------------------------
    if input_mode == "Demo Video (data/raw)":
        with col_video:
            st.markdown("<div class='mc-card-header'>Video Viewport (data/raw)</div>", unsafe_allow_html=True)
            raw_dir = Path("data/raw")
            video_files = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.mov"))
            video_names = [f.name for f in video_files]

            if not video_files:
                st.warning("No video files found in `data/raw/`.")
            else:
                c_sel, c_ctrl = st.columns([1.5, 1])
                selected_name = c_sel.selectbox("Select Experiment Video", video_names, index=0)
                selected_path = raw_dir / selected_name

                c1, c2 = c_ctrl.columns(2)
                run_demo_btn = c1.button("▶️ START", type="primary")
                stop_btn = c2.checkbox("⏹️ STOP", value=False)
                frame_window = st.empty()

                if run_demo_btn:
                    _process_video_stream(selected_path, frame_window, lambda: stop_btn)

    # -------------------------------------------------------------
    # Mode 2: Upload Pre-recorded MP4 / AVI Video
    # -------------------------------------------------------------
    elif input_mode == "Upload Video File":
        with col_video:
            st.markdown("<div class='mc-card-header'>Video Viewport (Upload)</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload experiment video file", type=["mp4", "avi", "mov"]
            )
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())

                c1, c2 = st.columns(2)
                run_btn = c1.button("▶️ START ANALYSIS", type="primary")
                stop_btn = c2.checkbox("⏹️ STOP", value=False)
                frame_window = st.empty()

                if run_btn:
                    _process_video_stream(tfile.name, frame_window, lambda: stop_btn)

    # -------------------------------------------------------------
    # Mode 3: Browser Webcam (Streamlit Community Cloud & Remote)
    # -------------------------------------------------------------
    elif input_mode == "Browser Webcam (WebRTC)":
        with col_video:
            st.markdown("<div class='mc-card-header'>Live Webcam Stream</div>", unsafe_allow_html=True)

            class VideoTransformer:
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    res = pipeline.process_frame(img)
                    annotated = _annotate(img, res)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            webrtc_streamer(
                key="bas-streamer",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=VideoTransformer().recv,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

    # -------------------------------------------------------------
    # Mode 4: Offline Dummy Generator (Smoke Testing)
    # -------------------------------------------------------------
    else:
        with col_video:
            st.markdown("<div class='mc-card-header'>Simulated Synthetic Feed</div>", unsafe_allow_html=True)
            from bas_assistant.video.source import DummyVideoSource

            if st.button("▶️ Run 100-Frame Simulation", type="primary"):
                source = DummyVideoSource(num_frames=100)
                source.start()
                frame_window = st.empty()
                pipeline.start_session()

                for _ in range(100):
                    frame = source.read()
                    if frame is None:
                        break

                    res = pipeline.process_frame(frame)
                    snap = build_status_snapshot(pipeline, res, DEFAULT_MICROPHONE_PROTOCOL)
                    annotated = _annotate(frame, res)

                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                    update_telemetry_bar(snap)
                    render_right_panels(snap)

                pipeline.end_session()


# =============================================================
# TAB 2: MODEL PERFORMANCE & CURVES
# =============================================================
with tab_metrics:
    st.markdown("<div class='mc-card-header'>Model Training, Loss Curves & Evaluation Graphs</div>", unsafe_allow_html=True)
    st.write("Visualizations generated during YOLO object detection and XGBoost step classifier training.")

    runs_base = Path("runs/detect")
    candidate_dirs = [
        runs_base / "runs" / "microphone_yolo" / "baseline-2",
        runs_base / "val",
        runs_base / "val-2",
        runs_base / "val-3",
        runs_base / "val-4",
    ]
    available_dirs = [d for d in candidate_dirs if d.exists()]

    if not available_dirs:
        st.info("No training runs found in `runs/detect/`.")
    else:
        selected_run = st.selectbox(
            "Select Evaluation Checkpoint / Run",
            available_dirs,
            format_func=lambda p: str(p.relative_to(runs_base.parent) if runs_base.parent in p.parents else p),
        )

        st.markdown("---")

        # 1. Training & Loss Curves
        results_img = selected_run / "results.png"
        if results_img.exists():
            st.subheader("1. Training & Validation Metric Curves over Epochs")
            st.image(str(results_img), caption="YOLO Training Metrics (Loss, mAP50, mAP50-95, Precision, Recall)", use_container_width=True)

        # 2. Confusion Matrices
        cm_raw = selected_run / "confusion_matrix.png"
        cm_norm = selected_run / "confusion_matrix_normalized.png"
        if cm_raw.exists() or cm_norm.exists():
            st.subheader("2. Multi-Class Confusion Matrices")
            cm_col1, cm_col2 = st.columns(2)
            if cm_raw.exists():
                cm_col1.image(str(cm_raw), caption="Confusion Matrix (Raw Counts)", use_container_width=True)
            if cm_norm.exists():
                cm_col2.image(str(cm_norm), caption="Confusion Matrix (Normalized)", use_container_width=True)

        # 3. Precision, Recall & F1 Curves
        f1_img = selected_run / "BoxF1_curve.png"
        pr_img = selected_run / "BoxPR_curve.png"
        p_img = selected_run / "BoxP_curve.png"
        r_img = selected_run / "BoxR_curve.png"

        if any(p.exists() for p in [f1_img, pr_img, p_img, r_img]):
            st.subheader("3. Detection Performance Curves (PR, F1, Precision, Recall)")
            curve_col1, curve_col2 = st.columns(2)
            if pr_img.exists():
                curve_col1.image(str(pr_img), caption="Precision-Recall Curve (BoxPR)", use_container_width=True)
            if f1_img.exists():
                curve_col2.image(str(f1_img), caption="F1-Confidence Curve (BoxF1)", use_container_width=True)

            curve_col3, curve_col4 = st.columns(2)
            if p_img.exists():
                curve_col3.image(str(p_img), caption="Precision-Confidence Curve (BoxP)", use_container_width=True)
            if r_img.exists():
                curve_col4.image(str(r_img), caption="Recall-Confidence Curve (BoxR)", use_container_width=True)

        # 4. Labels & Ground Truth vs Predictions
        labels_img = selected_run / "labels.jpg"
        val_labels_img = selected_run / "val_batch0_labels.jpg"
        val_pred_img = selected_run / "val_batch0_pred.jpg"

        if labels_img.exists() or (val_labels_img.exists() and val_pred_img.exists()):
            st.subheader("4. Dataset Distribution & Batch Predictions")
            if labels_img.exists():
                st.image(str(labels_img), caption="Bounding Box Label Distribution & Spatial Density", use_container_width=True)

            if val_labels_img.exists() and val_pred_img.exists():
                b_col1, b_col2 = st.columns(2)
                b_col1.image(str(val_labels_img), caption="Ground Truth Labels (Validation Batch)", use_container_width=True)
                b_col2.image(str(val_pred_img), caption="Model Predictions (Validation Batch)", use_container_width=True)


# =============================================================
# TAB 3: PROTOCOL FSM & ARCHITECTURE GRAPH
# =============================================================
with tab_architecture:
    st.markdown("<div class='mc-card-header'>Finite State Machine (FSM) & Architecture Dataflow</div>", unsafe_allow_html=True)

    st.subheader("1. Protocol Sequence State Machine (FSM)")
    st.markdown(
        """
        The deterministic Finite State Machine validates that the astronaut performs experiment steps in order,
        verifying hardware connection gates before allowing progression:
        """
    )

    fsm_mermaid = """
    graph LR
        START([🚀 Session Start]) --> M0["M0: Verify Phone On"]
        M0 -->|confirmed| M1["M1: Move Phone"]
        M1 -->|confirmed| M2["M2: Pick Mic Case"]
        M2 -->|confirmed| M3["M3: Open Mic Case"]
        M3 -->|confirmed| M4["M4: Remove Receiver"]
        M4 -->|confirmed| M5["M5: Connect Receiver"]
        M5 -->|G1 Verified| G1{"Gate G1: Receiver Connected"}
        G1 -->|Gate Passed| M6["M6: Remove Microphone"]
        M6 -->|G2 Verified| G2{"Gate G2: Mic Paired"}
        G2 -->|Protocol Complete| END([✅ Experiment Complete])

        M2 -.->|Repeat / Skip| ERR([⚠️ Alert Event / Log])
        M4 -.->|Out of Sequence| ERR
        style G1 fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
        style G2 fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
        style END fill:#28a745,stroke:#333,stroke-width:2px,color:#fff
    """
    st.markdown(f"```mermaid\n{fsm_mermaid}\n```")

    st.markdown("---")

    st.subheader("2. End-to-End Pipeline Dataflow DAG")
    pipeline_mermaid = """
    graph TD
        A[📹 Video Source: Camera / MP4 / Dummy] --> B[Frame Intake & Normalization]
        B --> C[MediaPipe Hands: 21 Landmarks / Hand]
        B --> D[YOLO Detection: Payload Objects & Phone]
        C --> E[Feature Fusion Vector: Spatial + Temporal]
        D --> E
        E --> F[XGBoost Step Classifier]
        F --> G[Deterministic Protocol FSM]
        G --> H[Verification Gates G1/G2: LED & Pairing State]
        G --> I[Mission Control Dashboard & Telemetry]
        G --> J[JSONL Session Storage & Event Logger]
        style A fill:#4a90e2,stroke:#333,stroke-width:2px,color:#fff
        style F fill:#9013fe,stroke:#333,stroke-width:2px,color:#fff
        style G fill:#f5a623,stroke:#333,stroke-width:2px,color:#000
        style J fill:#7ed321,stroke:#333,stroke-width:2px,color:#000
    """
    st.markdown(f"```mermaid\n{pipeline_mermaid}\n```")
