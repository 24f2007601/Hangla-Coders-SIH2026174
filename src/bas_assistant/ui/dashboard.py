"""PySide6 monitoring dashboard (ADR-0004) — Space-Exploration Mission Control UI.

Layout (matching wireframe & space-exploration theme):
- Left Column:
  - Live Video Feed (with HUD overlays for status, FPS, astronaut detection)
  - Mission Control Deck ([Play], [Pause], [Stop], [Reset], Recording, Voice Alerts, Timer)
- Right Column:
  - Experiment Header Card (Protocol title, mission metadata, live status badge)
  - Protocol Progression (M0-M6 with G1/G2 interleaved, completed/active/pending)
  - Verification Gates (G1/G2 badges, receiver LEDs, receiver detection)
  - Activity Logs & Alerts (Color-coded events with auto-scroll)
- Top Bar: Telemetry metric chips (Status, FPS, Persons Detected, Inference Latency, Events)

The GUI reads pipeline output through Qt signals emitted by `PipelineWorker` (running
in a QThread) — no business logic lives in widgets and the pipeline never knows the
GUI exists. All displayed state (steps, gates, LEDs, receiver, protocol completion)
is assembled from public pipeline properties by `bas_assistant.ui.state`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from bas_assistant.pipeline.pipeline import ExperimentPipeline
from bas_assistant.protocols import VideoSource
from bas_assistant.ui.state import build_status_snapshot, status_label, step_display_label
from bas_assistant.ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    ACCENT_RED,
    BG_CARD,
    BORDER_CARD,
    BORDER_MUTED,
    TEXT_MUTED,
    TEXT_PRIMARY,
    get_qss_stylesheet,
)
from bas_assistant.ui.widgets import (
    ActivityLogWidget,
    ControlDeckWidget,
    ProtocolProgressWidget,
    TelemetryChip,
    VerificationGatesWidget,
    VideoFeedWidget,
)
from bas_assistant.utils.visualization import annotate_frame
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL, ExperimentProtocol

logger = logging.getLogger(__name__)


def describe_source(source: VideoSource) -> str:
    """Short human-readable source kind for the video-feed HUD."""
    if hasattr(source, "num_frames"):  # DummyVideoSource
        return "SIMULATED"
    device = getattr(source, "_device", None)
    if isinstance(device, int):
        return f"WEBCAM {device}"
    if isinstance(device, str) and ("/" in device or "\\" in device or "." in device):
        return "FILE"
    return "SOURCE"


class PipelineWorker(QThread):
    """Runs the video source + pipeline loop off the GUI thread.

    Emits an annotated frame, a full live-state snapshot, and any new events.
    A fresh worker must be created per session (QThread objects are not reused).
    """

    frame_ready = Signal(np.ndarray)
    status_updated = Signal(dict)
    events_ready = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        pipeline: ExperimentPipeline,
        source: VideoSource,
        max_fps: int = 30,
        protocol: ExperimentProtocol = DEFAULT_MICROPHONE_PROTOCOL,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._source = source
        self._max_fps = max_fps
        self._protocol = protocol
        self._paused = False
        self._stopped = False
        self._total_events = 0
        self._source_kind = describe_source(source)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def request_stop(self) -> None:
        self._stopped = True

    def run(self) -> None:
        from bas_assistant.video.source import VideoSourceError

        try:
            self._source.start()
        except VideoSourceError as exc:
            self.failed.emit(str(exc))
            return

        session_id = self._pipeline.start_session()
        logger.info("Worker session %s started (source: %s)", session_id, self._source_kind)

        frame_interval = 1.0 / self._max_fps if self._max_fps > 0 else 0.0
        try:
            while not self._stopped:
                if self._paused:
                    time.sleep(0.02)
                    continue
                started = time.perf_counter()
                frame = self._source.read()
                if frame is None:
                    logger.info("Video source exhausted (no more frames)")
                    break
                # The source timestamp drives LED blink timing (G1/G2 windows).
                result = self._pipeline.process_frame(
                    frame, source_timestamp=self._source.timestamp
                )
                self._total_events += len(result.new_events)

                step_name = step_display_label(
                    result.classification.step if result.classification else None,
                    self._protocol,
                )
                confidence = result.classification.confidence if result.classification else 0.0
                annotated = annotate_frame(
                    frame, result.pose, step_name, confidence, result.fps, status_label(result)
                )
                self.frame_ready.emit(annotated)

                self.status_updated.emit(
                    build_status_snapshot(
                        self._pipeline,
                        result,
                        self._protocol,
                        active_events=self._total_events,
                        source_kind=self._source_kind,
                    )
                )
                if result.new_events:
                    self.events_ready.emit([e.to_dict() for e in result.new_events])
                if frame_interval > 0:
                    elapsed = time.perf_counter() - started
                    time.sleep(max(0.0, frame_interval - elapsed))
        finally:
            self._pipeline.end_session()
            self._source.stop()
            logger.info("Worker loop finished")


class ExperimentHeaderCard(QFrame):
    """Card presenting protocol name, mission subtitle, and current state badge."""

    def __init__(
        self,
        protocol_name: str = "Wireless Microphone Experiment Protocol",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            ExperimentHeaderCard {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_CARD};
                border-radius: 8px;
            }}
            """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        icon_label = QLabel("BAS-ISRO")
        icon_label.setStyleSheet(
            f"font-size: 11px; font-weight: 800; color: {ACCENT_CYAN}; "
            f"background: #0E243A; padding: 4px 8px; border-radius: 4px; "
            f"border: 1px solid {BORDER_MUTED};"
        )

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self._lbl_protocol = QLabel(protocol_name)
        self._lbl_protocol.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 700;"
        )

        self._lbl_sub = QLabel("ISRO BAS EXPERIMENT ASSISTANT • PROTOCOL HAR")
        self._lbl_sub.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )

        text_layout.addWidget(self._lbl_protocol)
        text_layout.addWidget(self._lbl_sub)

        self._status_badge = QLabel("STANDBY")
        self._status_badge.setStyleSheet(f"""
            background-color: #1E293B;
            color: {TEXT_MUTED};
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1px;
            border: 1px solid {BORDER_MUTED};
            """)

        layout.addWidget(icon_label)
        layout.addLayout(text_layout, 1)
        layout.addWidget(self._status_badge)

    def set_status(self, text: str, color: str = ACCENT_CYAN, bg: str = "#0E243A") -> None:
        self._status_badge.setText(text.upper())
        self._status_badge.setStyleSheet(f"""
            background-color: {bg};
            color: {color};
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1px;
            border: 1px solid {color};
            """)


class Dashboard(QMainWindow):
    """Space-exploration mission control dashboard for BAS Experiment Assistant."""

    def __init__(
        self,
        pipeline: ExperimentPipeline,
        source: VideoSource,
        max_fps: int = 30,
        protocol: ExperimentProtocol = DEFAULT_MICROPHONE_PROTOCOL,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._source = source
        self._protocol = protocol
        self._max_fps = max_fps
        self._worker: PipelineWorker | None = None
        self._last_frame_error: str | None = None
        self._source_failed = False
        self._source_label = describe_source(source)
        self.setWindowTitle("BAS Experiment Assistant — Mission Control Monitor")
        self._build_ui()
        self._connect_signals()
        self._apply_standby_state()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        # Central widget container
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(12)

        # 1. Top Telemetry Bar
        telemetry_bar = QHBoxLayout()
        telemetry_bar.setSpacing(8)

        self._chip_status = TelemetryChip("SYSTEM STATUS", "STOPPED", accent_color=TEXT_MUTED)
        self._chip_fps = TelemetryChip("FRAME RATE", "0.0", "FPS", accent_color=ACCENT_CYAN)
        self._chip_persons = TelemetryChip("PERSONS", "0", "ASTRONAUT", accent_color=ACCENT_BLUE)
        self._chip_latency = TelemetryChip("INFERENCE", "—", "ms", accent_color=ACCENT_CYAN)
        self._chip_events = TelemetryChip("TOTAL EVENTS", "0", accent_color=ACCENT_EMERALD)

        telemetry_bar.addWidget(self._chip_status)
        telemetry_bar.addWidget(self._chip_fps)
        telemetry_bar.addWidget(self._chip_persons)
        telemetry_bar.addWidget(self._chip_latency)
        telemetry_bar.addWidget(self._chip_events)

        # 2. Central 2-Column Work Area
        work_layout = QHBoxLayout()
        work_layout.setSpacing(12)

        # Left Column: Video Feed (Top) + Control Bar (Bottom)
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        self._video_feed = VideoFeedWidget()
        self._video_feed.set_source_label(self._source_label)
        self._control_deck = ControlDeckWidget()

        left_col.addWidget(self._video_feed, 5)
        left_col.addWidget(self._control_deck, 1)

        # Right Column: Header + Protocol Progress + Verification Gates + Logs
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        self._exp_header = ExperimentHeaderCard(
            f"{self._protocol.start.name.upper()} MICROPHONE PROTOCOL"
        )
        self._protocol_progress = ProtocolProgressWidget(protocol=self._protocol)
        self._verification_gates = VerificationGatesWidget(protocol=self._protocol)
        self._activity_log = ActivityLogWidget()

        right_col.addWidget(self._exp_header)
        right_col.addWidget(self._protocol_progress, 3)
        right_col.addWidget(self._verification_gates, 2)
        right_col.addWidget(self._activity_log, 3)

        work_layout.addLayout(left_col, 5)
        work_layout.addLayout(right_col, 3)

        main_layout.addLayout(telemetry_bar)
        main_layout.addLayout(work_layout, 1)

        self.setCentralWidget(central)
        self.resize(1280, 820)
        self.setMinimumSize(1024, 700)

    def _connect_signals(self) -> None:
        self._control_deck.btn_start.clicked.connect(self._on_start)
        self._control_deck.btn_pause.clicked.connect(self._on_pause)
        self._control_deck.btn_stop.clicked.connect(self._on_stop)
        self._control_deck.btn_reset.clicked.connect(self._on_reset)

    # -- Worker lifecycle ----------------------------------------------------

    def _create_worker(self) -> PipelineWorker:
        """Create a fresh worker (one per session; QThreads are not reusable)."""
        worker = PipelineWorker(
            self._pipeline, self._source, max_fps=self._max_fps, protocol=self._protocol
        )
        worker.frame_ready.connect(self._on_frame)
        worker.status_updated.connect(self._on_status)
        worker.events_ready.connect(self._on_events)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        return worker

    def _apply_standby_state(self) -> None:
        self._control_deck.btn_start.setEnabled(True)
        self._control_deck.btn_pause.setEnabled(False)
        self._control_deck.btn_stop.setEnabled(False)
        self._control_deck.btn_reset.setEnabled(False)
        self._control_deck.set_running_state(False)
        self._video_feed.set_live_state(False)
        self._chip_status.set_value("STOPPED", TEXT_MUTED)
        self._exp_header.set_status("STANDBY", TEXT_MUTED, "#1E293B")

    # -- Event handlers -----------------------------------------------------

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.resume()
            self._control_deck.btn_pause.setText("⏸  PAUSE")
            self._control_deck.set_running_state(True, paused=False)
            self._video_feed.set_live_state(True, paused=False)
            self._chip_status.set_value("RUNNING", ACCENT_EMERALD)
            self._exp_header.set_status("MONITORING", ACCENT_EMERALD, "#064E3B")
            return

        # Fresh session: brand-new worker; pipeline.start_session() resets the
        # FSM, vote buffers, LED history and gate state.
        self._last_frame_error = None
        self._source_failed = False
        self._worker = self._create_worker()
        self._worker.start()
        self._control_deck.btn_start.setEnabled(False)
        self._control_deck.btn_pause.setEnabled(True)
        self._control_deck.btn_stop.setEnabled(True)
        self._control_deck.btn_reset.setEnabled(True)
        self._control_deck.set_running_state(True, paused=False)
        self._video_feed.set_live_state(True, paused=False)
        self._chip_status.set_value("RUNNING", ACCENT_EMERALD)
        self._exp_header.set_status("MONITORING", ACCENT_EMERALD, "#064E3B")

    def _on_pause(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        if "PAUSE" in self._control_deck.btn_pause.text():
            self._worker.pause()
            self._control_deck.btn_pause.setText("▶  RESUME")
            self._control_deck.set_running_state(True, paused=True)
            self._video_feed.set_live_state(True, paused=True)
            self._chip_status.set_value("PAUSED", ACCENT_AMBER)
            self._exp_header.set_status("PAUSED", ACCENT_AMBER, "#78350F")
        else:
            self._worker.resume()
            self._control_deck.btn_pause.setText("⏸  PAUSE")
            self._control_deck.set_running_state(True, paused=False)
            self._video_feed.set_live_state(True, paused=False)
            self._chip_status.set_value("RUNNING", ACCENT_EMERALD)
            self._exp_header.set_status("MONITORING", ACCENT_EMERALD, "#064E3B")

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
        self._control_deck.btn_pause.setEnabled(False)
        self._control_deck.btn_stop.setEnabled(False)
        self._control_deck.set_running_state(False)
        self._chip_status.set_value("STOPPING", ACCENT_AMBER)
        self._exp_header.set_status("STOPPING", ACCENT_AMBER, "#78350F")

    def _on_reset(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        self._worker = None
        self._last_frame_error = None
        self._source_failed = False

        self._protocol_progress.reset_progress()
        self._verification_gates.reset_gates()
        self._activity_log.clear_events()
        self._chip_fps.set_value("0.0")
        self._chip_persons.set_value("0")
        self._chip_latency.set_value("—")
        self._chip_events.set_value("0")
        self._apply_standby_state()

    def _on_finished(self) -> None:
        self._control_deck.btn_start.setEnabled(True)
        self._control_deck.btn_pause.setEnabled(False)
        self._control_deck.btn_stop.setEnabled(False)
        self._control_deck.btn_pause.setText("⏸  PAUSE")
        self._control_deck.set_running_state(False)
        self._video_feed.set_live_state(False)
        if self._source_failed:
            # A source failure surfaced: keep the ERROR display (set by
            # _on_failed) instead of flashing back to STANDBY.
            return
        self._chip_status.set_value("STOPPED", TEXT_MUTED)
        self._exp_header.set_status("STANDBY", TEXT_MUTED, "#1E293B")

    def _on_failed(self, message: str) -> None:
        self._source_failed = True
        self._control_deck.btn_start.setEnabled(True)
        self._control_deck.btn_reset.setEnabled(True)
        self._control_deck.set_running_state(False)
        self._video_feed.set_live_state(False)
        self._chip_status.set_value("ERROR", ACCENT_RED)
        self._exp_header.set_status("ERROR", ACCENT_RED, "#7F1D1D")
        self._activity_log.add_event(
            {
                "type": "system_error",
                "message": f"Video source failed: {message}",
                "timestamp": time.time(),
            }
        )

    def _on_frame(self, frame: np.ndarray) -> None:
        self._video_feed.update_frame(frame)

    def _on_status(self, values: dict[str, Any]) -> None:
        status_text = str(values.get("status", "—")).upper()
        fps = float(values.get("fps", 0.0))
        persons = int(values.get("persons", 0))
        events_count = int(values.get("active_events", 0))
        latency = float(values.get("latency_ms", 0.0))
        confidence = float(values.get("confidence", 0.0))

        # Telemetry updates
        self._chip_fps.set_value(f"{fps:.1f}")
        self._chip_persons.set_value(str(persons))
        self._chip_events.set_value(str(events_count))
        self._chip_latency.set_value(f"{latency:.1f}")

        if values.get("frame_error"):
            self._chip_status.set_value("ERROR", ACCENT_RED)
        elif "ALERT" in status_text or "SKIPPED" in status_text or "OUT" in status_text:
            self._chip_status.set_value(status_text, ACCENT_AMBER)
        elif status_text == "ERROR":
            self._chip_status.set_value(status_text, ACCENT_RED)
        else:
            self._chip_status.set_value(status_text, ACCENT_EMERALD)

        # Protocol progression + gates (all state comes from the backend)
        self._protocol_progress.update_progress(
            done_steps=values.get("done_steps", []),
            expected_next_id=values.get("expected_next_id"),
            gate_status=str(values.get("gate_status", "not_required")),
            is_complete=bool(values.get("is_complete", False)),
        )
        self._protocol_progress.update_confidence(confidence)
        self._verification_gates.update_gates(
            gate_status=str(values.get("gate_status", "not_required")),
            led=values.get("led", {}),
        )

        # Frame errors: log each distinct failure once (no per-frame spam).
        frame_error = values.get("frame_error")
        if frame_error and frame_error != self._last_frame_error:
            self._last_frame_error = frame_error
            self._activity_log.add_event(
                {
                    "type": "system_error",
                    "message": f"Frame {values.get('frame_number')}: {frame_error}",
                    "timestamp": time.time(),
                }
            )

        # Protocol completion indicator
        if values.get("is_complete"):
            self._exp_header.set_status("PROTOCOL COMPLETE", ACCENT_CYAN, "#0E243A")

    def _on_events(self, events: list[dict]) -> None:
        for event in events:
            self._activity_log.add_event(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        super().closeEvent(event)


def run_dashboard(
    pipeline: ExperimentPipeline,
    source: VideoSource,
    max_fps: int = 30,
    protocol: ExperimentProtocol = DEFAULT_MICROPHONE_PROTOCOL,
) -> int:
    """Launch the PySide6 application with space theme applied (returns Qt's exit code)."""
    from bas_assistant.ui.theme import load_theme_fonts

    app = QApplication([])
    load_theme_fonts()
    app.setStyleSheet(get_qss_stylesheet())
    dashboard = Dashboard(pipeline, source, max_fps=max_fps, protocol=protocol)
    dashboard.show()
    return app.exec()


__all__ = ["Dashboard", "PipelineWorker", "run_dashboard"]
