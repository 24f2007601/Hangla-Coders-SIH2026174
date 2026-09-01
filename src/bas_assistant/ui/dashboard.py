"""PySide6 monitoring dashboard (ADR-0004) — Space-Exploration Mission Control UI.

Layout (matching wireframe & space-exploration theme):
- Left Column:
  - Live Video Feed (with HUD overlays for status, FPS, astronaut detection)
  - Mission Control Deck ([Play], [Pause], [Stop], Recording, Voice Alerts, Timer)
- Right Column:
  - Experiment Header Card (Protocol title, mission metadata, live status badge)
  - Protocol Step Navigator (Previous Step, Current Step with glow, Next Step, Confidence Gauge)
  - Activity Logs & Alerts (Color-coded events with auto-scroll)
- Top Bar: Telemetry metric chips (Status, FPS, Persons Detected, Inference Latency, Active Events)

The GUI reads pipeline output through Qt signals emitted by `PipelineWorker` (running in a QThread)
— no business logic lives in widgets and the pipeline never knows the GUI exists.
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
    StepNavigatorWidget,
    TelemetryChip,
    VideoFeedWidget,
)
from bas_assistant.utils.visualization import annotate_frame
from bas_assistant.validation.protocol import DEFAULT_TOY_PROTOCOL, ExperimentProtocol

logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Runs the video source + pipeline loop off the GUI thread."""

    frame_ready = Signal(np.ndarray)
    status_updated = Signal(dict)
    events_ready = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        pipeline: ExperimentPipeline,
        source: VideoSource,
        max_fps: int = 30,
        protocol: ExperimentProtocol = DEFAULT_TOY_PROTOCOL,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._source = source
        self._max_fps = max_fps
        self._protocol = protocol
        self._paused = False
        self._stopped = False

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

        self._pipeline.start_session()
        frame_interval = 1.0 / self._max_fps if self._max_fps > 0 else 0.0
        try:
            while not self._stopped:
                if self._paused:
                    time.sleep(0.02)
                    continue
                started = time.perf_counter()
                frame = self._source.read()
                if frame is None:
                    break
                result = self._pipeline.process_frame(frame)
                step_name = self._step_name(result.classification)
                step_id = result.classification.step if result.classification else None
                confidence = result.classification.confidence if result.classification else 0.0
                status = self._status_label(result)
                annotated = annotate_frame(
                    frame, result.pose, step_name, confidence, result.fps, status
                )
                self.frame_ready.emit(annotated)

                validator = getattr(self._pipeline, "_validator", None)
                done_steps = validator.done_steps if validator else []
                expected_next = (
                    validator.expected_next.name if validator and validator.expected_next else None
                )

                self.status_updated.emit(
                    {
                        "status": status,
                        "fps": result.fps,
                        "persons": 1 if result.person_id else 0,
                        "active_events": len(self._pipeline._event_manager.events),
                        "latency_ms": result.inference_latency_ms,
                        "step": step_name,
                        "step_id": step_id,
                        "confidence": confidence,
                        "done_steps": done_steps,
                        "expected_next": expected_next,
                        "is_complete": validator.is_complete if validator else False,
                    }
                )
                if result.new_events:
                    self.events_ready.emit([e.to_dict() for e in result.new_events])
                if frame_interval > 0:
                    elapsed = time.perf_counter() - started
                    time.sleep(max(0.0, frame_interval - elapsed))
        finally:
            self._pipeline.end_session()
            self._source.stop()

    def _step_name(self, classification) -> str:
        if classification is None:
            return "Idle / Standby"
        step = self._protocol.step(classification.step)
        return step.name if step else classification.step

    def _status_label(self, result) -> str:
        if result.has_error:
            return "error"
        if result.new_events:
            last = result.new_events[-1].type.replace("step_", "").replace("_", " ")
            return last
        return "monitoring"


class ExperimentHeaderCard(QFrame):
    """Card presenting protocol name, mission subtitle, and current state badge."""

    def __init__(
        self, protocol_name: str = "Sample Analysis Protocol", parent: QWidget | None = None
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
        protocol: ExperimentProtocol = DEFAULT_TOY_PROTOCOL,
    ) -> None:
        super().__init__()
        self._protocol = protocol
        self._worker = PipelineWorker(pipeline, source, max_fps=max_fps, protocol=protocol)
        self.setWindowTitle("BAS Experiment Assistant — Mission Control Monitor")
        self._build_ui()
        self._connect_signals()

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
        self._control_deck = ControlDeckWidget()

        left_col.addWidget(self._video_feed, 5)
        left_col.addWidget(self._control_deck, 1)

        # Right Column: Experiment Header + Step Navigator + Activity Logs
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        self._exp_header = ExperimentHeaderCard(
            f"{self._protocol.start.name.split()[0]} Sample Analysis Protocol"
        )
        self._step_navigator = StepNavigatorWidget(protocol=self._protocol)
        self._activity_log = ActivityLogWidget()

        right_col.addWidget(self._exp_header)
        right_col.addWidget(self._step_navigator, 3)
        right_col.addWidget(self._activity_log, 3)

        work_layout.addLayout(left_col, 5)
        work_layout.addLayout(right_col, 3)

        main_layout.addLayout(telemetry_bar)
        main_layout.addLayout(work_layout, 1)

        self.setCentralWidget(central)
        self.resize(1280, 760)
        self.setMinimumSize(1024, 640)

    def _connect_signals(self) -> None:
        self._control_deck.btn_start.clicked.connect(self._on_start)
        self._control_deck.btn_pause.clicked.connect(self._on_pause)
        self._control_deck.btn_stop.clicked.connect(self._on_stop)

        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_updated.connect(self._on_status)
        self._worker.events_ready.connect(self._on_events)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)

    # -- Event handlers -----------------------------------------------------

    def _on_start(self) -> None:
        if self._worker.isRunning():
            self._worker.resume()
            self._control_deck.btn_pause.setText("⏸  PAUSE")
            self._control_deck.set_running_state(True, paused=False)
            self._video_feed.set_live_state(True, paused=False)
            self._exp_header.set_status("MONITORING", ACCENT_EMERALD, "#064E3B")
        else:
            self._worker.start()
            self._control_deck.btn_start.setEnabled(False)
            self._control_deck.btn_pause.setEnabled(True)
            self._control_deck.btn_stop.setEnabled(True)
            self._control_deck.set_running_state(True, paused=False)
            self._video_feed.set_live_state(True, paused=False)
            self._chip_status.set_value("RUNNING", ACCENT_EMERALD)
            self._exp_header.set_status("MONITORING", ACCENT_EMERALD, "#064E3B")

    def _on_pause(self) -> None:
        if self._worker.isRunning():
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
        self._worker.request_stop()
        self._control_deck.btn_pause.setEnabled(False)
        self._control_deck.btn_stop.setEnabled(False)
        self._control_deck.set_running_state(False)
        self._chip_status.set_value("STOPPING", ACCENT_AMBER)
        self._exp_header.set_status("STOPPING", ACCENT_AMBER, "#78350F")

    def _on_finished(self) -> None:
        self._control_deck.btn_start.setEnabled(True)
        self._control_deck.btn_pause.setEnabled(False)
        self._control_deck.btn_stop.setEnabled(False)
        self._control_deck.btn_pause.setText("⏸  PAUSE")
        self._control_deck.set_running_state(False)
        self._video_feed.set_live_state(False)
        self._chip_status.set_value("STOPPED", TEXT_MUTED)
        self._exp_header.set_status("STANDBY", TEXT_MUTED, "#1E293B")

    def _on_failed(self, message: str) -> None:
        self._control_deck.btn_start.setEnabled(True)
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
        fps = values.get("fps", 0.0)
        persons = values.get("persons", 0)
        events_count = values.get("active_events", 0)
        latency = values.get("latency_ms", 0.0)
        step_name = str(values.get("step", "—"))
        step_id = values.get("step_id")
        confidence = float(values.get("confidence", 0.0))
        done_steps = values.get("done_steps", [])
        expected_next = values.get("expected_next")

        # Telemetry updates
        self._chip_fps.set_value(f"{fps:.1f}")
        self._chip_persons.set_value(str(persons))
        self._chip_events.set_value(str(events_count))
        self._chip_latency.set_value(f"{latency:.1f}")

        if "ALERT" in status_text or "SKIPPED" in status_text or "OUT" in status_text:
            self._chip_status.set_value(status_text, ACCENT_AMBER)
        elif status_text == "ERROR":
            self._chip_status.set_value(status_text, ACCENT_RED)
        else:
            self._chip_status.set_value(status_text, ACCENT_EMERALD)

        # Step Navigator updates
        self._step_navigator.update_steps(
            current_step_name=step_name,
            current_step_id=step_id,
            done_steps=done_steps,
            expected_next_name=expected_next,
        )
        self._step_navigator.update_confidence(confidence)

    def _on_events(self, events: list[dict]) -> None:
        for event in events:
            self._activity_log.add_event(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._worker.request_stop()
        self._worker.wait(3000)
        super().closeEvent(event)


def run_dashboard(pipeline: ExperimentPipeline, source: VideoSource, max_fps: int = 30) -> int:
    """Launch the PySide6 application with space theme applied (returns Qt's exit code)."""
    from bas_assistant.ui.theme import load_theme_fonts

    app = QApplication([])
    load_theme_fonts()
    app.setStyleSheet(get_qss_stylesheet())
    dashboard = Dashboard(pipeline, source, max_fps=max_fps)
    dashboard.show()
    return app.exec()


__all__ = ["Dashboard", "PipelineWorker", "run_dashboard"]
