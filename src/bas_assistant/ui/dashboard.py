"""PySide6 monitoring dashboard (ADR-0004) — functional skeleton, no heavy styling.

Layout: live video panel, status panel, activity panel, event log, and
START/PAUSE/STOP controls. The GUI reads pipeline output through signals emitted by
`PipelineWorker` (running in a QThread) — no business logic lives in widgets and the
pipeline never knows the GUI exists.

PySide6 is an optional dependency; importing this module fails with a helpful
message when it is not installed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bas_assistant.pipeline.pipeline import ExperimentPipeline
from bas_assistant.protocols import VideoSource
from bas_assistant.utils.visualization import annotate_frame
from bas_assistant.validation.protocol import DEFAULT_TOY_PROTOCOL

logger = logging.getLogger(__name__)

_EVENT_COLORS = {
    "step_confirmed": (0x1B, 0x7A, 0x33),
    "step_skipped": (0xE6, 0x6A, 0x00),
    "step_repeated": (0x7A, 0x5C, 0x2E),
    "out_of_sequence": (0xC6, 0x28, 0x28),
    "protocol_complete": (0x15, 0x65, 0xC0),
    "system_error": (0xC6, 0x28, 0x28),
}


def _event_color(event_type: str) -> tuple[int, int, int]:
    return _EVENT_COLORS.get(event_type, (0x21, 0x21, 0x21))


class PipelineWorker(QThread):
    """Runs the video source + pipeline loop off the GUI thread."""

    frame_ready = Signal(np.ndarray)
    status_updated = Signal(dict)
    events_ready = Signal(list)
    failed = Signal(str)

    def __init__(
        self, pipeline: ExperimentPipeline, source: VideoSource, max_fps: int = 30
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._source = source
        self._max_fps = max_fps
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
                confidence = result.classification.confidence if result.classification else 0.0
                status = self._status_label(result)
                annotated = annotate_frame(
                    frame, result.pose, step_name, confidence, result.fps, status
                )
                self.frame_ready.emit(annotated)
                self.status_updated.emit(
                    {
                        "status": status,
                        "fps": result.fps,
                        "persons": 1 if result.person_id else 0,
                        "active_events": len(self._pipeline._event_manager.events),
                        "latency_ms": result.inference_latency_ms,
                        "step": step_name,
                        "confidence": confidence,
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
            return "idle"
        step = DEFAULT_TOY_PROTOCOL.step(classification.step)
        return step.name if step else classification.step

    def _status_label(self, result) -> str:
        if result.has_error:
            return "error"
        if result.new_events:
            last = result.new_events[-1].type.replace("step_", "").replace("_", " ")
            return last
        return "monitoring"


class Dashboard(QMainWindow):
    """Functional dashboard: video + status + activity + event log + controls."""

    def __init__(
        self, pipeline: ExperimentPipeline, source: VideoSource, max_fps: int = 30
    ) -> None:
        super().__init__()
        self._worker = PipelineWorker(pipeline, source, max_fps=max_fps)
        self.setWindowTitle("BAS Experiment Assistant — Monitor")
        self._build_ui()
        self._connect_signals()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        self._video_label = QLabel("No video")
        self._video_label.setMinimumSize(640, 360)
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._status_values = {
            "status": QLabel("stopped"),
            "fps": QLabel("—"),
            "persons": QLabel("0"),
            "events": QLabel("0"),
            "latency": QLabel("—"),
        }
        status_form = QFormLayout()
        status_form.addRow("System Status", self._status_values["status"])
        status_form.addRow("FPS", self._status_values["fps"])
        status_form.addRow("Persons Detected", self._status_values["persons"])
        status_form.addRow("Active Events", self._status_values["events"])
        status_form.addRow("Inference Latency", self._status_values["latency"])
        status_box = QGroupBox("System Status")
        status_box.setLayout(status_form)

        self._activity_values = {
            "person": QLabel("—"),
            "activity": QLabel("—"),
            "confidence": QLabel("—"),
        }
        activity_form = QFormLayout()
        activity_form.addRow("Person", self._activity_values["person"])
        activity_form.addRow("Activity", self._activity_values["activity"])
        activity_form.addRow("Confidence", self._activity_values["confidence"])
        activity_box = QGroupBox("Activity")
        activity_box.setLayout(activity_form)

        self._start_button = QPushButton("START")
        self._pause_button = QPushButton("PAUSE")
        self._stop_button = QPushButton("STOP")
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        controls = QHBoxLayout()
        controls.addWidget(self._start_button)
        controls.addWidget(self._pause_button)
        controls.addWidget(self._stop_button)
        controls.addStretch(1)

        side = QVBoxLayout()
        side.addWidget(status_box)
        side.addWidget(activity_box)
        side.addLayout(controls)
        side.addStretch(1)

        self._event_log = QListWidget()
        self._event_log.setMaximumHeight(160)

        top = QHBoxLayout()
        top.addWidget(self._video_label, 3)
        top.addLayout(side, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top, 5)
        layout.addWidget(QLabel("Event Log"))
        layout.addWidget(self._event_log, 1)

        self.setCentralWidget(central)
        self.resize(1024, 640)

    def _connect_signals(self) -> None:
        self._start_button.clicked.connect(self._on_start)
        self._pause_button.clicked.connect(self._on_pause)
        self._stop_button.clicked.connect(self._on_stop)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status_updated.connect(self._on_status)
        self._worker.events_ready.connect(self._on_events)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)

    # -- Event handlers -----------------------------------------------------

    def _on_start(self) -> None:
        if self._worker.isRunning():
            self._worker.resume()
            self._pause_button.setText("PAUSE")
        else:
            self._worker.start()
        self._start_button.setEnabled(False)
        self._pause_button.setEnabled(True)
        self._stop_button.setEnabled(True)

    def _on_pause(self) -> None:
        if self._worker.isRunning():
            if self._pause_button.text() == "PAUSE":
                self._worker.pause()
                self._pause_button.setText("RESUME")
            else:
                self._worker.resume()
                self._pause_button.setText("PAUSE")

    def _on_stop(self) -> None:
        self._worker.request_stop()
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._status_values["status"].setText("stopping")

    def _on_finished(self) -> None:
        self._start_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._status_values["status"].setText("stopped")

    def _on_failed(self, message: str) -> None:
        self._start_button.setEnabled(True)
        self._status_values["status"].setText("error")
        self._add_event(
            {
                "type": "system_error",
                "message": f"Video source failed: {message}",
                "timestamp": time.time(),
            }
        )

    def _on_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self._video_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_status(self, values: dict[str, Any]) -> None:
        self._status_values["status"].setText(str(values["status"]))
        self._status_values["fps"].setText(f"{values['fps']:.1f}")
        self._status_values["persons"].setText(str(values["persons"]))
        self._status_values["events"].setText(str(values["active_events"]))
        self._status_values["latency"].setText(f"{values['latency_ms']:.1f} ms")
        self._activity_values["person"].setText(f"Person {values['persons']}")
        self._activity_values["activity"].setText(str(values["step"]))
        self._activity_values["confidence"].setText(f"{values['confidence']:.2f}")

    def _on_events(self, events: list[dict]) -> None:
        for event in events:
            self._add_event(event)

    def _add_event(self, event: dict[str, Any]) -> None:
        message = event.get("message") or event.get("type", "")
        timestamp = event.get("timestamp", 0.0)
        r, g, b = _event_color(event.get("type", ""))
        item = QListWidgetItem(f"{time.strftime('%H:%M:%S', time.localtime(timestamp))}  {message}")
        item.setForeground(QBrush(QColor(r, g, b)))
        self._event_log.insertItem(0, item)
        while self._event_log.count() > 200:
            self._event_log.takeItem(self._event_log.count() - 1)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._worker.request_stop()
        self._worker.wait(3000)
        super().closeEvent(event)


def run_dashboard(pipeline: ExperimentPipeline, source: VideoSource, max_fps: int = 30) -> int:
    """Launch the PySide6 application (returns Qt's exit code)."""
    app = QApplication([])
    dashboard = Dashboard(pipeline, source, max_fps=max_fps)
    dashboard.show()
    return app.exec()


__all__ = ["Dashboard", "run_dashboard"]
