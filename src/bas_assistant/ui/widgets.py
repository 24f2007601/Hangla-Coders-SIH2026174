"""Custom space-exploration UI widgets for the BAS Experiment Assistant dashboard.

Implements:
- `VideoFeedWidget`: Live camera viewport with HUD overlays.
- `StepNavigatorWidget`: Protocol step tracker (Previous, Current with glow, Next).
- `ActivityLogWidget`: Color-coded activity and alert log stream with status badges.
- `TelemetryChip`: High-contrast metric cards with tabular numerical displays.
- `ControlDeckWidget`: Mission control bar with START/PAUSE/STOP and toggles.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from bas_assistant.ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    ACCENT_RED,
    BG_BASE,
    BG_CARD,
    BG_VOID,
    BORDER_CARD,
    BORDER_MUTED,
    EVENT_TYPE_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from bas_assistant.validation.protocol import DEFAULT_TOY_PROTOCOL, ExperimentProtocol


class TelemetryChip(QFrame):
    """A sleek metric card displaying a label and glowing value."""

    def __init__(
        self,
        label: str,
        initial_value: str = "--",
        unit: str = "",
        accent_color: str = ACCENT_CYAN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._accent_color = accent_color
        self._unit = unit

        self.setStyleSheet(
            f"TelemetryChip {{"
            f"  background-color: {BG_CARD};"
            f"  border: 1px solid {BORDER_CARD};"
            f"  border-radius: 6px;"
            f"  padding: 6px 10px;"
            f"  min-width: 90px;"
            f"}}"
            f"TelemetryChip:hover {{"
            f"  border: 1px solid {accent_color};"
            f"  background-color: #15223C;"
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self._lbl_title = QLabel(label.upper())
        self._lbl_title.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )

        val_layout = QHBoxLayout()
        val_layout.setContentsMargins(0, 0, 0, 0)
        val_layout.setSpacing(4)

        self._lbl_value = QLabel(initial_value)
        self._lbl_value.setStyleSheet(
            f"color: {accent_color}; font-size: 14px; font-weight: 700; "
            f"font-family: 'Consolas', 'Segoe UI', monospace;"
        )

        val_layout.addWidget(self._lbl_value)
        if unit:
            self._lbl_unit = QLabel(unit)
            self._lbl_unit.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
            val_layout.addWidget(self._lbl_unit)
        val_layout.addStretch(1)

        layout.addWidget(self._lbl_title)
        layout.addLayout(val_layout)

    def set_value(self, value: str, custom_color: str | None = None) -> None:
        self._lbl_value.setText(value)
        if custom_color:
            self._lbl_value.setStyleSheet(
                f"color: {custom_color}; font-size: 14px; font-weight: 700; "
                f"font-family: 'Consolas', 'Segoe UI', monospace;"
            )


class VideoFeedWidget(QFrame):
    """High-tech video container with HUD overlays and standby graphic."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"VideoFeedWidget {{"
            f"  background-color: {BG_VOID};"
            f"  border: 1px solid {BORDER_CARD};"
            f"  border-radius: 8px;"
            f"}}"
        )

        self._is_live = False
        self._is_paused = False
        self._last_frame: QPixmap | None = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(0)

        # Top HUD Bar
        hud_bar = QHBoxLayout()
        hud_bar.setContentsMargins(12, 8, 12, 4)

        self._status_pill = QLabel("OFF AIR")
        self._status_pill.setStyleSheet(
            f"background-color: #1F2937; color: {TEXT_MUTED}; padding: 4px 10px; "
            f"border-radius: 12px; font-size: 11px; font-weight: 700; letter-spacing: 1px; "
            f"border: 1px solid #374151;"
        )

        self._feed_title = QLabel("CAMERA FEED -- SENSOR RIG 01")
        self._feed_title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600; letter-spacing: 1.5px;"
        )

        self._fps_hud = QLabel("0.0 FPS")
        self._fps_hud.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 700; font-family: monospace;"
        )

        hud_bar.addWidget(self._status_pill)
        hud_bar.addWidget(self._feed_title)
        hud_bar.addStretch(1)
        hud_bar.addWidget(self._fps_hud)

        # Video Canvas
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setMinimumSize(480, 270)
        self._render_standby_placeholder()

        main_layout.addLayout(hud_bar)
        main_layout.addWidget(self._video_label, 1)

    def set_live_state(self, live: bool, paused: bool = False) -> None:
        self._is_live = live
        self._is_paused = paused
        if not live:
            self._status_pill.setText("OFF AIR")
            self._status_pill.setStyleSheet(
                f"background-color: #1E293B; color: {TEXT_MUTED}; padding: 4px 10px; "
                "border-radius: 12px; font-size: 11px; font-weight: 700; "
                "border: 1px solid #334155;"
            )
            self._fps_hud.setText("-- FPS")
            self._render_standby_placeholder()
        elif paused:
            self._status_pill.setText("PAUSED")
            self._status_pill.setStyleSheet(
                f"background-color: #78350F; color: #FEF08A; padding: 4px 10px; "
                f"border-radius: 12px; font-size: 11px; font-weight: 700; "
                f"border: 1px solid {ACCENT_AMBER};"
            )
        else:
            self._status_pill.setText("LIVE MONITOR")
            self._status_pill.setStyleSheet(
                f"background-color: #064E3B; color: #A7F3D0; padding: 4px 10px; "
                f"border-radius: 12px; font-size: 11px; font-weight: 700; "
                f"border: 1px solid {ACCENT_EMERALD};"
            )

    def update_frame(self, frame: np.ndarray, fps: float = 0.0) -> None:
        self._is_live = True
        self._fps_hud.setText(f"{fps:.1f} FPS")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self._last_frame = pixmap
        self._scale_and_set_pixmap(pixmap)

    def _scale_and_set_pixmap(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._is_live and self._last_frame is not None:
            self._scale_and_set_pixmap(self._last_frame)
        elif not self._is_live:
            self._render_standby_placeholder()

    def _render_standby_placeholder(self) -> None:
        width = max(self._video_label.width(), 480)
        height = max(self._video_label.height(), 270)
        pix = QPixmap(width, height)
        pix.fill(QColor(BG_VOID))

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw subtle grid
        grid_pen = QPen(QColor(BORDER_CARD))
        grid_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        grid_step = 40
        for x in range(0, width, grid_step):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, grid_step):
            painter.drawLine(0, y, width, y)

        # Draw central reticle / orbit crosshair
        center_x, center_y = width // 2, height // 2
        painter.setPen(QPen(QColor(BORDER_MUTED), 1))
        painter.drawEllipse(QPoint(center_x, center_y), 60, 60)
        painter.drawEllipse(QPoint(center_x, center_y), 90, 90)
        painter.drawLine(center_x - 110, center_y, center_x + 110, center_y)
        painter.drawLine(center_x, center_y - 110, center_x, center_y + 110)

        # Text
        painter.setPen(QColor(TEXT_PRIMARY))
        font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(font)
        text_rect = QRect(0, center_y + 40, width, 30)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "AWAITING EXPERIMENT STREAM")

        painter.setPen(QColor(TEXT_MUTED))
        sub_font = QFont("Segoe UI", 10)
        painter.setFont(sub_font)
        sub_rect = QRect(0, center_y + 65, width, 24)
        painter.drawText(
            sub_rect, Qt.AlignmentFlag.AlignCenter, "Press [PLAY] to initiate video pipeline"
        )

        painter.end()
        self._video_label.setPixmap(pix)


class StepItemCard(QFrame):
    """A card row representing a single protocol step state (Prev, Current, Next)."""

    def __init__(self, step_role: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_role = step_role  # "PREVIOUS", "CURRENT", "NEXT"
        self._is_active = step_role == "CURRENT"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Status icon / badge
        self._badge = QLabel(step_role[:4])
        self._badge.setFixedWidth(64)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Step details layout
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(2)

        self._lbl_role = QLabel(f"{step_role} STEP")
        self._lbl_role.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )

        self._lbl_name = QLabel("--")
        self._lbl_name.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        self._lbl_name.setWordWrap(True)

        details_layout.addWidget(self._lbl_role)
        details_layout.addWidget(self._lbl_name)

        layout.addWidget(self._badge)
        layout.addLayout(details_layout, 1)

        self.update_style()

    def set_step(self, step_text: str, is_completed: bool = False) -> None:
        self._lbl_name.setText(step_text)
        self.update_style(is_completed=is_completed)

    def update_style(self, is_completed: bool = False) -> None:
        if self._step_role == "CURRENT":
            self.setStyleSheet(
                "StepItemCard {"
                "  background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                "    stop:0 rgba(0, 240, 255, 0.12), stop:1 rgba(17, 28, 51, 0.9));"
                f"  border: 1.5px solid {ACCENT_CYAN};"
                "  border-radius: 8px;"
                "}"
            )
            self._badge.setStyleSheet(
                f"background-color: {ACCENT_CYAN}; color: #04131E; font-weight: 800; "
                f"font-size: 11px; border-radius: 4px; padding: 4px;"
            )
            self._badge.setText("[ ACTIVE ]")
            self._lbl_role.setStyleSheet(
                f"color: {ACCENT_CYAN}; font-size: 10px; font-weight: 800; letter-spacing: 1.2px;"
            )
            self._lbl_name.setStyleSheet("color: #FFFFFF; font-size: 13px; font-weight: 700;")
        elif self._step_role == "PREVIOUS":
            self.setStyleSheet(
                f"StepItemCard {{"
                f"  background-color: {BG_CARD};"
                f"  border: 1px solid {BORDER_CARD};"
                f"  border-radius: 8px;"
                f"}}"
            )
            if is_completed:
                self._badge.setStyleSheet(
                    "background-color: #064E3B; color: #34D399; font-weight: 700; "
                    "font-size: 11px; border-radius: 4px; padding: 4px;"
                )
                self._badge.setText("[ DONE ]")
            else:
                self._badge.setStyleSheet(
                    f"background-color: #1E293B; color: {TEXT_MUTED}; font-weight: 700; "
                    f"font-size: 11px; border-radius: 4px; padding: 4px;"
                )
                self._badge.setText("[ -- ]")
            self._lbl_name.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 500;"
            )
        else:  # NEXT
            self.setStyleSheet(
                f"StepItemCard {{"
                f"  background-color: {BG_CARD};"
                f"  border: 1px dashed {BORDER_MUTED};"
                f"  border-radius: 8px;"
                f"}}"
            )
            self._badge.setStyleSheet(
                f"background-color: #1E293B; color: {ACCENT_BLUE}; font-weight: 700; "
                f"font-size: 11px; border-radius: 4px; padding: 4px;"
            )
            self._badge.setText("[ NEXT ]")
            self._lbl_name.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-weight: 500;")


class StepNavigatorWidget(QFrame):
    """Protocol step tracker matching the wireframe (Previous, Current, Next, Confidence)."""

    def __init__(
        self,
        protocol: ExperimentProtocol = DEFAULT_TOY_PROTOCOL,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._protocol = protocol

        self.setStyleSheet(
            f"StepNavigatorWidget {{"
            f"  background-color: {BG_CARD};"
            f"  border: 1px solid {BORDER_CARD};"
            f"  border-radius: 8px;"
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header with protocol info
        header_layout = QHBoxLayout()
        lbl_title = QLabel("PROTOCOL STEP PROGRESSION")
        lbl_title.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
        )

        self._step_counter = QLabel("0 / 8")
        self._step_counter.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-family: monospace; font-weight: 700;"
        )

        header_layout.addWidget(lbl_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self._step_counter)

        # 3-step cards
        self._prev_card = StepItemCard("PREVIOUS")
        self._curr_card = StepItemCard("CURRENT")
        self._next_card = StepItemCard("NEXT")

        # Confidence Meter Section
        conf_layout = QVBoxLayout()
        conf_layout.setContentsMargins(0, 4, 0, 0)
        conf_layout.setSpacing(4)

        conf_header = QHBoxLayout()
        conf_label = QLabel("CONFIDENCE GAUGE")
        conf_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )

        self._conf_value_label = QLabel("0.0%")
        self._conf_value_label.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 12px; font-weight: 700; font-family: monospace;"
        )

        conf_header.addWidget(conf_label)
        conf_header.addStretch(1)
        conf_header.addWidget(self._conf_value_label)

        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setFixedHeight(10)

        conf_layout.addLayout(conf_header)
        conf_layout.addWidget(self._conf_bar)

        layout.addLayout(header_layout)
        layout.addWidget(self._prev_card)
        layout.addWidget(self._curr_card)
        layout.addWidget(self._next_card)
        layout.addLayout(conf_layout)

        self.reset_steps()

    def reset_steps(self) -> None:
        self._prev_card.set_step("None (Start of protocol)", is_completed=False)
        first_step = self._protocol.steps[0] if self._protocol.steps else None
        second_step = self._protocol.steps[1] if len(self._protocol.steps) > 1 else None

        self._curr_card.set_step(f"{first_step.id}: {first_step.name}" if first_step else "Idle")
        self._next_card.set_step(f"{second_step.id}: {second_step.name}" if second_step else "None")
        self._step_counter.setText(f"0 / {len(self._protocol.steps)}")
        self.update_confidence(0.0)

    def update_steps(
        self,
        current_step_name: str,
        current_step_id: str | None = None,
        done_steps: list[str] | None = None,
        expected_next_name: str | None = None,
    ) -> None:
        done = done_steps or []
        done_count = len(done)
        total = len(self._protocol.steps)
        self._step_counter.setText(f"{done_count} / {total}")

        # Determine Previous Step
        if done:
            last_done_id = done[-1]
            last_step_obj = self._protocol.step(last_done_id)
            prev_text = f"{last_done_id}: {last_step_obj.name}" if last_step_obj else last_done_id
            self._prev_card.set_step(prev_text, is_completed=True)
        else:
            self._prev_card.set_step("--", is_completed=False)

        # Current active step display
        curr_display = (
            f"{current_step_id}: {current_step_name}" if current_step_id else current_step_name
        )
        self._curr_card.set_step(curr_display)

        # Next Step
        if expected_next_name and expected_next_name != "None":
            self._next_card.set_step(expected_next_name)
        else:
            # Derive from protocol if possible
            if current_step_id and self._protocol.is_known(current_step_id):
                idx = self._protocol.index_of(current_step_id)
                if idx + 1 < len(self._protocol.steps):
                    next_obj = self._protocol.steps[idx + 1]
                    self._next_card.set_step(f"{next_obj.id}: {next_obj.name}")
                else:
                    self._next_card.set_step("Protocol Complete")
            else:
                self._next_card.set_step("--")

    def update_confidence(self, confidence: float) -> None:
        pct = int(max(0.0, min(1.0, confidence)) * 100)
        self._conf_bar.setValue(pct)
        self._conf_value_label.setText(f"{pct}%")

        # Dynamic color shift
        if pct >= 80:
            color = ACCENT_EMERALD
            chunk_gradient = f"stop:0 #059669, stop:1 {ACCENT_EMERALD}"
        elif pct >= 50:
            color = ACCENT_AMBER
            chunk_gradient = f"stop:0 #D97706, stop:1 {ACCENT_AMBER}"
        else:
            color = ACCENT_RED if pct > 0 else TEXT_MUTED
            chunk_gradient = f"stop:0 #DC2626, stop:1 {ACCENT_RED}"

        self._conf_value_label.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 700; font-family: monospace;"
        )
        self._conf_bar.setStyleSheet(
            f"QProgressBar {{"
            f"  background-color: #0A0F1D;"
            f"  border: 1px solid {BORDER_MUTED};"
            f"  border-radius: 5px;"
            f"}}"
            f"QProgressBar::chunk {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {chunk_gradient});"
            f"  border-radius: 4px;"
            f"}}"
        )


class ActivityLogWidget(QFrame):
    """Color-coded activity & event stream widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"ActivityLogWidget {{"
            f"  background-color: {BG_CARD};"
            f"  border: 1px solid {BORDER_CARD};"
            f"  border-radius: 8px;"
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("ACTIVITY LOGS & ALERTS")
        title.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
        )

        self._count_badge = QLabel("0 EVENTS")
        self._count_badge.setStyleSheet(
            f"background-color: {BG_BASE}; color: {TEXT_MUTED}; padding: 2px 6px; "
            f"border-radius: 4px; font-size: 10px; font-family: monospace;"
        )

        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self._count_badge)

        self._list = QListWidget()
        self._list.setMinimumHeight(140)

        layout.addLayout(header_layout)
        layout.addWidget(self._list, 1)

    def add_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "unknown")
        message = event.get("message") or event_type
        timestamp = event.get("timestamp", time.time())
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

        color, _, tag = EVENT_TYPE_COLORS.get(event_type, (TEXT_MUTED, "INFO", "[INFO]"))

        # Emphasize alerts
        if "skipped" in event_type or "out_of_sequence" in event_type:
            formatted_text = f"[{time_str}] {tag} ALERT! {message}"
        else:
            formatted_text = f"[{time_str}] {tag} > {message}"

        item = QListWidgetItem(formatted_text)
        item.setForeground(QBrush(QColor(color)))

        self._list.insertItem(0, item)
        while self._list.count() > 300:
            self._list.takeItem(self._list.count() - 1)

        self._count_badge.setText(f"{self._list.count()} EVENTS")


class ControlDeckWidget(QFrame):
    """Mission control deck with START/PAUSE/STOP buttons and status toggles."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"ControlDeckWidget {{"
            f"  background-color: {BG_CARD};"
            f"  border: 1px solid {BORDER_CARD};"
            f"  border-radius: 8px;"
            f"}}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        # Control Buttons
        self.btn_start = QPushButton("PLAY")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setMinimumWidth(90)

        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.setMinimumWidth(90)
        self.btn_pause.setEnabled(False)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setMinimumWidth(90)
        self.btn_stop.setEnabled(False)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        layout.addSpacing(16)

        # Indicators
        self._rec_indicator = QLabel("RECORDING: [OFF]")
        self._rec_indicator.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; font-family: monospace; "
            f"padding: 4px 8px; background: {BG_BASE}; border-radius: 4px; "
            f"border: 1px solid {BORDER_MUTED};"
        )

        self._voice_indicator = QLabel("VOICE ALERTS: [ON]")
        self._voice_indicator.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 700; font-family: monospace; "
            f"padding: 4px 8px; background: {BG_BASE}; border-radius: 4px; "
            f"border: 1px solid {BORDER_MUTED};"
        )

        layout.addWidget(self._rec_indicator)
        layout.addWidget(self._voice_indicator)
        layout.addStretch(1)

        # Session Timer
        self._session_start_time: float | None = None
        self._timer_label = QLabel("00:00:00")
        self._timer_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700; "
            f"font-family: monospace; padding: 2px 6px;"
        )
        layout.addWidget(QLabel("TIME:"))
        layout.addWidget(self._timer_label)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._on_clock_tick)

    def set_running_state(self, running: bool, paused: bool = False) -> None:
        if running and not paused:
            if self._session_start_time is None:
                self._session_start_time = time.time()
                self._clock_timer.start(1000)
            self._rec_indicator.setText("RECORDING: [ON]")
            self._rec_indicator.setStyleSheet(
                f"color: {ACCENT_RED}; font-size: 11px; font-weight: 700; font-family: monospace; "
                f"padding: 4px 8px; background: #2A0E12; border-radius: 4px; "
                f"border: 1px solid {ACCENT_RED};"
            )
        elif paused:
            self._rec_indicator.setText("RECORDING: [PAUSED]")
            self._rec_indicator.setStyleSheet(
                f"color: {ACCENT_AMBER}; font-size: 11px; font-weight: 700; "
                f"font-family: monospace; padding: 4px 8px; background: #241A0B; "
                f"border-radius: 4px; border: 1px solid {ACCENT_AMBER};"
            )
        else:
            self._session_start_time = None
            self._clock_timer.stop()
            self._timer_label.setText("00:00:00")
            self._rec_indicator.setText("RECORDING: [OFF]")
            self._rec_indicator.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; font-family: monospace; "
                f"padding: 4px 8px; background: {BG_BASE}; border-radius: 4px; "
                f"border: 1px solid {BORDER_MUTED};"
            )

    def _on_clock_tick(self) -> None:
        if self._session_start_time is not None:
            elapsed = int(time.time() - self._session_start_time)
            mins, secs = divmod(elapsed, 60)
            hours, mins = divmod(mins, 60)
            self._timer_label.setText(f"{hours:02d}:{mins:02d}:{secs:02d}")


__all__ = [
    "TelemetryChip",
    "VideoFeedWidget",
    "StepNavigatorWidget",
    "ActivityLogWidget",
    "ControlDeckWidget",
]
