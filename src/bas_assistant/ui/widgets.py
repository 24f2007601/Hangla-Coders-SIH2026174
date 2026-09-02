"""Custom space-exploration UI widgets for the BAS Experiment Assistant dashboard.

Implements:
- `VideoFeedWidget`: Live camera viewport with HUD overlays.
- `ProtocolProgressWidget`: Full protocol strip (M0-M6 with G1/G2 interleaved)
  showing completed / active / pending state per entry plus a confidence gauge.
- `VerificationGatesWidget`: G1/G2 gate badges, receiver LED indicators and
  receiver detection status.
- `ActivityLogWidget`: Color-coded activity and alert log stream with status badges.
- `TelemetryChip`: High-contrast metric cards with tabular numerical displays.
- `ControlDeckWidget`: Mission control bar with START/PAUSE/STOP/RESET and toggles.
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

from bas_assistant.ui.state import gate_badge_states
from bas_assistant.ui.theme import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    ACCENT_RED,
    BG_BASE,
    BG_CARD,
    BG_VOID,
    BORDER_CARD,
    BORDER_MUTED,
    EVENT_TYPE_COLORS,
    LED_STATE_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from bas_assistant.validation.protocol import (
    DEFAULT_MICROPHONE_PROTOCOL,
    GATE_ONE_MICROPHONE_PAIRED,
    GATE_RECEIVER_CONNECTED,
    STEP_REMOVE_RECEIVER,
    ExperimentProtocol,
)


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

        self._source_title = QLabel("SOURCE: --")
        self._source_title.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 1px; font-family: monospace;"
        )

        self._fps_hud = QLabel("0.0 FPS")
        self._fps_hud.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 700; font-family: monospace;"
        )

        hud_bar.addWidget(self._status_pill)
        hud_bar.addWidget(self._feed_title)
        hud_bar.addWidget(self._source_title)
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

    def set_source_label(self, text: str) -> None:
        """Set the video source indicator (e.g. 'WEBCAM 0' / 'FILE' / 'SIMULATED')."""
        self._source_title.setText(f"SOURCE: {text.upper()}")

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


# Entry states for the protocol progress strip.
STATE_DONE = "done"
STATE_ACTIVE = "active"
STATE_PENDING = "pending"
STATE_GATE_PASSED = "gate_passed"
STATE_GATE_PENDING = "gate_pending"
STATE_GATE_NOT_REQUIRED = "gate_not_required"

# (bg, border, text_color, badge_text) per entry state.
_ENTRY_STATE_STYLES: dict[str, tuple[str, str, str, str]] = {
    STATE_DONE: ("#064E3B", ACCENT_EMERALD, "#34D399", "DONE"),
    STATE_ACTIVE: ("rgba(0, 240, 255, 0.10)", ACCENT_CYAN, ACCENT_CYAN, "ACTIVE"),
    STATE_PENDING: (BG_CARD, BORDER_CARD, TEXT_MUTED, "PENDING"),
    STATE_GATE_PASSED: ("#064E3B", ACCENT_EMERALD, "#34D399", "PASSED"),
    STATE_GATE_PENDING: ("#241A0B", ACCENT_AMBER, ACCENT_AMBER, "PENDING"),
    STATE_GATE_NOT_REQUIRED: (BG_CARD, BORDER_CARD, TEXT_MUTED, "WAIT"),
}


class ProtocolRowItem(QFrame):
    """A compact row representing one protocol entry (step or verification gate)."""

    def __init__(
        self, entry_id: str, name: str, is_gate: bool, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._entry_id = entry_id
        self._name = name
        self._state = STATE_GATE_NOT_REQUIRED if is_gate else STATE_PENDING

        self.setFixedHeight(34)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(8)

        self._badge = QLabel(entry_id)
        self._badge.setFixedWidth(36)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setStyleSheet(
            f"background-color: #0E243A; color: {ACCENT_CYAN}; font-weight: 800; "
            f"font-size: 11px; border-radius: 4px; padding: 3px 0; "
            f"font-family: monospace; border: 1px solid {BORDER_MUTED};"
        )

        self._lbl_name = QLabel(name)
        self._lbl_name.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 500;")

        self._lbl_state = QLabel("--")
        self._lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_state.setFixedWidth(72)

        layout.addWidget(self._badge)
        layout.addWidget(self._lbl_name, 1)
        layout.addWidget(self._lbl_state)

        self._apply_state()

    @property
    def entry_id(self) -> str:
        return self._entry_id

    def set_state(self, state: str) -> None:
        if state not in _ENTRY_STATE_STYLES:
            return
        self._state = state
        self._apply_state()

    def _apply_state(self) -> None:
        bg, border, color, badge_text = _ENTRY_STATE_STYLES[self._state]
        self.setStyleSheet(
            f"ProtocolRowItem {{"
            f"  background-color: {bg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 6px;"
            f"}}"
        )
        self._lbl_state.setText(badge_text)
        self._lbl_state.setStyleSheet(
            f"background-color: {bg}; color: {color}; font-weight: 800; "
            f"font-size: 10px; letter-spacing: 1px; border-radius: 4px; padding: 2px 4px;"
        )
        if self._state == STATE_ACTIVE:
            self._lbl_name.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: 700;")
        else:
            self._lbl_name.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 500;"
            )


class ProtocolProgressWidget(QFrame):
    """Full protocol progression strip: M0-M6 with G1/G2 interleaved at their
    runtime positions (G1 between 'Remove receiver' and 'Connect receiver',
    G2 after the final step), plus a classifier confidence gauge."""

    def __init__(
        self,
        protocol: ExperimentProtocol = DEFAULT_MICROPHONE_PROTOCOL,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._protocol = protocol

        self.setStyleSheet(
            f"ProtocolProgressWidget {{"
            f"  background-color: {BG_CARD};"
            f"  border: 1px solid {BORDER_CARD};"
            f"  border-radius: 8px;"
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        lbl_title = QLabel("PROTOCOL PROGRESSION")
        lbl_title.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
        )

        self._step_counter = QLabel("0 / 7")
        self._step_counter.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-family: monospace; font-weight: 700;"
        )

        header_layout.addWidget(lbl_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self._step_counter)

        layout.addLayout(header_layout)

        self._rows: list[ProtocolRowItem] = []
        for entry_id, name, is_gate in self._display_entries():
            row = ProtocolRowItem(entry_id, name, is_gate)
            self._rows.append(row)
            layout.addWidget(row)

        # Confidence Meter Section
        conf_layout = QVBoxLayout()
        conf_layout.setContentsMargins(0, 6, 0, 0)
        conf_layout.setSpacing(4)

        conf_header = QHBoxLayout()
        conf_label = QLabel("STEP CONFIDENCE")
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

        layout.addLayout(conf_layout)

        self.reset_progress()

    def _display_entries(self) -> list[tuple[str, str, bool]]:
        """Ordered (id, name, is_gate) entries matching the runtime protocol flow."""
        entries: list[tuple[str, str, bool]] = []
        for step in self._protocol.steps:
            entries.append((step.id, step.name, False))
            # G1 is armed by the pipeline immediately after the receiver is
            # removed (before the receiver is connected) per the protocol flow
            # M0..M4 -> G1 -> M5 -> M6 -> G2.
            if step.id == STEP_REMOVE_RECEIVER:
                gate = self._protocol.gate(GATE_RECEIVER_CONNECTED)
                if gate:
                    entries.append((gate.id, gate.name, True))
        # G2 is the final verification after the last step.
        final_gate = self._protocol.gate(GATE_ONE_MICROPHONE_PAIRED)
        if final_gate:
            entries.append((final_gate.id, final_gate.name, True))
        return entries

    def reset_progress(self) -> None:
        for row in self._rows:
            is_gate = row.entry_id.startswith("G")
            row.set_state(STATE_GATE_NOT_REQUIRED if is_gate else STATE_PENDING)
        self._step_counter.setText(f"0 / {len(self._protocol.steps)}")
        self.update_confidence(0.0)

    def update_progress(
        self,
        done_steps: list[str] | None = None,
        expected_next_id: str | None = None,
        gate_status: str = "not_required",
        is_complete: bool = False,
    ) -> None:
        done = done_steps or []
        gate_states = gate_badge_states(gate_status)
        total = len(self._protocol.steps)
        self._step_counter.setText(
            f"{total} / {total}" if is_complete else f"{len(done)} / {total}"
        )

        for row in self._rows:
            entry_id = row.entry_id
            if entry_id in gate_states:
                phase = gate_states[entry_id]
                if phase == "PASSED":
                    row.set_state(STATE_GATE_PASSED)
                elif phase == "PENDING":
                    row.set_state(STATE_GATE_PENDING)
                else:
                    row.set_state(STATE_GATE_NOT_REQUIRED)
            elif entry_id in done:
                row.set_state(STATE_DONE)
            elif entry_id == expected_next_id and not is_complete:
                row.set_state(STATE_ACTIVE)
            else:
                row.set_state(STATE_PENDING)

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


class VerificationGatesWidget(QFrame):
    """Verification-gate panel: G1/G2 status badges, receiver LED indicator
    lamps, and receiver detection status driven entirely by pipeline state."""

    def __init__(
        self,
        protocol: ExperimentProtocol = DEFAULT_MICROPHONE_PROTOCOL,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._protocol = protocol

        self.setStyleSheet(
            f"VerificationGatesWidget {{"
            f"  background-color: {BG_CARD};"
            f"  border: 1px solid {BORDER_CARD};"
            f"  border-radius: 8px;"
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("VERIFICATION GATES")
        title.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 700; letter-spacing: 1px;"
        )
        layout.addWidget(title)

        gate_layout = QHBoxLayout()
        gate_layout.setSpacing(8)

        g1 = self._protocol.gate(GATE_RECEIVER_CONNECTED)
        g2 = self._protocol.gate(GATE_ONE_MICROPHONE_PAIRED)

        self._g1_badge = self._make_gate_badge("G1", g1.name if g1 else "Verify connection")
        self._g2_badge = self._make_gate_badge("G2", g2.name if g2 else "Verify pairing")
        gate_layout.addWidget(self._g1_badge, 1)
        gate_layout.addWidget(self._g2_badge, 1)
        layout.addLayout(gate_layout)

        # LED indicators
        led_header = QLabel("RECEIVER LEDS")
        led_header.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )
        layout.addWidget(led_header)

        led_layout = QHBoxLayout()
        led_layout.setSpacing(8)

        self._led_labels: dict[str, QLabel] = {}
        for side in ("LEFT", "RIGHT"):
            lamp = QLabel(side)
            lamp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            led_layout.addWidget(lamp, 1)
            self._led_labels[side] = lamp

        layout.addLayout(led_layout)

        # Receiver status + guidance
        self._receiver_label = QLabel("RECEIVER: NOT DETECTED")
        self._receiver_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; font-family: monospace;"
        )
        layout.addWidget(self._receiver_label)

        self._guidance_label = QLabel(g1.description if g1 else "")
        self._guidance_label.setWordWrap(True)
        self._guidance_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(self._guidance_label)

        self.reset_gates()

    @staticmethod
    def _make_gate_badge(gate_id: str, name: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName(f"gate_{gate_id}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        lbl_id = QLabel(gate_id)
        lbl_id.setStyleSheet(
            f"color: {ACCENT_CYAN}; font-size: 12px; font-weight: 800; font-family: monospace;"
        )
        lbl_name = QLabel(name)
        lbl_name.setWordWrap(True)
        lbl_name.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 9px;")

        lbl_status = QLabel("NOT REQUIRED")
        lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_status.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 1px;"
        )

        layout.addWidget(lbl_id)
        layout.addWidget(lbl_name)
        layout.addWidget(lbl_status)
        frame._lbl_status = lbl_status  # noqa: SLF001 - internal widget handle
        return frame

    def reset_gates(self) -> None:
        self._set_gate_badge(self._g1_badge, "NOT REQUIRED", TEXT_MUTED, BG_CARD, BORDER_CARD)
        self._set_gate_badge(self._g2_badge, "NOT REQUIRED", TEXT_MUTED, BG_CARD, BORDER_CARD)
        self._set_led_lamp("LEFT", "unknown")
        self._set_led_lamp("RIGHT", "unknown")
        self._receiver_label.setText("RECEIVER: NOT DETECTED")
        self._receiver_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; font-family: monospace;"
        )

    def update_gates(
        self,
        gate_status: str = "not_required",
        led: dict[str, Any] | None = None,
    ) -> None:
        led = led or {}
        states = gate_badge_states(gate_status)

        phase_g1 = states.get(GATE_RECEIVER_CONNECTED, "NOT_REQUIRED")
        if phase_g1 == "PASSED":
            self._set_gate_badge(self._g1_badge, "PASSED", "#34D399", "#064E3B", ACCENT_EMERALD)
        elif phase_g1 == "PENDING":
            self._set_gate_badge(self._g1_badge, "PENDING", ACCENT_AMBER, "#241A0B", ACCENT_AMBER)
        else:
            self._set_gate_badge(self._g1_badge, "NOT REQUIRED", TEXT_MUTED, BG_CARD, BORDER_CARD)

        phase_g2 = states.get(GATE_ONE_MICROPHONE_PAIRED, "NOT_REQUIRED")
        if phase_g2 == "PASSED":
            self._set_gate_badge(self._g2_badge, "PASSED", "#34D399", "#064E3B", ACCENT_EMERALD)
        elif phase_g2 == "PENDING":
            self._set_gate_badge(self._g2_badge, "PENDING", ACCENT_AMBER, "#241A0B", ACCENT_AMBER)
        else:
            self._set_gate_badge(self._g2_badge, "NOT REQUIRED", TEXT_MUTED, BG_CARD, BORDER_CARD)

        self._set_led_lamp("LEFT", str(led.get("left", "unknown")))
        self._set_led_lamp("RIGHT", str(led.get("right", "unknown")))

        if led.get("receiver_detected"):
            confidence = float(led.get("receiver_confidence", 0.0))
            self._receiver_label.setText(f"RECEIVER: DETECTED ({confidence * 100:.0f}%)")
            self._receiver_label.setStyleSheet(
                f"color: {ACCENT_EMERALD}; font-size: 11px; font-weight: 700; "
                f"font-family: monospace;"
            )
        else:
            self._receiver_label.setText("RECEIVER: NOT DETECTED")
            self._receiver_label.setStyleSheet(
                f"color: {ACCENT_AMBER}; font-size: 11px; font-weight: 700; "
                f"font-family: monospace;"
            )

    @staticmethod
    def _set_gate_badge(badge: QFrame, text: str, color: str, bg: str, border: str) -> None:
        lbl: QLabel = badge._lbl_status  # noqa: SLF001 - internal widget handle
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: 800; letter-spacing: 1px; "
            f"background-color: {bg}; border-radius: 4px; padding: 2px;"
        )
        badge.setStyleSheet(
            f"QFrame#gate_G1, QFrame#gate_G2 {{"
            f"  background-color: {bg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 6px;"
            f"}}"
            f"QFrame#gate_G1 QLabel, QFrame#gate_G2 QLabel {{"
            f"  background-color: transparent;"
            f"}}"
        )

    def _set_led_lamp(self, side: str, state: str) -> None:
        color = LED_STATE_COLORS.get(state, ACCENT_AMBER)
        label = self._led_labels[side]
        label.setText(f"{side}: {state.upper()}")
        label.setStyleSheet(
            f"color: {color}; background-color: #0A0F1D; border: 1.5px solid {color}; "
            f"border-radius: 10px; padding: 4px 8px; font-size: 10px; font-weight: 800; "
            f"font-family: monospace; letter-spacing: 1px;"
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

    def clear_events(self) -> None:
        self._list.clear()
        self._count_badge.setText("0 EVENTS")


class ControlDeckWidget(QFrame):
    """Mission control deck with START/PAUSE/STOP/RESET buttons and status toggles."""

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

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setMinimumWidth(90)
        self.btn_reset.setEnabled(False)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_reset)
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
        self._elapsed_base = 0.0
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
            if self._session_start_time is None or not self._clock_timer.isActive():
                # Fresh start or resume from pause: keep accumulated elapsed.
                self._session_start_time = time.time()
                self._clock_timer.start(1000)
            self._rec_indicator.setText("RECORDING: [ON]")
            self._rec_indicator.setStyleSheet(
                f"color: {ACCENT_RED}; font-size: 11px; font-weight: 700; font-family: monospace; "
                f"padding: 4px 8px; background: #2A0E12; border-radius: 4px; "
                f"border: 1px solid {ACCENT_RED};"
            )
        elif paused:
            if self._clock_timer.isActive():
                self._elapsed_base += time.time() - self._session_start_time
                self._clock_timer.stop()
            self._rec_indicator.setText("RECORDING: [PAUSED]")
            self._rec_indicator.setStyleSheet(
                f"color: {ACCENT_AMBER}; font-size: 11px; font-weight: 700; "
                f"font-family: monospace; padding: 4px 8px; background: #241A0B; "
                f"border-radius: 4px; border: 1px solid {ACCENT_AMBER};"
            )
        else:
            self.reset_timer()
            self._rec_indicator.setText("RECORDING: [OFF]")
            self._rec_indicator.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; font-family: monospace; "
                f"padding: 4px 8px; background: {BG_BASE}; border-radius: 4px; "
                f"border: 1px solid {BORDER_MUTED};"
            )

    def reset_timer(self) -> None:
        self._session_start_time = None
        self._elapsed_base = 0.0
        self._clock_timer.stop()
        self._timer_label.setText("00:00:00")

    def _on_clock_tick(self) -> None:
        if self._session_start_time is not None:
            elapsed = int(self._elapsed_base + time.time() - self._session_start_time)
            mins, secs = divmod(elapsed, 60)
            hours, mins = divmod(mins, 60)
            self._timer_label.setText(f"{hours:02d}:{mins:02d}:{secs:02d}")


__all__ = [
    "TelemetryChip",
    "VideoFeedWidget",
    "ProtocolProgressWidget",
    "VerificationGatesWidget",
    "ActivityLogWidget",
    "ControlDeckWidget",
]
