"""Space-exploration theme and styling system for the PySide6 dashboard.

Defines color tokens, gradients, fonts, and comprehensive QSS stylesheets
matching the ISRO BAS mission-control aesthetic: deep navy/void backgrounds,
glowing cyan accents, glassmorphic dark panels, and high-contrast telemetry.
"""

from __future__ import annotations

# -- Color Palette Tokens ---------------------------------------------------
BG_VOID = "#070B14"
BG_BASE = "#0B1120"
BG_CARD = "#111C33"
BG_CARD_HOVER = "#162544"
BG_CARD_TRANSLUCENT = "rgba(17, 28, 51, 0.85)"
BG_INPUT = "#0A0F1D"

BORDER_MUTED = "#1E2C4A"
BORDER_ACCENT = "#00F0FF"
BORDER_ACCENT_GLOW = "rgba(0, 240, 255, 0.45)"
BORDER_CARD = "#1E355A"

ACCENT_CYAN = "#00F0FF"
ACCENT_BLUE = "#38BDF8"
ACCENT_EMERALD = "#10B981"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"
ACCENT_PURPLE = "#818CF8"

TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"
TEXT_ACCENT = "#00F0FF"

# RGB Tuples and badge info for programmatic usage
EVENT_TYPE_COLORS: dict[str, tuple[str, str, str]] = {
    # event_type: (hex_color, label_badge, tag)
    "step_confirmed": (ACCENT_EMERALD, "CONFIRMED", "[OK]"),
    "step_skipped": (ACCENT_AMBER, "SKIPPED", "[ALERT]"),
    "step_repeated": (ACCENT_PURPLE, "REPEATED", "[REPEAT]"),
    "out_of_sequence": (ACCENT_RED, "OUT OF SEQ", "[INVALID]"),
    "protocol_complete": (ACCENT_CYAN, "COMPLETE", "[COMPLETE]"),
    "session_started": (ACCENT_BLUE, "SESSION", "[SESSION]"),
    "system_error": (ACCENT_RED, "ERROR", "[ERROR]"),
    "unknown": (TEXT_MUTED, "INFO", "[INFO]"),
}


def get_qss_stylesheet() -> str:
    """Generate the complete mission-control dark space QSS stylesheet."""
    return f"""
    QMainWindow {{
        background-color: {BG_VOID};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }}

    QWidget {{
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }}

    /* --- Group Boxes & Cards --- */
    QGroupBox {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_CARD};
        border-radius: 8px;
        margin-top: 14px;
        padding: 14px 10px 10px 10px;
        font-weight: bold;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {ACCENT_CYAN};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 14px;
        padding: 2px 8px;
        background-color: {BG_BASE};
        border: 1px solid {BORDER_MUTED};
        border-radius: 4px;
        color: {ACCENT_CYAN};
    }}

    /* --- Push Buttons --- */
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1E293B, stop:1 #0F172A);
        border: 1px solid {BORDER_MUTED};
        border-radius: 6px;
        color: {TEXT_PRIMARY};
        padding: 8px 16px;
        font-weight: bold;
        font-size: 12px;
        letter-spacing: 0.5px;
        min-height: 22px;
    }}

    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2A3B52, stop:1 #162238);
        border: 1px solid {ACCENT_CYAN};
        color: #FFFFFF;
    }}

    QPushButton:pressed {{
        background-color: #0F172A;
        border: 1px solid {ACCENT_BLUE};
    }}

    QPushButton:disabled {{
        background-color: #0A0F1D;
        border: 1px solid #162035;
        color: {TEXT_MUTED};
    }}

    /* Primary Start / Play Button */
    QPushButton#btn_start {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #065F46, stop:1 #047857);
        border: 1px solid {ACCENT_EMERALD};
        color: #FFFFFF;
        font-weight: bold;
    }}
    QPushButton#btn_start:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #047857, stop:1 #10B981);
        border: 1px solid #34D399;
    }}
    QPushButton#btn_start:disabled {{
        background: #0D281E;
        border: 1px solid #134E39;
        color: #4B6B5D;
    }}

    /* Pause Button */
    QPushButton#btn_pause {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #78350F, stop:1 #92400E);
        border: 1px solid {ACCENT_AMBER};
        color: #FFFFFF;
    }}
    QPushButton#btn_pause:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #92400E, stop:1 #B45309);
        border: 1px solid #FCD34D;
    }}
    QPushButton#btn_pause:disabled {{
        background: #1C160C;
        border: 1px solid #382B14;
        color: #5C4B2E;
    }}

    /* Stop Button */
    QPushButton#btn_stop {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7F1D1D, stop:1 #991B1B);
        border: 1px solid {ACCENT_RED};
        color: #FFFFFF;
    }}
    QPushButton#btn_stop:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #991B1B, stop:1 #DC2626);
        border: 1px solid #F87171;
    }}
    QPushButton#btn_stop:disabled {{
        background: #1F0D0D;
        border: 1px solid #3B1616;
        color: #5E3232;
    }}

    /* --- List Widgets --- */
    QListWidget {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER_MUTED};
        border-radius: 6px;
        padding: 4px;
        color: {TEXT_PRIMARY};
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        outline: none;
    }}

    QListWidget::item {{
        padding: 6px 8px;
        border-radius: 4px;
        margin-bottom: 2px;
        border-bottom: 1px solid rgba(30, 44, 74, 0.4);
    }}

    QListWidget::item:hover {{
        background-color: rgba(0, 240, 255, 0.08);
        border: 1px solid rgba(0, 240, 255, 0.2);
    }}

    /* --- Progress Bar --- */
    QProgressBar {{
        background-color: #0A0F1D;
        border: 1px solid {BORDER_MUTED};
        border-radius: 5px;
        text-align: center;
        color: {TEXT_PRIMARY};
        font-size: 11px;
        font-weight: bold;
        height: 16px;
    }}

    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 {ACCENT_CYAN});
        border-radius: 4px;
    }}

    /* --- Scrollbars --- */
    QScrollBar:vertical {{
        background: {BG_VOID};
        width: 8px;
        margin: 0px;
        border-radius: 4px;
    }}

    QScrollBar::handle:vertical {{
        background: #1E293B;
        min-height: 20px;
        border-radius: 4px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {ACCENT_BLUE};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}
    """


def load_theme_fonts() -> None:
    """Load default system fonts into Qt font database if available."""
    from pathlib import Path

    from PySide6.QtGui import QFontDatabase

    font_paths = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/consolab.ttf"),
    ]
    for p in font_paths:
        if p.exists():
            QFontDatabase.addApplicationFont(str(p))


__all__ = [
    "ACCENT_AMBER",
    "ACCENT_BLUE",
    "ACCENT_CYAN",
    "ACCENT_EMERALD",
    "ACCENT_PURPLE",
    "ACCENT_RED",
    "BG_BASE",
    "BG_CARD",
    "BG_CARD_HOVER",
    "BG_CARD_TRANSLUCENT",
    "BG_INPUT",
    "BG_VOID",
    "BORDER_ACCENT",
    "BORDER_ACCENT_GLOW",
    "BORDER_CARD",
    "BORDER_MUTED",
    "EVENT_TYPE_COLORS",
    "TEXT_ACCENT",
    "TEXT_MUTED",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "get_qss_stylesheet",
    "load_theme_fonts",
]
