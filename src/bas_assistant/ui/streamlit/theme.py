"""Theme and styling configuration for the Streamlit dashboard."""

from __future__ import annotations

from bas_assistant.ui.theme import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_EMERALD,
    BG_BASE,
    BG_CARD,
    BG_INPUT,
    BG_VOID,
    BORDER_CARD,
    BORDER_MUTED,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def get_streamlit_css() -> str:
    """Return the custom CSS stylesheet for space-exploration mission control."""
    return f"""
<style>
    .stApp {{
        background-color: {BG_VOID};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {BG_BASE} !important;
        border-right: 1px solid {BORDER_CARD};
    }}
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
