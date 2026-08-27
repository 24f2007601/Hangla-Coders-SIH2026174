"""Sequence validation: the deterministic protocol FSM (ADR-0002)."""

from bas_assistant.validation.fsm import ExperimentFSM
from bas_assistant.validation.protocol import (
    DEFAULT_TOY_PROTOCOL,
    STEP_ADJUST_FOCUS,
    STEP_CLOSE_TRAY,
    STEP_COMPLETE,
    STEP_OPEN_TRAY,
    STEP_PICK_SAMPLE,
    STEP_PLACE_UNDER_SCOPE,
    STEP_RECORD_READING,
    STEP_START,
    ExperimentProtocol,
    ProtocolStep,
)

__all__ = [
    "STEP_ADJUST_FOCUS",
    "STEP_CLOSE_TRAY",
    "STEP_COMPLETE",
    "STEP_OPEN_TRAY",
    "STEP_PICK_SAMPLE",
    "STEP_PLACE_UNDER_SCOPE",
    "STEP_RECORD_READING",
    "STEP_START",
    "DEFAULT_TOY_PROTOCOL",
    "ExperimentFSM",
    "ExperimentProtocol",
    "ProtocolStep",
]
