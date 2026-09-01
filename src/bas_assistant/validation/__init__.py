"""Sequence validation: the deterministic protocol FSM (ADR-0002)."""

from bas_assistant.validation.fsm import ExperimentFSM
from bas_assistant.validation.protocol import (
    DEFAULT_MICROPHONE_PROTOCOL,
    GATE_ONE_MICROPHONE_PAIRED,
    GATE_RECEIVER_CONNECTED,
    STEP_CONNECT_RECEIVER,
    STEP_MOVE_PHONE,
    STEP_OPEN_MICROPHONE_CASE,
    STEP_PICK_MICROPHONE_CASE,
    STEP_REMOVE_MICROPHONE,
    STEP_REMOVE_RECEIVER,
    STEP_VERIFY_PHONE_ON,
    ExperimentProtocol,
    ProtocolStep,
    VerificationGate,
)

__all__ = [
    "DEFAULT_MICROPHONE_PROTOCOL",
    "ExperimentFSM",
    "ExperimentProtocol",
    "GATE_ONE_MICROPHONE_PAIRED",
    "GATE_RECEIVER_CONNECTED",
    "ProtocolStep",
    "STEP_CONNECT_RECEIVER",
    "STEP_MOVE_PHONE",
    "STEP_OPEN_MICROPHONE_CASE",
    "STEP_PICK_MICROPHONE_CASE",
    "STEP_REMOVE_MICROPHONE",
    "STEP_REMOVE_RECEIVER",
    "STEP_VERIFY_PHONE_ON",
    "VerificationGate",
]