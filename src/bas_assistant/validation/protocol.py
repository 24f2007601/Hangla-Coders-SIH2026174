"""Wireless microphone experiment protocol and canonical step/gate definitions.

The protocol is the deterministic source of truth the FSM validates against.

Experiment Steps are M0-M6.
Verification Gates G1 and G2 are handled separately from classifier labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bas_assistant.classification.models import BACKGROUND_STEP

# ---------------------------------------------------------------------------
# Canonical Experiment Step IDs
# ---------------------------------------------------------------------------

STEP_VERIFY_PHONE_ON = "M0"
STEP_MOVE_PHONE = "M1"
STEP_PICK_MICROPHONE_CASE = "M2"
STEP_OPEN_MICROPHONE_CASE = "M3"
STEP_REMOVE_RECEIVER = "M4"
STEP_CONNECT_RECEIVER = "M5"
STEP_REMOVE_MICROPHONE = "M6"


# ---------------------------------------------------------------------------
# Canonical Verification Gate IDs
# ---------------------------------------------------------------------------

GATE_RECEIVER_CONNECTED = "G1"
GATE_ONE_MICROPHONE_PAIRED = "G2"


@dataclass(frozen=True, slots=True)
class ProtocolStep:
    """A recognized Experiment Step in the ordered protocol."""

    id: str
    name: str
    index: int
    verification_oriented: bool = False


@dataclass(frozen=True, slots=True)
class VerificationGate:
    """A deterministic verification condition evaluated separately from steps."""

    id: str
    name: str
    index: int
    required_after_step: str
    description: str


class ExperimentProtocol:
    """Ordered wireless-microphone protocol — the source of truth."""

    def __init__(
        self,
        steps: Sequence[ProtocolStep],
        gates: Sequence[VerificationGate] = (),
        background_step: str = BACKGROUND_STEP,
    ) -> None:
        step_ids = [step.id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"protocol step ids must be unique: {step_ids}")

        gate_ids = [gate.id for gate in gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError(f"protocol gate ids must be unique: {gate_ids}")

        self._steps = tuple(steps)
        self._gates = tuple(gates)
        self._by_id = {step.id: step for step in steps}
        self._gates_by_id = {gate.id: gate for gate in gates}
        self._background = background_step

    @property
    def steps(self) -> tuple[ProtocolStep, ...]:
        return self._steps

    @property
    def gates(self) -> tuple[VerificationGate, ...]:
        return self._gates

    @property
    def background(self) -> str:
        return self._background

    @property
    def start(self) -> ProtocolStep:
        return self._steps[0]

    @property
    def end(self) -> ProtocolStep:
        return self._steps[-1]

    def step(self, step_id: str) -> ProtocolStep | None:
        return self._by_id.get(step_id)

    def gate(self, gate_id: str) -> VerificationGate | None:
        return self._gates_by_id.get(gate_id)

    def is_known(self, step_id: str) -> bool:
        return step_id in self._by_id

    def is_gate(self, gate_id: str) -> bool:
        return gate_id in self._gates_by_id

    def index_of(self, step_id: str) -> int:
        if not self.is_known(step_id):
            raise ValueError(f"unknown step id: {step_id!r}")
        return self._by_id[step_id].index

    def labels(self) -> list[str]:
        """Classifier output labels: background + Experiment Steps only."""
        return [self._background, *[step.id for step in self._steps]]


# ---------------------------------------------------------------------------
# Wireless Microphone Experiment Protocol
# ---------------------------------------------------------------------------

DEFAULT_MICROPHONE_PROTOCOL = ExperimentProtocol(
    steps=[
        ProtocolStep(
            STEP_VERIFY_PHONE_ON,
            "Verify phone powered on",
            0,
            verification_oriented=True,
        ),
        ProtocolStep(
            STEP_MOVE_PHONE,
            "Move phone to working station",
            1,
        ),
        ProtocolStep(
            STEP_PICK_MICROPHONE_CASE,
            "Pick microphone case",
            2,
        ),
        ProtocolStep(
            STEP_OPEN_MICROPHONE_CASE,
            "Open microphone case",
            3,
        ),
        ProtocolStep(
            STEP_REMOVE_RECEIVER,
            "Remove receiver",
            4,
        ),
        ProtocolStep(
            STEP_CONNECT_RECEIVER,
            "Connect receiver to phone",
            5,
        ),
        ProtocolStep(
            STEP_REMOVE_MICROPHONE,
            "Remove one microphone",
            6,
        ),
    ],
    gates=[
        VerificationGate(
            id=GATE_RECEIVER_CONNECTED,
            name="Verify receiver connection",
            index=0,
            required_after_step=STEP_CONNECT_RECEIVER,
            description="Both receiver blue LEDs must be blinking.",
        ),
        VerificationGate(
            id=GATE_ONE_MICROPHONE_PAIRED,
            name="Verify one-microphone pairing",
            index=1,
            required_after_step=STEP_REMOVE_MICROPHONE,
            description=(
                "Exactly one receiver blue LED must be steady illuminated "
                "while the other continues blinking for the configured dwell time."
            ),
        ),
    ],
)


__all__ = [
    "GATE_ONE_MICROPHONE_PAIRED",
    "GATE_RECEIVER_CONNECTED",
    "STEP_CONNECT_RECEIVER",
    "STEP_MOVE_PHONE",
    "STEP_OPEN_MICROPHONE_CASE",
    "STEP_PICK_MICROPHONE_CASE",
    "STEP_REMOVE_MICROPHONE",
    "STEP_REMOVE_RECEIVER",
    "STEP_VERIFY_PHONE_ON",
    "DEFAULT_MICROPHONE_PROTOCOL",
    "ExperimentProtocol",
    "ProtocolStep",
    "VerificationGate",
]