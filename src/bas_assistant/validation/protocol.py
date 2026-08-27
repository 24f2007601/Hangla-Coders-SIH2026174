"""The toy Experiment Protocol definition and canonical step ids.

The protocol is the deterministic source of truth the FSM validates against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bas_assistant.classification.models import BACKGROUND_STEP

# Canonical step ids for the toy "Sample Analysis" protocol.
STEP_START = "S0"
STEP_OPEN_TRAY = "S1"
STEP_PICK_SAMPLE = "S2"
STEP_PLACE_UNDER_SCOPE = "S3"
STEP_ADJUST_FOCUS = "S4"
STEP_RECORD_READING = "S5"
STEP_CLOSE_TRAY = "S6"
STEP_COMPLETE = "S7"


@dataclass(frozen=True, slots=True)
class ProtocolStep:
    id: str
    name: str
    index: int


class ExperimentProtocol:
    """An ordered, predefined sequence of Experiment Steps — the source of truth."""

    def __init__(
        self, steps: Sequence[ProtocolStep], background_step: str = BACKGROUND_STEP
    ) -> None:
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"protocol step ids must be unique: {ids}")
        self._steps = tuple(steps)
        self._by_id = {s.id: s for s in steps}
        self._background = background_step

    @property
    def steps(self) -> tuple[ProtocolStep, ...]:
        return self._steps

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

    def is_known(self, step_id: str) -> bool:
        return step_id in self._by_id

    def index_of(self, step_id: str) -> int:
        if not self.is_known(step_id):
            raise ValueError(f"unknown step id: {step_id!r}")
        return self._by_id[step_id].index

    def labels(self) -> list[str]:
        """All classifier output labels: background + every step id."""
        return [self._background, *[s.id for s in self._steps]]


# The demo source of truth: "Sample Analysis" — a 7-step toy protocol.
DEFAULT_TOY_PROTOCOL = ExperimentProtocol(
    [
        ProtocolStep(STEP_START, "Start experiment", 0),
        ProtocolStep(STEP_OPEN_TRAY, "Open sample tray", 1),
        ProtocolStep(STEP_PICK_SAMPLE, "Pick sample", 2),
        ProtocolStep(STEP_PLACE_UNDER_SCOPE, "Place sample under scope", 3),
        ProtocolStep(STEP_ADJUST_FOCUS, "Adjust focus knob", 4),
        ProtocolStep(STEP_RECORD_READING, "Record reading", 5),
        ProtocolStep(STEP_CLOSE_TRAY, "Close sample tray", 6),
        ProtocolStep(STEP_COMPLETE, "Complete experiment", 7),
    ]
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
    "ExperimentProtocol",
    "ProtocolStep",
]
