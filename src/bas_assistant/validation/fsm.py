"""Deterministic Experiment Protocol FSM (ADR-0002).

Validates confirmed Experiment Step events against the ordered protocol and emits
`confirmed` / `skipped` / `repeated` / `out-of-sequence` outcomes plus a
`protocol_complete` event when the last step is reached.

The FSM reads ONLY confirmed step events — it never sees raw frames or features.
"""

from __future__ import annotations

from bas_assistant.events.models import (
    EVENT_COMPLETE,
    EVENT_CONFIRMED,
    EVENT_OUT_OF_SEQUENCE,
    EVENT_REPEATED,
    EVENT_SKIPPED,
    Event,
)
from bas_assistant.validation.protocol import DEFAULT_TOY_PROTOCOL, ExperimentProtocol, ProtocolStep


class ExperimentFSM:
    """Tracks Protocol State and validates each confirmed step against it.

    Protocol State is: the index of the last completed step, the set of completed
    steps, and (derived) the expected next step.
    """

    def __init__(self, protocol: ExperimentProtocol = DEFAULT_TOY_PROTOCOL) -> None:
        self._protocol = protocol
        self.reset()

    # -- Protocol State -----------------------------------------------------

    @property
    def protocol(self) -> ExperimentProtocol:
        return self._protocol

    @property
    def current_index(self) -> int:
        """Index of the last completed step; -1 before the first step."""
        return self._current_index

    @property
    def done_steps(self) -> list[str]:
        return list(self._done)

    @property
    def is_complete(self) -> bool:
        return self._current_index >= len(self._protocol.steps) - 1

    @property
    def expected_next(self) -> ProtocolStep | None:
        """The step that should happen next, or None when the protocol is complete."""
        if self.is_complete:
            return None
        return self._protocol.steps[self._current_index + 1]

    # -- Lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self._current_index = -1
        self._done: list[str] = []

    # -- Validation ---------------------------------------------------------

    def on_step_confirmed(
        self, step_id: str, timestamp: float, confidence: float = 0.0
    ) -> list[Event]:
        """Validate one confirmed step and return the events it produced.

        Returns an empty list for unknown/background step ids (nothing to validate).
        """
        if not self._protocol.is_known(step_id):
            return []

        idx = self._protocol.index_of(step_id)
        old_index = self._current_index
        events: list[Event] = []

        if idx == old_index + 1:
            # Exactly the expected next step -> confirmed.
            self._advance_through(idx)
            events.append(
                Event(
                    timestamp=timestamp,
                    type=EVENT_CONFIRMED,
                    step=step_id,
                    message=f"Step {step_id} confirmed.",
                    details=self._guidance(idx),
                )
            )
        elif idx <= old_index:
            # Already completed -> repeated.
            events.append(
                Event(
                    timestamp=timestamp,
                    type=EVENT_REPEATED,
                    step=step_id,
                    message=f"Step {step_id} was already completed; repeated.",
                    details={"expected_next": self._expected_next_id()},
                )
            )
        else:
            # Jumped ahead: intervening steps are skipped; this step is out-of-sequence.
            for skipped_idx in range(old_index + 1, idx):
                skipped = self._protocol.steps[skipped_idx]
                events.append(
                    Event(
                        timestamp=timestamp,
                        type=EVENT_SKIPPED,
                        step=skipped.id,
                        message=f"Step {skipped.id} skipped (never observed).",
                        details={"expected_next": self._protocol.steps[old_index + 1].id},
                    )
                )
            self._advance_through(idx)
            events.append(
                Event(
                    timestamp=timestamp,
                    type=EVENT_OUT_OF_SEQUENCE,
                    step=step_id,
                    message=f"Step {step_id} out of sequence.",
                    details=self._guidance(idx),
                )
            )

        if idx == len(self._protocol.steps) - 1:
            events.append(
                Event(
                    timestamp=timestamp,
                    type=EVENT_COMPLETE,
                    step=step_id,
                    message="Experiment protocol complete.",
                    details={},
                )
            )
        return events

    # -- Internals ----------------------------------------------------------

    def _advance_through(self, idx: int) -> None:
        # Only the observed step is marked done; steps skipped by a forward jump are
        # never completed, so they must not appear in `done_steps`.
        self._done.append(self._protocol.steps[idx].id)
        self._current_index = idx

    def _expected_next_id(self) -> str | None:
        expected = self.expected_next
        return expected.id if expected else None

    def _guidance(self, idx: int) -> dict:
        expected = self.expected_next
        return {"suggested_next": expected.name if expected else None}


__all__ = ["ExperimentFSM"]
