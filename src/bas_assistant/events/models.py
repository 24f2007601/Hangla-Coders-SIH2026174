"""Event types and the shared Event dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVENT_CONFIRMED = "step_confirmed"
EVENT_SKIPPED = "step_skipped"
EVENT_REPEATED = "step_repeated"
EVENT_OUT_OF_SEQUENCE = "out_of_sequence"
EVENT_COMPLETE = "protocol_complete"
EVENT_SESSION_STARTED = "session_started"
EVENT_SESSION_ENDED = "session_ended"
EVENT_SYSTEM_ERROR = "system_error"


@dataclass(slots=True)
class Event:
    """A significant, logged occurrence during a session.

    Types: step_confirmed / step_skipped / step_repeated / out_of_sequence /
    protocol_complete / session_started / session_ended / system_error.
    """

    id: str = ""
    timestamp: float = 0.0
    type: str = ""
    step: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type,
            "step": self.step,
            "message": self.message,
            "details": dict(self.details),
        }


__all__ = [
    "EVENT_COMPLETE",
    "EVENT_CONFIRMED",
    "EVENT_OUT_OF_SEQUENCE",
    "EVENT_REPEATED",
    "EVENT_SESSION_ENDED",
    "EVENT_SESSION_STARTED",
    "EVENT_SKIPPED",
    "EVENT_SYSTEM_ERROR",
    "Event",
]
