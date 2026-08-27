"""Event management for the pipeline."""

from bas_assistant.events.manager import EventManager
from bas_assistant.events.models import (
    EVENT_COMPLETE,
    EVENT_CONFIRMED,
    EVENT_OUT_OF_SEQUENCE,
    EVENT_REPEATED,
    EVENT_SESSION_ENDED,
    EVENT_SESSION_STARTED,
    EVENT_SKIPPED,
    EVENT_SYSTEM_ERROR,
    Event,
)

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
    "EventManager",
]
