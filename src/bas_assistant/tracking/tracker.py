"""Person tracking. The PoC uses a single-person stub; a real tracker is deferred."""

from __future__ import annotations

from bas_assistant.models import BoundingBox, TrackedPerson


class DummyPersonTracker:
    """Assigns id 1 to the first detection each frame.

    The PoC validates a single astronaut, so a stable single-id stub is enough.
    """

    def update(self, detections: list[BoundingBox]) -> list[TrackedPerson]:
        if not detections:
            return []
        return [TrackedPerson(person_id=1, bbox=detections[0])]


__all__ = ["DummyPersonTracker"]
