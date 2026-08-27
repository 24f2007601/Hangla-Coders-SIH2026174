"""Thread-safe event registry with a publish/subscribe mechanism.

The GUI and the JSON log subscribe to the EventManager so they observe pipeline
output without the pipeline knowing anything about them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from copy import deepcopy

from bas_assistant.events.models import Event

Observer = Callable[[Event], None]


class EventManager:
    """Holds events published during a session and notifies observers.

    Thread-safe: pipeline (worker) thread publishes; UI thread reads/observes.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._observers: list[Observer] = []
        self._lock = threading.RLock()
        self._counter = 0

    def subscribe(self, observer: Observer) -> None:
        with self._lock:
            self._observers.append(observer)

    def publish(self, event: Event) -> Event:
        with self._lock:
            self._counter += 1
            event.id = str(self._counter)
            self._events.append(deepcopy(event))
            observers = list(self._observers)
        for observer in observers:
            observer(event)
        return event

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._counter = 0


__all__ = ["EventManager"]
