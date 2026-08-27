"""Fixed-size sliding window buffer used for temporal feature accumulation."""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class FeatureWindow(Generic[T]):
    """A bounded FIFO buffer; `push` reports when the window becomes full."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("window size must be >= 1")
        self._size = size
        self._items: deque[T] = deque(maxlen=size)

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return len(self._items) == self._size

    def push(self, item: T) -> bool:
        """Append one item; returns True exactly when the window transitions to full."""
        was_full = self.is_full
        self._items.append(item)
        return self.is_full and not was_full

    def items(self) -> list[T]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()


__all__ = ["FeatureWindow"]
