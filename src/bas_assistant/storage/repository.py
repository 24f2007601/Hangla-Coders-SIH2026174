"""JSON result repository — writes a structured, streamable session log.

One JSON-lines file per session (`data/processed/<session_id>.jsonl`) with typed
records (``observation`` / ``event`` / ``summary``). Plain JSON is the PoC record
backend (ADR-0001); a SQLite/SQLAlchemy repository can replace this behind the same
`ResultRepository` protocol.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from bas_assistant.events.models import Event

logger = logging.getLogger(__name__)


class JsonResultRepository:
    """Streams observations and events for one session to a JSONL file."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._session_id = ""
        self._file = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def start_session(self, session_id: str | None = None) -> str:
        self._session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._output_dir / f"{self._session_id}.jsonl"
        self._file = self._path.open("w", encoding="utf-8")
        logger.info("Started session %s -> %s", self._session_id, self._path)
        return self._session_id

    def record_observation(self, observation: dict) -> None:
        self._write({"kind": "observation", **observation})

    def record_event(self, event: Event) -> None:
        self._write({"kind": "event", **event.to_dict()})

    def end_session(self, summary: dict | None = None) -> Path:
        record = {"kind": "summary", "session_id": self._session_id}
        if summary:
            record.update(summary)
        self._write(record)
        if self._file is not None:
            self._file.close()
            self._file = None
        logger.info("Ended session %s", self._session_id)
        return self._path

    def _write(self, record: dict) -> None:
        if self._file is None:
            raise RuntimeError("repository session not started; call start_session() first")
        self._file.write(json.dumps(record, default=str) + "\n")


__all__ = ["JsonResultRepository"]
