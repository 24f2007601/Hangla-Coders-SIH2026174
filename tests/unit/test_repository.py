"""Unit tests for the JSON session-log repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bas_assistant.events.models import Event
from bas_assistant.storage.repository import JsonResultRepository


def test_session_lifecycle_writes_jsonl(tmp_path: Path) -> None:
    repo = JsonResultRepository(tmp_path)
    session_id = repo.start_session()
    assert session_id.startswith("session_")
    assert repo.session_id == session_id

    repo.record_observation({"frame": 0, "person_id": 1, "step": "S1"})
    repo.record_event(Event(timestamp=1.0, type="step_confirmed", step="S1", message="ok"))
    path = repo.end_session({"steps_completed": ["S1"]})

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["observation", "event", "summary"]

    records = [json.loads(line) for line in lines]
    assert records[0]["frame"] == 0
    assert records[1]["type"] == "step_confirmed"
    assert records[1]["step"] == "S1"
    assert records[2]["steps_completed"] == ["S1"]
    assert records[2]["session_id"] == session_id


def test_custom_session_id(tmp_path: Path) -> None:
    repo = JsonResultRepository(tmp_path)
    assert repo.start_session("my_session") == "my_session"
    assert (tmp_path / "my_session.jsonl").exists()


def test_write_before_start_raises(tmp_path: Path) -> None:
    repo = JsonResultRepository(tmp_path)
    with pytest.raises(RuntimeError):
        repo.record_observation({"frame": 0})


def test_event_to_dict_round_trip(tmp_path: Path) -> None:
    repo = JsonResultRepository(tmp_path)
    repo.start_session()
    repo.record_event(
        Event(timestamp=2.0, type="out_of_sequence", step="S4", details={"expected_next": "S3"})
    )
    path = repo.end_session()
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["kind"] == "event"
    assert record["details"]["expected_next"] == "S3"
