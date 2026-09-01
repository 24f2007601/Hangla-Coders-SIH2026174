"""Unit tests for the deterministic Experiment Protocol FSM."""

from __future__ import annotations

from bas_assistant.events.models import (
    EVENT_COMPLETE,
    EVENT_CONFIRMED,
    EVENT_OUT_OF_SEQUENCE,
    EVENT_REPEATED,
    EVENT_SKIPPED,
)
from bas_assistant.validation.fsm import ExperimentFSM
from bas_assistant.validation.protocol import DEFAULT_MICROPHONE_PROTOCOL

STEPS = [step.id for step in DEFAULT_MICROPHONE_PROTOCOL.steps]


def test_initial_state() -> None:
    fsm = ExperimentFSM()
    assert fsm.current_index == -1
    assert fsm.done_steps == []
    assert fsm.expected_next is not None
    assert fsm.expected_next.id == STEPS[0]


def test_full_correct_sequence_confirms_every_step() -> None:
    fsm = ExperimentFSM()
    types: list[str] = []

    for i, step in enumerate(STEPS):
        events = fsm.on_step_confirmed(step, timestamp=float(i))
        assert len(events) >= 1
        assert events[0].type == EVENT_CONFIRMED
        assert events[0].step == step
        types.append(events[0].type)

    assert types == [EVENT_CONFIRMED] * len(STEPS)
    assert fsm.is_complete
    assert fsm.expected_next is None
    assert fsm.done_steps == STEPS


def test_complete_event_emitted_at_last_step() -> None:
    fsm = ExperimentFSM()
    complete_seen = False

    for i, step in enumerate(STEPS):
        events = fsm.on_step_confirmed(step, timestamp=float(i))
        complete_seen |= any(event.type == EVENT_COMPLETE for event in events)

    assert complete_seen


def test_skip_step_scenario() -> None:
    """Skip M4 (remove receiver) and jump directly to M5."""
    fsm = ExperimentFSM()

    for step in ("M0", "M1", "M2", "M3"):
        events = fsm.on_step_confirmed(step, timestamp=1.0)
        assert events[0].type == EVENT_CONFIRMED

    events = fsm.on_step_confirmed("M5", timestamp=2.0)

    assert any(event.type == EVENT_SKIPPED and event.step == "M4" for event in events)
    assert any(event.type == EVENT_OUT_OF_SEQUENCE and event.step == "M5" for event in events)
    assert fsm.current_index == 5

    for step in ("M6",):
        events = fsm.on_step_confirmed(step, timestamp=3.0)
        assert events[0].type == EVENT_CONFIRMED

    assert fsm.is_complete is True
    assert "M4" not in fsm.done_steps


def test_jump_ahead_skips_multiple_steps() -> None:
    fsm = ExperimentFSM()

    fsm.on_step_confirmed("M0", timestamp=1.0)
    events = fsm.on_step_confirmed("M6", timestamp=2.0)

    skipped = [event.step for event in events if event.type == EVENT_SKIPPED]

    assert skipped == ["M1", "M2", "M3", "M4", "M5"]
    assert any(event.type == EVENT_OUT_OF_SEQUENCE and event.step == "M6" for event in events)


def test_repeated_step_detected() -> None:
    fsm = ExperimentFSM()

    fsm.on_step_confirmed("M0", timestamp=1.0)
    fsm.on_step_confirmed("M1", timestamp=2.0)

    events = fsm.on_step_confirmed("M1", timestamp=3.0)

    assert any(event.type == EVENT_REPEATED and event.step == "M1" for event in events)
    assert fsm.done_steps == ["M0", "M1"]


def test_unknown_and_background_steps_ignored() -> None:
    fsm = ExperimentFSM()

    assert fsm.on_step_confirmed("NOPE", timestamp=1.0) == []
    assert fsm.on_step_confirmed("background", timestamp=1.0) == []
    assert fsm.current_index == -1


def test_reset_restores_initial_state() -> None:
    fsm = ExperimentFSM()

    fsm.on_step_confirmed("M0", timestamp=1.0)

    assert fsm.current_index == 0

    fsm.reset()

    assert fsm.current_index == -1
    assert fsm.done_steps == []
