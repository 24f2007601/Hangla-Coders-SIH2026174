# Dashboard Integration Plan — Wire PySide6 Mission Control to the Live Microphone Pipeline

> Status: **Completed (2026-09-02)**. All items below were implemented and verified:
> `pytest` (114 tests) + `ruff` + `black` clean; offscreen smoke tests passed for
> (a) dummy pipeline full lifecycle START/PAUSE/RESUME/STOP/RESET/second-session,
> (b) real factory pipeline (GPU YOLO + XGBoost) through the dashboard, (c) real
> webcam + MediaPipe + YOLO + XGBoost, (d) webcam-failure → ERROR → RESET path.
> Bug found and fixed during verification: worker `finished` signal overwrote the
> ERROR state after a source failure (`_source_failed` flag added).
> Scope decided with user: **PySide6 desktop target** (not the browser/WebSocket path from
> `docs/continuation.md` §6–7 — no web frontend exists in the repo), and **full panel
> coverage** per `docs/dashboard-design-plan.md` and the continuation.md live-data contract.

## Confirmed current state (inspected 2026-09-02)

1. `scripts/run_dashboard.py` **crashes on import today**: `ui/dashboard.py` and
   `ui/widgets.py` import `DEFAULT_TOY_PROTOCOL` from `validation.protocol`, which no
   longer exists — the codebase moved to `DEFAULT_MICROPHONE_PROTOCOL` (M0–M6 + G1/G2).
2. `PipelineWorker` (ui/dashboard.py) forwards frames/events but drops most of the
   live-data contract: no gate status, no LED states, no receiver detection/confidence,
   no frame number, no frame errors. It also does **not** pass `source_timestamp` to
   `ExperimentPipeline.process_frame()` — required for correct LED blink timing (G1/G2).
3. **No RESET flow**: after STOP the worker's `_stopped` flag stays `True`; START can
   never begin a new session. Widgets are never reset. (continuation.md §8 requires
   START/PAUSE/STOP/RESET.)
4. Activity log has no colors for `gate_g1_pending`, `gate_g1_passed`,
   `gate_g2_pending`, `gate_g2_passed`, `session_ended` — they render muted "unknown".
5. Pipeline has no public read-only access to protocol/LED state; the worker pokes
   privates (`_validator`, `_event_manager`) and would need `_led_estimator` too.

## Ground rules (from AGENTS.md / continuation.md)

- Do NOT modify ML logic: YOLO, MediaPipe, feature extraction, XGBoost, LED estimator,
  G1/G2 logic, FSM protocol rules stay untouched. The `ExperimentPipeline`/FSM remains
  the source of truth.
- Only **read-only additions** to the pipeline (public properties).
- No new dependencies, no new processes. One QThread + Qt signals is the transport.
- Reporting integrity: never invent FPS/accuracy numbers.

## Changes (ordered)

### 1. `src/bas_assistant/pipeline/pipeline.py` — read-only surface (no behavior change)

Add public properties to `ExperimentPipeline`:
- `protocol_state` → dict: `current_index`, `done_steps` (copy), `expected_next`
  (id/name or None), `is_complete`.
- `led_observation` → `LEDStateEstimator.observation` dataclass (receiver_detected,
  receiver_bbox, receiver_confidence, left, right, g1_passed, g2_passed, ready, ...).
- `frame_number` → int.
- `session_id` → from `repository.session_id` (None when no session).

### 2. `src/bas_assistant/ui/theme.py`

- `EVENT_TYPE_COLORS`: add entries for the 4 gate events + `session_ended`.
- New `LED_STATE_COLORS` map: blinking → emerald, steady → blue, off → muted,
  unknown → amber.

### 3. `src/bas_assistant/ui/widgets.py`

- Fix `DEFAULT_TOY_PROTOCOL` → `DEFAULT_MICROPHONE_PROTOCOL`.
- **New `ProtocolProgressWidget`** (replaces 3-card `StepNavigatorWidget`):
  - one row per protocol entry M0–M6 with G1/G2 interleaved at protocol positions,
  - per-row state badge: `DONE` (green) / `ACTIVE` (cyan glow) / `PENDING` (dim) /
    `GATE PASSED` (green) / `GATE PENDING` (amber) / `NOT REQUIRED` (dim),
  - progress counter "N / 7", confidence gauge (kept from old widget).
- **New `VerificationGatesWidget`**:
  - G1 and G2 status badges (NOT REQUIRED / PENDING amber / PASSED green),
  - left/right LED indicator lamps driven by `LED_STATE_COLORS`,
  - receiver status (DETECTED + confidence % / NOT DETECTED),
  - gate guidance line from `VerificationGate.description`.
- `ControlDeckWidget`: add **RESET** button; session timer pauses/resumes with the
  session (currently counts through PAUSE).
- `VideoFeedWidget`: `set_source_label(str)` for webcam/file indicator.

### 4. New `src/bas_assistant/ui/state.py`

Pure, Qt-free helper `build_status_snapshot(pipeline, result) -> dict` assembling the
full status dict — unit-testable without Qt:

```
status, fps, persons, active_events, latency_ms, frame_number, session_id,
step, step_id, confidence, done_steps, expected_next (id+name), is_complete,
gate_status, led: {left, right, receiver_detected, receiver_confidence,
g1_passed, g2_passed, ready}, frame_error, source_kind
```

Gate-status → badge-state mapping helper `gate_badge_state(gate_status)`:
`not_required → NOT REQUIRED`, `G1_PENDING → G1 PENDING`, `G1_PASSED → G1 PASSED`,
`G2_PENDING → G2 PENDING`, `G2_PASSED → G2 PASSED`.

### 5. `src/bas_assistant/ui/dashboard.py`

- Fix imports/defaults to `DEFAULT_MICROPHONE_PROTOCOL`; header title →
  "Wireless Microphone Experiment Protocol"; pass protocol through `run_dashboard()`.
- `PipelineWorker`:
  - pass `source_timestamp=self._source.timestamp` into `process_frame()`,
  - emit enriched `status_updated` via `build_status_snapshot(...)` (no more private
    pokes),
  - worker is created fresh per session by the Dashboard (no stale `_stopped` flag).
- `Dashboard`:
  - right column: header → protocol progress → verification gates → activity log,
  - wire all new fields; frame errors → ERROR chip + `system_error` event in log;
    source errors → existing `failed` path,
  - **RESET handler**: stop worker, wait, reset all widgets, fresh worker on next START
    (pipeline's own `start_session()` resets FSM/votes/LED history per continuation §8),
  - protocol completion → header badge `PROTOCOL COMPLETE`.

### 6. `scripts/run_dashboard.py`

- Support `--source dummy` → `DummyVideoSource` (offline smoke test / demo fallback).
- Pass protocol through to `run_dashboard()`.

### 7. Tests (`tests/unit/`)

- Pipeline public properties with dummy components.
- `build_status_snapshot` field mapping (dummy pipeline + FrameResult).
- `gate_badge_state` mapping.
- Existing 49 tests must keep passing.

### 8. Docs

- README dashboard section: run instructions (`--source`, `--pose`, `--classifier`,
  `--source dummy`), panels implemented list. Accurate Implemented/Not-Implemented only.

## Verification sequence

1. `pytest` (full suite, offline).
2. `ruff check .` + `black --check src scripts tests`.
3. Offscreen smoke: `QT_QPA_PLATFORM=offscreen python scripts/run_dashboard.py
   --source dummy --pose dummy` — no crash; start/pause/stop/reset exercised.
4. Recorded-video integration (`--source <validated experiment video>`) if available;
   then live webcam.
5. Manual flow: START → M0..M4 confirm → G1 pending/passed → M5 → M6 → G2 →
   `protocol_complete` badge; PAUSE/RESUME/STOP/RESET; bad-source error path.

## Final report (deliverables)

1. Files changed 2. Architecture chosen 3. Status dict schema (the "WebSocket schema"
   equivalent for Qt signals) 4. Dashboard→pipeline mapping 5. Run commands
   6. Tests performed 7. Remaining limitations.
