# AGENTS.md

Guidance for AI agents and team members working in this repository. Read this first, then `CONTEXT.md` (glossary) and `docs/adr/` (key decisions). For depth, see the focused reference docs: `docs/architecture.md`, `docs/implementation-roadmap.md`, `docs/standards.md`, `docs/success-criteria.md`.

> **Provenance note:** the original planning documents (`SCAFFOLD.md`, `ISRO_*.md`) were the historical source; their essential content has been folded into this file and the docs above and the originals removed.

## Project Overview

We are building an **AI-based autonomous experiment execution and validation assistant for human spaceflight** (SIH 2026 ISRO problem statement: "AI Human Activity Recognition for On-board BAS Experiments"). An offline camera-based AI assistant observes an astronaut performing a predefined experiment, recognizes which protocol step is happening, validates it against the expected sequence, and guides/records the outcome.

**Product function:** Observe → Understand → Predict → Validate → Guide → Record. This is **not** generic HAR — it is *protocol-aware* HAR: the system asks *"What experiment step is being performed, is it valid at this point, and what should happen next?"*

Core design principle: **use pretrained-and-frozen models where the problem is already solved, train only task-specific components, and use deterministic logic where the problem is deterministic.** Do not make the whole system one large neural network.

## PoC Scope (5 days)

We are building an **initial limited prototype (proof of concept)**, not a PhD-thesis product. The scope was decided in ADR-0001.

### In scope — skeleton + ONE full vertical slice

```
webcam
  → MediaPipe pose (33 landmarks) + hands (21/hand)
  → hand-object distance features
  → XGBoost step classifier (trained on small custom dataset)
  → FSM sequence validation of a simple 7-step toy protocol
  → structured JSON log
  → PySide6 desktop GUI (live video, step status, event log)
```

Proves the whole Observe→Validate→Guide→Record loop with real components. GUI is PySide6 (ADR-0004); step classifier baseline is XGBoost (ADR-0003); sequence validation is a deterministic hand-rolled FSM (ADR-0002).

### Out of scope for the PoC (deferred, but architecture keeps them replaceable)

- YOLO object detection fine-tuning (stub the detector; MediaPipe hands give nearest-object signals)
- Real BAS payload / microgravity rig (use a simple webcam toy-protocol rig)
- ONNX Runtime export / edge deployment (create a clean inference boundary only)
- RTSP/IP streaming (optional Flask MJPEG if time remains)
- Offline voice TTS alerts (optional pyttsx3 if time remains)
- Full orientation-agnostic 3D HMR — never claim this as implemented

Do NOT over-engineer, add microservices, Docker/K8s, or spend significant time on UI styling.

## Architecture

Interface-first, modular, decoupled stages (each is replaceable):

```
camera/video
  → input pipeline
  → person detection   (stub-able)
  → person tracking    (stub-able)
  → pose estimation    (MediaPipe; PoseEstimator protocol)
  → pose normalization
  → feature extraction (spatial + temporal)
  → temporal sequence
  → step classifier    (XGBoost / dummy fallback)
  → sequence validator (FSM; confirmed/skipped/repeated/out-of-sequence)
  → event/result manager
  → database (SQLite/SQLAlchemy or JSON)  →  PySide6 dashboard
```

### Key rules

- The step classifier receives a **standardized feature/sequence representation**, never MediaPipe internals.
- `PoseResult` keypoints are model-independent (no MediaPipe structs leaking).
- The FSM reads only confirmed step events; it never sees raw frames.
- UI reads pipeline output through the event/result manager; no business logic in widgets; no DB logic in ML classes.
- The pipeline must run end-to-end with mocks/stubs where real components are absent (e.g., `dummy_classifier`).
- The pipeline must not crash because one frame fails — but never silently swallow errors.

### Interface contracts

Define abstract protocols wherever a component may be replaced:

```python
class VideoSource(Protocol): ...
class PersonDetector(Protocol): ...
class PersonTracker(Protocol): ...
class PoseEstimator(Protocol):
    def estimate(self, frame: np.ndarray) -> PoseResult: ...
class FeatureExtractor(Protocol): ...
class ActivityClassifier(Protocol): ...
class ResultRepository(Protocol): ...
```

### Pose representation (standardized, model-independent)

```python
PoseResult
    ├── timestamp
    ├── person_id
    ├── keypoints
    ├── confidence
    ├── bounding_box
    └── metadata
```

### Orientation-agnostic design

`pose/normalization.py` implements only basic normalization (translation, scale, configurable coordinate normalization). Future techniques (body-centric coordinates, rotation normalization, gravity-independent handling, camera-coordinate normalization, 2D/3D) are structured for but **not claimed as implemented**. Full orientation-agnostic 3D HMR is a future research task — see `docs/architecture.md` for the payload-relative path.

## Repo Structure

Target structure (package name to be finalized):

```
configs/default.yaml            # typed Pydantic config
data/{raw,processed,samples}/
models/                         # trained artifacts (.onnx/.json) — gitignored
notebooks/
scripts/{run_pipeline,run_demo,benchmark}.py
src/<package>/
  config/  pipeline/  video/  detection/  tracking/  pose/
  features/  classification/  events/  storage/  ui/  utils/
tests/{unit,integration}/
docs/adr/                       # key decisions
docs/architecture.md            # full target architecture
docs/implementation-roadmap.md  # future scope + full build plan
docs/standards.md               # best practices & guidelines
docs/success-criteria.md        # success definition
CONTEXT.md                      # domain glossary
AGENTS.md                       # this file
```

## Tech Stack

| Component | Primary | Fallback / upgrade |
|---|---|---|
| Language | Python 3.11+ | — |
| Dep management | `uv` | pip |
| Video I/O | OpenCV + NumPy | — |
| Pose + hands | MediaPipe Pose + Hands (pretrained, frozen) | MoveNet / RTMPose |
| Object detection | YOLOv8n (fine-tuned, later) | YOLO11n; stub in PoC |
| Step classifier | XGBoost (scikit-learn) on engineered features | LightGBM → GRU/1D-CNN/TCN (upgrade) |
| Sequence validation | Plain Python FSM (hand-rolled) | `transitions` library |
| Data/backend | SQLite + SQLAlchemy or plain JSON; Pydantic for config | — |
| GUI | PySide6 | Streamlit (only if time forces) |
| Voice (later) | pyttsx3 (offline) | Piper TTS |
| Streaming (later) | Flask/aiohttp MJPEG | ffmpeg RTSP |
| Edge export (later) | ONNX Runtime | TensorRT / TFLite |
| Testing | pytest | — |
| Quality | Ruff, Black, mypy where practical | — |

**Rule of thumb:** pretrained-and-frozen wherever a solved problem exists (pose, hands, TTS); fine-tune small where task-specific (YOLO on props); hand-write logic where deterministic (FSM).

## Build Spec (PoC scaffold)

### Pipeline orchestration

```python
pipeline.process_frame(frame)
# Frame → Detection → Tracking → Pose → Normalization → Feature Extraction → Classification → Event
```

### Feature extraction

- **Spatial:** normalized joint coordinates, joint distances, joint angles, limb lengths, relative body geometry.
- **Temporal:** joint velocity, joint acceleration, movement magnitude, temporal differences, sequence statistics.
- First implementation returns a minimal feature vector; step classifier uses a sliding window (~1–2 s) with a background/no-step class and majority-vote smoothing over the window to avoid flicker.

### Classification

- `ActivityClassifier` interface. `DummyClassifier` returns `("unknown", 0.0)` so the pipeline runs with no model.
- XGBoost wrapper must gracefully handle a missing trained model. Do NOT train a model during scaffolding.
- Train on a small custom dataset (see Dataset section) — this is the bottleneck, start it first.

### Demo mode (`python scripts/run_demo.py`)

1. Open webcam or sample video. Clear error if no webcam; allow `--source <path>`.
2. Read frames → run pipeline → display processed video.
3. Draw estimated pose if available; show current activity label; show FPS.

### PySide6 dashboard (functional skeleton, no styling effort)

- Video panel; Status panel (`System Status`, `FPS`, `Persons Detected`, `Active Events`, `Inference Latency`); Activity panel (`Person 1`, `Activity`, `Confidence`); Event/log panel; START/PAUSE/STOP.

### Storage

- SQLite + SQLAlchemy models: `PersonObservation`, `PoseObservation`, `ActivityObservation`, `Event` (id, timestamp, person_id, ...). Serialize keypoints/features as appropriate. Keep DB layer isolated from ML code. (Plain JSON is acceptable for the PoC record.)

### Configuration (`configs/default.yaml`)

```yaml
video:        { source: 0, width: 1280, height: 720, target_fps: 30 }
pose:         { model: mediapipe, min_detection_confidence: 0.5, min_tracking_confidence: 0.5 }
classifier:   { model_type: dummy, model_path: models/activity_classifier.onnx }
database:     { url: sqlite:///data/project.db }
pipeline:     { sequence_length: 30 }
```

Loaded via typed Pydantic settings.

### Logging & error handling

- Structured logging; log pipeline startup, component init, frame/inference/database failures, and shutdown. Avoid per-frame spam.
- Failures must be observable — never silently swallow errors.

### Testing & CI

- Unit tests: pose normalization, feature extraction, dummy classifier, pipeline execution, database repository.
- At least one integration test: dummy frame → pipeline → classification → result (no GPU, no camera). Runnable with `pytest`.
- GitHub Actions CI (`ci.yml`): install deps → Ruff → pytest → fail on failure.
- README: overview + Mermaid architecture + accurate Implemented/Not-Yet-Implemented status + setup + running + development.

## Dataset & Protocol

### Toy experiment protocol (the demo source of truth)

"Sample Analysis" — a 7-step toy protocol demoable on a webcam with props:

```
S0 Start
S1 Open sample tray
S2 Pick sample
S3 Place sample under scope
S4 Adjust focus knob
S5 Record reading
S6 Close tray
S7 Complete
```

Encoded in the FSM with allowed transitions; the FSM knows the expected next step and flags `confirmed` / `skipped` / `repeated` / `out-of-sequence`.

### Dataset strategy (start first — it is the bottleneck)

- Record 20–40 short clips (webcam/phone, fixed camera): correct runs, **plus deliberately skipped, repeated, and out-of-order steps**, plus idle/background. Do NOT collect only clean demonstrations — negative and abnormal cases are essential for validation.
- Augment: speed variation, lighting variation, mirrored orientation (partially addresses orientation without full 3D HMR).
- Use MediaPipe (pretrained) to auto-extract pose/hand landmarks from clips → this becomes the feature dataset, not raw pixels. Keeps training CPU-friendly.
- Label each clip's ground-truth step sequence.

## Success Definition (PoC)

The PoC is **done** when:

1. The project installs and imports cleanly (`uv sync`).
2. `pytest` passes, including the no-GPU/no-camera integration test.
3. The dummy pipeline executes end-to-end.
4. `python scripts/run_demo.py` starts correctly (webcam or sample video).
5. The full vertical slice runs: webcam → pose → features → XGBoost step → FSM → JSON log → GUI showing step status and events.
6. The FSM correctly produces confirmed/skipped/out-of-sequence outcomes on the toy protocol (demoed via the skip-Step-3 scenario).
7. README accurately reflects what is and is not implemented.

**Reporting integrity (non-negotiable):** never invent accuracy, FPS, latency, or dataset size; never claim a model is trained, a feature is implemented, or "mission-ready" unless confirmed. Distinguish `Implemented / Tested / Planned / Proposed / Target / Assumption` in all reporting. See `docs/success-criteria.md` for the full demo scenario and product-level success framing.

## Domain Model

`CONTEXT.md` is the authoritative glossary. **Use the canonical terms below; never mix in the avoided words.**

| Canonical term | Means | Avoid |
|---|---|---|
| Experiment Protocol | ordered predefined sequence of steps, the source of truth | procedure, recipe |
| Experiment Step | single discrete action in a protocol, atomic unit of validation | action, activity, stage |
| Step Recognition | classifying observed behavior into a Step (XGBoost over features) | activity classification, action recognition |
| Sequence Validation | deterministic FSM checking a Step against Protocol State | sequence model, learned validation |
| Step Classifier | maps feature window → predicted Step label ("what is happening?") | HAR model |
| Protocol State | FSM's current position: done steps, expected next, outcomes | current state, session state |
| Payload | experiment apparatus/rack, the spatial scene reference | rack, bench, workbench |
| Hand-Object Interaction | hand↔nearest-object relationship; strong step-identity signal | manipulation, grasping |
| Pose / Keypoints | model-independent body landmarks (MediaPipe 33-point) | skeleton data, landmarks |
| Observation | timestamped detection/pose/step record | detection, reading |
| Event | significant logged occurrence (confirmed/skipped/out-of-sequence) | alert, message, notification |
| Session | one continuous run producing one structured record | run, recording |
| Vertical Slice | one end-to-end path with real components proving the loop | demo, feature, module |

## Agent Skills & Agents

### Repo-local skills (`.agents/skills/`, registered via `opencode.json` → `skills.paths`)

- **grill-with-docs** — entry point that invokes `grilling` + `domain-modeling` to interview the user and write docs (ADRs + glossary) as the design sharpens. Use when planning or changing scope.
- **grilling** — interview the user relentlessly (design-tree, rounds of frontier questions) to stress-test a plan before committing.
- **domain-modeling** — build/sharpen the domain model: maintain `CONTEXT.md` (glossary only) and `docs/adr/` decisions. Use when discussing terminology or recording a decision.

### Global skills auto-loaded (used by agents below)

- **code-review** / **code-ultrareview** — review diffs for standards + spec compliance, correctness, and drift. Use before committing or opening a PR.
- **ui-ux-pro-max** — UI/UX design guidance for the PySide6 dashboard (keep functional, not polished).
- **accessibility** — WCAG 2.2 audit for the desktop GUI (contrast, focus, labels).
- **brainstorming** — structured design dialogue before implementation.
- **find-skills** — discover more skills from the ecosystem if a gap appears.

### Project agents (`.opencode/agent/`)

| Agent | Mode | Purpose | Backing skill(s) |
|---|---|---|---|
| `grill` | subagent | Interview/stress-test a plan before implementation | grilling |
| `domain-modeler` | subagent | Maintain CONTEXT.md glossary + ADRs | domain-modeling |
| `reviewer` | subagent | Review diffs for standards/spec/correctness before commit/PR | code-review, code-ultrareview |
| `ui-designer` | subagent | Design + a11y-audit the PySide6 GUI | ui-ux-pro-max, accessibility |

### Recommended workflow

1. Scope/plan any feature with `grill` (or the `grill-with-docs` flow) before writing code.
2. Have `domain-modeler` record new domain terms and ADR-worthy decisions inline.
3. Implement with the `build` agent following the architecture + conventions above.
4. Before committing, run `reviewer` over the diff; fix findings; re-run pytest + ruff.

## Config

`opencode.json` registers `.agents/skills` as a project skill path and loads `AGENTS.md` as instructions. After editing `opencode.json`, `.opencode/agent/*.md`, or `.agents/skills/*/SKILL.md`, **restart opencode** for changes to take effect.

## Status

Everything above is the *plan* for the 5-day PoC. The repo currently contains planning docs, this repo setup (`AGENTS.md`, `CONTEXT.md`, `docs/adr/`, `docs/*.md`, `.agents/`, `.opencode/`). No implementation code exists yet — the scaffold itself is the first build task. Never present planned capabilities as implemented.