# Architecture

This is the **full target architecture** for the eventual product, not just the 5-day PoC. The PoC implements a subset (the vertical slice in `AGENTS.md`); the rest is designed here so components are replaceable and the system can grow without rework. Read `AGENTS.md` first for the PoC scope.

## Design principle

**Pretrained-and-frozen where the problem is solved, fine-tune small where it's task-specific, hand-write logic where it's deterministic.** Do not make the whole system one large neural network.

The product function is Observe → Understand → Predict → Validate → Guide → Record, where validation is deterministic (FSM) and the only ML is step classification.

## High-level data flow (target)

```
Fixed camera(s)
   │
   v
Video preprocessing
   │
   ├─────────────┬──────────────┬──────────────┐
   │             │              │              │
   v             v              v              v
Object        Human pose    Hand tracking   (later) 3D mesh / orientation
detection     MediaPipe     MediaPipe       4D-Humans / ROMP / SMPL
YOLOv8n       (33 lm)       (21/hand)       optional module
   │             │              │
   └─────────────┴──────┬───────┘
                        │
                        v
                  Feature fusion
        body/hand features + object/payload features
        hand-object distances + rack-relative coordinates + motion
                        │
                        v
                  Temporal step model
              XGBoost baseline → GRU/1D-CNN/TCN upgrade
                        │
                        v
              Experiment Protocol FSM
        confirmed / skipped / repeated / out-of-sequence
                        │
      ┌─────────────────┼──────────────────┐
      v                 v                  v
 Voice alert      Structured log     GUI + video
 pyttsx3          JSON/CSV           PySide6
      │                 │                  │
      └─────────────────┴──────────────────┘
                        │
                        v
             Local recording + IP/RTSP stream (later)
```

## Modular pipeline (PoC implementation)

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
  → sequence validator (FSM)
  → event/result manager
  → database (SQLite/SQLAlchemy or JSON)  →  PySide6 dashboard
```

Each stage is a replaceable component behind a `Protocol`. The step classifier receives a **standardized feature/sequence representation**, never MediaPipe internals. `PoseResult` keypoints are model-independent.

### PoC implementation status

The modular pipeline above is **implemented** in `src/bas_assistant/` (see `AGENTS.md` → Status). Component status (per `docs/standards.md` labels):

| Stage | Status | Note |
|---|---|---|
| Video input | Implemented | OpenCV webcam/file + `DummyVideoSource` |
| Detection / tracking | Implemented (stub) | Full-frame stub; YOLO fine-tune deferred |
| Pose estimation | Implemented | MediaPipe (frozen) + `DummyPoseEstimator` fallback |
| Normalization | Implemented | Translation + scale only; orientation-agnostic 3D HMR is **not** claimed |
| Feature extraction | Implemented | 34-dim spatial + temporal window vector |
| Step classifier | Implemented (wrapper) | `DummyClassifier` default; `XGBoostStepClassifier` loads a trained model when present — **none trained yet** |
| Sequence validator | Implemented | Deterministic 7-step FSM; skip-step scenario tested |
| Events / storage | Implemented | Thread-safe `EventManager`; JSONL session log |
| GUI | Implemented | PySide6 dashboard; verified offscreen (no camera on dev machine) |
| SQLite backend, voice, streaming, ONNX | Not started | Deferred per ADR-0001 |

## Component contracts

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

### Step recognition vs sequence validation

A deliberate separation (ADR-0002):

- **Step Classifier** answers *"what is happening?"* — maps a feature window → predicted Step label (+ background/no-step class).
- **Sequence Validation (FSM)** answers *"is it allowed here?"* — checks the Step against Protocol State and emits `confirmed` / `skipped` / `repeated` / `out-of-sequence`.

The FSM reads only confirmed step events; it never sees raw frames.

## Orientation-agnostic design

`pose/normalization.py` implements only basic normalization (translation, scale, configurable coordinate normalization). Future techniques are structured for but **not claimed as implemented**:

- body-centric coordinate systems
- rotation normalization
- gravity-independent orientation handling
- camera-coordinate normalization
- 2D/3D pose normalization

### Payload-relative spatial reasoning (the future path)

Instead of reasoning only in camera-frame coordinates (`hand = (x, y)`), represent interaction relative to the payload/rack, which serves as the scene reference:

```
hand relative to rack
tool relative to rack
hand relative to tool
tool relative to rack slot
```

This makes the representation less dependent on the astronaut's orientation relative to gravity/floor. Full orientation-agnostic 3D HMR (4D-Humans / ROMP / SMPL) is an **optional future module** — never claim it as implemented unless actually built and validated.

## Component detail (target)

### Object detection
YOLOv8n fine-tuned on ~150–300 labeled frames of experiment props (or Roboflow-hosted). Exports to ONNX for CPU inference. In the PoC this is stubbed; MediaPipe hands + nearest-object heuristics provide the hand-object signals.

### Pose + hands
MediaPipe Pose (33 landmarks) + MediaPipe Hands (21/hand), pretrained and frozen, CPU real-time.

### Features
- **Spatial:** normalized joint coordinates, joint distances, joint angles, limb lengths, relative body geometry.
- **Temporal:** joint velocity, joint acceleration, movement magnitude, temporal differences, sequence statistics.
- **Interaction:** hand-to-object distance, hand velocity, nearest object per hand, wrist angle relative to torso, rack-relative coordinates.

### Step classifier
Baseline XGBoost over hand-crafted window features (sliding window ~1–2 s, background/no-step class, majority-vote smoothing). Upgrade path: LightGBM → small GRU/1D-CNN/TCN over landmark time series. ST-GCN / video backbones (X3D/MoViNet) are research-grade and only if data/time allow.

### Sequence validation
Plain Python FSM encoding the toy protocol with allowed transitions. Checks each confirmed Step against `expected_next`:
- match → advance state, log success, suggest next step
- repeated / never observed within window → `skipped`
- observed out of expected position → `out-of-sequence`

### Output layer
- **Voice alerts (later):** pyttsx3 (offline) on skip/out-of-sequence, optional step confirmation + next-step hint.
- **Structured log:** timestamped JSON/CSV per session: `{timestamp, step_name, status, confidence}`. Human-readable.
- **Video:** OpenCV `VideoWriter` for local storage; optional Flask/aiohttp MJPEG endpoint or ffmpeg RTSP for IP streaming (later).
- **GUI:** PySide6 desktop app (ADR-0004): live video, status panel, activity panel, event log, START/PAUSE/STOP.

### Storage
SQLite + SQLAlchemy models (`PersonObservation`, `PoseObservation`, `ActivityObservation`, `Event`), keypoints/features serialized as appropriate, isolated from ML code. Plain JSON acceptable for the PoC record.

### Inference boundary (edge deployment)
Keep an ONNX Runtime abstraction so Python inference can be swapped for ONNX at the edge without reworking the pipeline. Do not force every model to ONNX from day one.

## Boundaries & non-negotiables

- UI reads pipeline output through the event/result manager; no business logic in widgets.
- No DB logic in ML classes.
- The pipeline must run end-to-end with mocks/stubs where real components are absent.
- One frame failure must not crash the pipeline; failures must be observable, never silently swallowed.