# Phase Development Roadmap
## BAS Assistant — AI Experiment Execution & Validation Assistant
### ISRO SIH 2026 | Problem 2026174 | Deadline: 20 September 2026

> **Reporting integrity:** Status labels throughout this document follow `docs/standards.md`.
> Never claim a phase or task is done unless explicitly verified. Labels used:
> `Implemented` · `Tested` · `Planned` · `Proposed` · `Target` · `Assumption`

---

## Current State Snapshot

| Component | Status | Notes |
|-----------|--------|-------|
| Repo scaffold + CI | **Tested** | 49/49 pytest pass, ruff + black clean |
| Modular pipeline (end-to-end) | **Tested** | Runs with dummy inputs; no webcam verified |
| MediaPipe pose estimator | **Implemented** | Breaks on MediaPipe 1.0 — fix: downgrade to `0.10.21` |
| DummyPoseEstimator | **Tested** | Used in all offline tests |
| Feature extraction (34-dim) | **Tested** | Spatial + temporal window features |
| DummyClassifier | **Tested** | Returns `("unknown", 0.0)` — pipeline runs |
| XGBoostStepClassifier (wrapper) | **Implemented** | Loads model if present; **no model trained yet** |
| 7-step FSM (Sample Analysis) | **Tested** | confirmed/skipped/out-of-sequence validated |
| PySide6 dashboard | **Implemented** | Verified offscreen; live webcam not tested |
| JSONL session log | **Tested** | Written by `run_pipeline.py --source dummy` |
| Voice alerts (pyttsx3) | NOT STARTED | Deferred |
| YOLO object detection | NOT STARTED | Stubbed; deferred |
| IP/RTSP streaming | NOT STARTED | Deferred |
| ONNX edge export | NOT STARTED | Deferred |
| **Trained step classifier** | **BOTTLENECK** | No dataset recorded, no model trained |

---

## The 4 Phases to Demo Day

```
Phase 0 (NOW)     --> Fix MediaPipe + verify live webcam
Phase 1 (Week 1)  --> Dataset collection + feature extraction
Phase 2 (Week 2)  --> Train XGBoost + full vertical slice live
Phase 3 (Week 3)  --> System integration (voice, GUI polish, streaming)
Phase 4 (Week 4)  --> Polish, packaging, edge prep, demo rehearsal
                                              |
                                    20 September 2026 -- DEMO DAY
```

---

## Phase 0 — Environment Fix & Live Verification
### Duration: 1–2 days | Status: Planned

**Goal:** Get the pipeline running on a real webcam with MediaPipe pose overlay before writing any new code.

### Tasks

- [ ] **Fix MediaPipe version**
  ```powershell
  uv add "mediapipe==0.10.21"
  ```
  Pin in `pyproject.toml`: `mediapipe>=0.10.21,<1.0`

- [ ] **Verify dummy pipeline still passes**
  ```powershell
  uv run pytest --tb=short -q
  ```
  Must still say 49 passed.

- [ ] **Run demo with dummy source** (no webcam needed)
  ```powershell
  uv run python scripts/run_demo.py --source dummy --pose dummy
  ```

- [ ] **Run demo with real webcam**
  ```powershell
  uv run python scripts/run_demo.py
  ```
  Should open window, draw pose skeleton, show FPS + activity label.

- [ ] **Launch full PySide6 dashboard on webcam**
  ```powershell
  uv run python scripts/run_dashboard.py
  ```

- [ ] **Commit fix to `prith` branch**
  ```powershell
  git add pyproject.toml uv.lock
  git commit -m "fix: pin mediapipe<1.0 to restore mp.solutions API"
  git push origin prith
  ```

### Acceptance Criteria
- `pytest` passes (49 tests)
- Webcam feed visible with pose skeleton drawn
- FPS counter showing >= 15 FPS on live webcam
- PySide6 dashboard opens without crash

---

## Phase 1 — Dataset Collection & Feature Extraction
### Duration: Days 3–7 | Status: Planned

**Goal:** Record, label, and process the custom toy-protocol dataset. This is the #1 bottleneck — everything downstream (classifier training, FSM validation, demo) depends on it.

### The Toy Protocol to Record

```
S0  Start         (idle / begin posture)
S1  Open tray     (hands open container/tray)
S2  Pick sample   (pick-up gesture, object in hand)
S3  Place sample  (place object under camera/scope area)
S4  Adjust focus  (hand rotates knob-like motion)
S5  Record reading (hand near notepad/button/tablet)
S6  Close tray    (hands close container)
S7  Complete       (hands down / idle)
```

### Tasks

#### 1.1 — Set up props & recording station
- [ ] Choose 6–8 physical props to represent each step (a tray, small object, scope, knob, notepad, etc.)
- [ ] Fix camera at a stable angle (tripod or clamp) — consistent position matters
- [ ] Ensure consistent lighting (avoid shadows across hands)
- [ ] Test recording at 1080p, 30 FPS minimum

#### 1.2 — Record clips (20–40 total)

| Clip type | Count | Description |
|-----------|-------|-------------|
| Correct full runs | 8–10 | All 7 steps in order, different speeds |
| Correct partial runs | 3–5 | Stop mid-protocol |
| Skipped step (S3) | 4–5 | Go S2 to S4, skip placing sample |
| Skipped step (S5) | 2–3 | Go S4 to S6, skip recording |
| Repeated step | 3–4 | Do same step twice in a row |
| Out-of-order steps | 3–4 | e.g., S4 before S3 |
| Idle / background | 5–6 | No action, standing, walking past |
| Partial / ambiguous | 3–5 | Interrupted actions, half-completed gestures |

> **Why negatives?** The FSM validation only works if the classifier can distinguish wrong sequences.
> Training on only clean data produces a classifier that never fires `skipped` or `out-of-sequence`.

#### 1.3 — Label clips
- [ ] Create `data/raw/labels.csv` with columns: `clip_filename, step_id, step_name, is_correct, notes`
- [ ] Label start/end timestamps for each step within each clip

#### 1.4 — Extract features from clips (MediaPipe auto-labeling)
- [ ] Write `scripts/extract_features.py` that:
  1. Reads each labeled clip
  2. Runs MediaPipe Pose + Hands on every frame
  3. Runs the existing `PoseFeatureExtractor` to get the 34-dim feature vector
  4. Writes `data/processed/features.csv` with columns: `frame_id, clip_id, step_label, feature_0 ... feature_33`
- [ ] Run extraction:
  ```powershell
  uv run python scripts/extract_features.py --input data/raw/ --output data/processed/features.csv
  ```

#### 1.5 — Augment the dataset
- [ ] Mirror clips horizontally (left/right orientation swap)
- [ ] Speed variation: process at 0.75x and 1.25x frame rate
- [ ] Brightness jitter: apply +-20% brightness to raw frames before extraction

#### 1.6 — Inspect & validate
- [ ] Plot per-step sample counts — no class should have < 100 windows
- [ ] Verify feature extraction produces non-NaN values for all frames
- [ ] Run a quick sanity check: visualize 10 random feature vectors

### Acceptance Criteria
- `data/processed/features.csv` exists with >= 500 labeled feature windows
- All 8+ step labels (S0–S7 + idle) represented with >= 50 windows each
- Negative cases (skip, repeat, out-of-order) make up >= 30% of the dataset
- No NaN values in feature matrix

---

## Phase 2 — Train Step Classifier & Live Vertical Slice
### Duration: Days 8–12 | Status: Planned

**Goal:** Train the XGBoost step classifier, drop it into the pipeline, and run the full vertical slice live on a webcam.

### Tasks

#### 2.1 — Train XGBoost step classifier
- [ ] Write `scripts/train_classifier.py` that:
  1. Loads `data/processed/features.csv`
  2. Splits train/val (80/20, stratified by step label)
  3. Trains `XGBoostClassifier` with reasonable defaults
  4. Evaluates on validation set (per-class accuracy, confusion matrix)
  5. Saves model to `models/step_classifier.json`
  6. Saves label encoder to `models/label_encoder.json`

  ```powershell
  uv run python scripts/train_classifier.py \
    --features data/processed/features.csv \
    --output models/step_classifier.json
  ```

- [ ] Review confusion matrix — identify confused step pairs
- [ ] Iterate: tune `max_depth`, `n_estimators`, `learning_rate` if accuracy < 70%
- [ ] If XGBoost underperforms, try LightGBM (same interface, usually better on small data)

#### 2.2 — Wire trained model into pipeline
- [ ] Update `configs/default.yaml`:
  ```yaml
  classifier:
    model_type: xgboost
    model_path: models/step_classifier.json
  ```
- [ ] Verify `XGBoostStepClassifier` loads the model:
  ```powershell
  uv run python scripts/run_pipeline.py --source dummy
  ```
  Should log step labels instead of `unknown`

#### 2.3 — Run full vertical slice live
- [ ] Run with webcam:
  ```powershell
  uv run python scripts/run_demo.py
  ```
- [ ] Perform each of the 7 steps in front of the camera
- [ ] Verify the step label on screen updates correctly for each step
- [ ] Verify FSM transitions: S0 to S7 all log `confirmed`

#### 2.4 — Demo the skip-step-3 scenario (the core differentiator)
- [ ] Run S0 -> S1 -> S2 -> **skip S3** -> do S4
- [ ] Verify the GUI/terminal shows `skipped` or `out-of-sequence` for S3
- [ ] Verify the JSONL log records the event correctly
- [ ] **Record this demo on video** — this is the centerpiece of the presentation

#### 2.5 — Run all tests again
```powershell
uv run ruff check .
uv run black --check src scripts tests
uv run pytest --tb=short -q
```
All must pass. If new scripts break tests, fix them.

### Acceptance Criteria
- Trained `step_classifier.json` in `models/`
- Per-step validation accuracy >= 70% (Target — not a claim until measured)
- Full vertical slice runs live: webcam -> pose -> features -> XGBoost -> FSM -> GUI
- Skip-step-3 scenario produces `skipped`/`out-of-sequence` event live
- All 49+ pytest pass

---

## Phase 3 — System Integration
### Duration: Days 13–18 | Status: Planned

**Goal:** Add voice alerts, polish the GUI, add video recording, and optionally enable streaming.

### Tasks

#### 3.1 — Voice alerts (pyttsx3)
- [ ] Add dependency: `uv add pyttsx3`
- [ ] Create `src/bas_assistant/alerts/voice.py`:
  ```python
  class VoiceAlert:
      def speak(self, message: str) -> None: ...
  ```
- [ ] Wire into `EventManager`: on `skipped` / `out-of-sequence` events -> trigger voice
- [ ] Alert messages:
  - Skipped: "Step [N] skipped. Please [step name]."
  - Out-of-sequence: "Invalid step order. Expected [step name]."
  - Confirmed: "Step [N] confirmed." (optional)
- [ ] Test offline — pyttsx3 is fully local, no internet needed

#### 3.2 — Video recording
- [ ] Add OpenCV `VideoWriter` to `src/bas_assistant/video/recorder.py`
- [ ] Record the annotated frame (with pose overlay + step label) to `data/recordings/session_<id>.mp4`
- [ ] Add `--record` flag to `run_demo.py` and `run_dashboard.py`

#### 3.3 — GUI polish (PySide6 dashboard)
- [ ] Verify all panels work with live webcam
- [ ] Add protocol progress indicator (step tracker S0->S1->...->S7 with pass/fail markers)
- [ ] Display expected next step prominently
- [ ] Show FSM outcome: `confirmed` in green, `skipped` in red, `out-of-sequence` in orange

#### 3.4 — Structured log output
- [ ] Ensure JSONL log captures: `{timestamp, step_id, step_name, status, confidence, person_id}`
- [ ] Add CSV export option for the session log
- [ ] Verify log is written even if the GUI crashes

#### 3.5 — IP streaming (optional — add only if time allows)
- [ ] Add Flask/aiohttp MJPEG endpoint in `src/bas_assistant/streaming/mjpeg_server.py`
- [ ] Serve annotated frames at `http://localhost:8080/stream`
- [ ] Test: open in browser on another device on the same WiFi
- [ ] **Skip this if Phases 3.1–3.4 consume all available time**

#### 3.6 — Performance optimization
- [ ] Profile: measure FPS with `scripts/benchmark.py` on webcam
- [ ] If FPS < 15: reduce MediaPipe model complexity from 1 to 0
- [ ] If FPS < 10: add frame skip (process every Nth frame, display all)
- [ ] Target: >= 20 FPS live on CPU (Target — not a claim until measured)

### Acceptance Criteria
- Voice alert fires on skipped/out-of-sequence events
- Annotated video recorded locally during a session
- GUI shows step tracker with pass/fail markers
- JSONL + CSV log both written correctly
- Live FPS >= 15 on CPU webcam (Target)

---

## Phase 4 — Polish, Packaging & Demo Rehearsal
### Duration: Days 19–23 | Status: Planned

**Goal:** Make the demo bulletproof. Package everything. Rehearse the skip-step-3 scenario.

### Tasks

#### 4.1 — Standalone offline test
- [ ] Unplug the machine from the internet
- [ ] Run the full pipeline from cold start
- [ ] Verify: no internet calls, no API keys needed, everything local
- [ ] Voice alerts work offline
- [ ] Pose estimation works offline

#### 4.2 — Demo script
Write `docs/demo-script.md` with the exact procedure:
1. Start the dashboard
2. Perform S1–S7 correctly -> show all green `confirmed`
3. Perform run with S3 skipped -> show red `skipped`, voice alert fires
4. Show JSONL log in a text editor
5. Show video recording playback

#### 4.3 — Architecture diagram
- [ ] Create a clean Mermaid diagram of the full pipeline
- [ ] Include: Camera -> MediaPipe -> Features -> XGBoost -> FSM -> GUI/Voice/Log
- [ ] Use canonical domain terms from `CONTEXT.md`
- [ ] Export as `docs/architecture-diagram.png` for the pitch deck

#### 4.4 — README final update
- [ ] Move trained classifier from "Not yet implemented" to "Implemented + Tested"
- [ ] Update accuracy/FPS numbers ONLY with actually measured values
- [ ] Add demo video link/thumbnail
- [ ] Verify setup instructions work on a clean machine

#### 4.5 — Optional: ONNX export
- [ ] Export XGBoost model to ONNX (if time allows)
- [ ] Verify inference via ONNX Runtime matches XGBoost output
- [ ] Enables edge deployment on Jetson Nano / Raspberry Pi — mention as future work if not implemented

#### 4.6 — Optional: 3D HMR module (orientation bonus)
- [ ] Only attempt if Phases 1–3 are fully complete and time remains
- [ ] Use pretrained 4D-Humans or ROMP — inference only, never train
- [ ] Never claim orientation-agnostic unless actually integrated and tested

#### 4.7 — Pitch framing (talking points)
- **Offline:** No internet connection in orbit — system runs 100% locally
- **Bandwidth:** No cloud inference — all processing on-device
- **Microgravity:** Orientation normalization (basic); mention 3D HMR as future work
- **Comms delay:** Local decision-making; no round-trip to ground
- **Novelty:** Protocol-aware validation (not generic HAR) — the FSM is the differentiator

#### 4.8 — Final rehearsal (3x dry runs)
- [ ] Full demo run 1: note all failures, fix them
- [ ] Full demo run 2: note remaining issues
- [ ] Full demo run 3: timing run (aim for 5-minute demo)
- [ ] Pre-record backup demo video in case of webcam failure

### Acceptance Criteria
- System runs offline (internet unplugged)
- Skip-step-3 demo works reliably 3 runs in a row
- README and docs accurate (no invented numbers)
- Pitch talking points prepared
- Backup demo video recorded

---

## Team Division (6 Members)

| Role | Members | Phases | Responsibilities |
|------|---------|--------|-----------------|
| Hardware | 2 | 0, 1 | Camera, mounting, lighting, props, physical rig, IP networking, webcam verification |
| ML / Backend | 1 | 1, 2 | Feature extraction script, XGBoost training, model evaluation, pipeline wiring |
| Frontend / GUI | 2 | 3, 4 | PySide6 dashboard polish, protocol tracker, event log, video recording UI |
| Product / Integration | 1 | 2, 3, 4 | Full vertical slice integration, voice alerts, streaming, demo rehearsal, pitch |

---

## Upgrade Paths (if baseline underperforms)

| Component | Baseline | Upgrade | Trigger |
|-----------|---------|---------|---------|
| Step classifier | XGBoost on 34-dim features | LightGBM -> GRU/1D-CNN/TCN | Validation accuracy < 70% |
| Detection | Full-frame stub | YOLOv8n fine-tuned on props | Hand-object signals too noisy |
| Pose | MediaPipe 0.10 | MoveNet / RTMPose | Dependency or accuracy issues |
| Hands | MediaPipe Hands | 100DOH hand-object detector | Heuristic too noisy |
| Orientation | Basic translation/scale | Pretrained HMR (4D-Humans/ROMP) | Everything else solid + time remains |
| Streaming | None | Flask MJPEG -> ffmpeg RTSP | When streaming is required |
| Edge export | None | ONNX Runtime -> TensorRT/TFLite | Final packaging for Jetson/Pi |

---

## Crash-Mode Priorities (36h crunch)

If time runs out, cut from the bottom of this list:

1. FSM validator on pre-recorded video — the core differentiator, non-negotiable
2. MediaPipe pose/hand overlay + step label live on webcam — visual proof
3. Structured JSONL log — always written, always correct
4. Voice alert on skip/out-of-sequence — high value, low effort
5. PySide6 dashboard with step tracker (OpenCV window fallback acceptable)
6. Local video recording
7. MJPEG IP streaming
8. 3D HMR orientation bonus — skip entirely under time pressure; frame as future work

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MediaPipe 1.0 breaks demo | High (already seen) | High | Downgrade to 0.10.21 immediately (Phase 0) |
| Insufficient dataset -> poor classifier | High | High | Record negatives early; use DummyClassifier as fallback if model fails |
| Webcam unavailable on demo day | Medium | High | Pre-record demo video as backup |
| Low FPS on demo machine | Medium | Medium | Profile early; reduce model complexity; frame skip |
| XGBoost underfits | Medium | Medium | Upgrade to LightGBM or add interaction features |
| pyttsx3 voice fails silently | Low | Low | Test offline; add fallback text-only alert |
| Time runs out before GUI polish | Medium | Low | Use OpenCV window fallback — judges care about the FSM, not the GUI |

---

## Definition of Done (Demo Day)

- [ ] System installs and runs on a clean machine with `uv sync`
- [ ] `pytest` passes — all tests green
- [ ] Live webcam pipeline runs at >= 15 FPS
- [ ] Step classifier labels steps correctly (>= 70% accuracy on validation set)
- [ ] FSM produces `confirmed` for correct steps in real-time
- [ ] FSM produces `skipped`/`out-of-sequence` when S3 is intentionally skipped
- [ ] Voice alert fires on skip/out-of-sequence
- [ ] JSONL log written correctly for the session
- [ ] GUI shows step tracker with pass/fail markers
- [ ] System runs offline (internet unplugged)
- [ ] Skip-step-3 demo runs reliably 3x in a row
- [ ] Backup demo video recorded
- [ ] README accurately reflects what is and is not implemented

---

*Last updated: 2026-08-28 | Status labels per `docs/standards.md` | Deadline: 20 September 2026*
