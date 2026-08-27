# Implementation Roadmap — Future Scope & Full Build Plan

This document describes the **future scope and the full build plan** for the eventual system, beyond the 5-day PoC (which is `AGENTS.md`). The deadline is **20 September 2026** (SIH internal event). The PoC is Day 0–5; the roadmap below extends to the deadline.

## What you're actually building

Five subsystems glued together:

1. **Perception** — see the astronaut, the payload/equipment, and their hands
2. **State tracking** — know which Experiment Step is currently happening
3. **Sequence validator** — the FSM rulebook that checks observed Step order against the protocol
4. **Alerting** — voice output when something is wrong
5. **Logging + streaming + GUI** — the "boring but graded" wrapper

Only **step classification** is real ML. Detection and pose are solved problems you use off-the-shelf; sequence logic is a state machine, not ML.

## PoC first (Days 1–5)

1. Scaffold the repo (structure, interfaces, config, tests, CI, README). — **Done**
2. Build the vertical slice: webcam → MediaPipe → features → XGBoost → FSM → JSON log → PySide6 GUI. — **Implemented** (real pipeline components; step classifier still `dummy` — no trained model yet). Live webcam + MediaPipe run pending on hardware.
3. Record the toy-protocol dataset and train the step classifier (start this first — it is the bottleneck). — **Not started**; next task.
4. Demo the skip-step-3 scenario. — **Validated in tests** (integration + FSM unit tests); live webcam demo pending.

## Full build plan (to Sept 20)

### Phase 1 — Foundations & data (Week 1)
- Finalize the toy protocol and step list (already defined: "Sample Analysis", 7 steps).
- Record + label the custom clips (20–40 clips: correct runs, skips, repeats, out-of-order, idle/background).
- Get YOLOv8n + MediaPipe running end-to-end on a single recorded video (no real-time yet).

### Phase 2 — Core ML (Week 2)
- Build the feature-fusion pipeline (hand/object/pose → per-frame features).
- Train the step classifier (start XGBoost, iterate; upgrade only if underperforming).
- Build + test the FSM on labeled clips, including deliberately skipped/reordered ones.

### Phase 3 — Systems integration (Week 3)
- Wire voice alerts (pyttsx3), structured logging (JSON/CSV), video writer, streaming endpoint (Flask MJPEG or ffmpeg RTSP).
- Build the GUI (PySide6) and connect it to the live pipeline.
- Move to real-time webcam; profile and optimize for CPU (frame skipping, lower-res inference, ONNX export).

### Phase 4 — Polish, edge deployment, pitch (Week 4)
- If pursuing the orientation bonus, bolt on a pretrained HMR model (4D-Humans/ROMP) now — inference-only, never train.
- Test standalone with no internet — literally unplug it.
- Package: demo video, architecture diagram, README, pitch framing against real BAS/space-station constraints (bandwidth, microgravity, comms delay).

## Dataset strategy (start first — the bottleneck)

- **Custom dataset is the most important asset** — public HAR datasets (EPIC-KITCHENS, NTU RGB+D, Breakfast) are transfer-learning resources, not substitutes for the exact protocol.
- **Do not collect only clean demonstrations.** Negative and abnormal cases (skips, repeats, wrong order, idle/ambiguous/partial actions, occlusions, lighting variation, multiple people, different speeds/orientations) are essential for sequence validation.
- Use MediaPipe (pretrained) to auto-extract pose/hand landmarks from clips → this becomes the feature dataset, not raw pixels. Keeps training CPU-friendly.
- Label each clip's ground-truth step sequence.
- Augment: speed variation, lighting variation, mirrored orientation (partially addresses orientation without full 3D HMR).
- YOLO object-detection dataset: ~150–500 annotated frames of props (astronaut, tool_A, tool_B, container, button, rack, rack_slot_*). Planning estimate, not an achieved size.
- 500–1500 annotated images is a planning estimate only — never report as achieved.

## Crash-mode priorities (36h crunch)

Cut from the bottom if time runs out:

1. FSM validator working on pre-recorded video with simple/heuristic step detection (nearest-object + dwell time) — the core differentiator.
2. MediaPipe pose/hand overlay + YOLO detection running live on webcam.
3. Voice alert + structured log file.
4. Minimal GUI (even a single OpenCV window with text overlays).
5. Local video recording.
6. Streaming to an IP (MJPEG is fine).
7. 3D HMR bonus — skip entirely under time pressure; mention as future work.

## Upgrade paths

| Component | Baseline (PoC) | Upgrade | When |
|---|---|---|---|
| Step classifier | XGBoost on features | LightGBM → GRU/1D-CNN/TCN → ST-GCN/video backbone | baseline underperforms / more data |
| Detection | stub (hand-object heuristics) | YOLOv8n fine-tuned, ONNX export | after step classifier works |
| Pose | MediaPipe | MoveNet / RTMPose / YOLO-pose | dependency or accuracy issues |
| Hands | MediaPipe Hands | 100DOH hand-object detector | heuristic too noisy |
| Orientation | basic normalization | pretrained HMR (4D-Humans/ROMP/SMPL) | only if everything else solid |
| Streaming | — | Flask/aiohttp MJPEG → ffmpeg RTSP | when streaming is required |
| Edge export | — | ONNX Runtime → TensorRT (Jetson) / TFLite (Pi) | final packaging |

## Team division (6 members)

- **Hardware (2):** camera, mounting, lighting, payload-rack setup, physical experiment apparatus, camera calibration, IP/RTSP networking.
- **Frontend (2):** PySide6 GUI, live video, experiment state, timeline, logs, alerts, visualization.
- **Backend + ML (1):** video pipeline, YOLO inference, MediaPipe, feature extraction, XGBoost, FSM, JSON/CSV, TTS.
- **Product / AI integration (1):** dataset pipeline, training, evaluation, model optimization, integration, final demo.

## Strongest demo (target)

1. Astronaut begins experiment.
2. System detects the current step.
3. Astronaut intentionally skips Step 3.
4. System detects the invalid transition.
5. Voice: "Step 3 skipped. Please attach Tool A."
6. GUI marks the step as invalid.
7. Event written to the structured log.
8. Video continues recording/streaming.

This demonstrates multiple problem-statement requirements in one scenario. Planned demonstration, not an achieved result unless implemented and tested.

## Non-goals / guardrails

- Full orientation-agnostic 3D HMR is never claimed as implemented unless actually built and validated.
- No over-engineering, microservices, Docker/K8s.
- Real BAS payload / microgravity rig is not required for the demo — a webcam toy-protocol rig suffices.
- Never invent accuracy, FPS, latency, or dataset size in any report or pitch.