# Success Criteria — What "Done" Means

## PoC acceptance criteria (the 5-day build is done when)

1. The project installs and imports cleanly (`uv sync`).
2. `pytest` passes, including the no-GPU/no-camera integration test.
3. The dummy pipeline executes end-to-end.
4. `python scripts/run_demo.py` starts correctly (webcam or sample video; clear error if no webcam).
5. The full vertical slice runs: webcam → pose → features → XGBoost step → FSM → JSON log → GUI showing step status and events.
6. The FSM correctly produces `confirmed` / `skipped` / `repeated` / `out-of-sequence` outcomes on the toy protocol, demoed via the skip-Step-3 scenario.
7. README accurately reflects what is and is not implemented.

## Demo scenario (the graded centerpiece)

The **skip-step** scenario proves the product's core differentiator — protocol-aware validation, not generic HAR:

1. Astronaut begins the experiment.
2. System detects the current step (confirmed).
3. Astronaut intentionally **skips Step 3** (e.g., "Place sample under scope").
4. System detects the invalid transition → flags `skipped` / `out-of-sequence`.
5. GUI marks the step as invalid and suggests the expected next step.
6. Event written to the structured JSON/CSV log.
7. Video continues (recording/streaming if implemented).

A working end-to-end system at ~90% step accuracy beats a fancy model that only runs in a notebook. Judges reward integration.

## Product-level success framing

- **Positioning:** "AI-Based Autonomous Experiment Execution and Validation Assistant for Human Spaceflight" — not "a HAR system."
- **Function:** Observe → Understand → Predict → Validate → Guide → Record.
- **Novelty claim (defensible):** the *integration* of activity recognition + human–object–payload interaction + protocol state specifically for onboard experiment sequence validation — not that HAR/detection/FSM are new inventions.
- **Differentiator question:** not "What activity is the person performing?" but "What experiment step is being performed, is it valid at this point, and what should happen next?"

## What NOT to claim

Never claim, unless implemented and validated:

- full orientation-agnostic 3D HMR
- microgravity testing or real BAS payload validation
- specific accuracy/FPS/latency/dataset-size numbers
- "mission-ready" or production edge deployment
- successful demo until actually run

Always label status: `Implemented / Tested / Planned / Proposed / Target / Assumption`.

## Product-level success (beyond the PoC)

The system is a success when it can, offline on a standalone machine:

1. Recognize and validate the sequence of a predefined experiment.
2. Suggest the next step at each step boundary.
3. Alert (voice) on skipped or out-of-sequence steps.
4. Generate a timestamped structured lightweight record of conducted steps and outcomes.
5. Record video locally and stream to a specified IP (later).
6. Run entirely without internet.

The PoC de-risks this by proving items 1, 2, 4, and the GUI on a toy protocol. Items 3 (voice), 5 (streaming), and edge packaging are the post-PoC roadmap (`docs/implementation-roadmap.md`).