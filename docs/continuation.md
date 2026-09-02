You are now responsible for integrating the already-working BAS Assistant perception/protocol system with our existing dashboard frontend.

IMPORTANT:
This is NOT a greenfield project.
The YOLO + MediaPipe + XGBoost + protocol-validation pipeline is already implemented and tested.
Your job is to build the backend bridge/API between that Python system and the existing dashboard UI.

DO NOT rewrite, retrain, redesign, or unnecessarily modify the ML pipeline.
DO NOT replace YOLO, MediaPipe, XGBoost, the FSM, G1/G2 logic, feature extraction, or the existing video source.
Treat the existing perception pipeline as the source of truth.

Work directly inside the repository and inspect the codebase before making changes.

==================================================
1. PROJECT CONTEXT
==================================================

Repository:
MakInShort/Hangla-Coders-SIH2026174

Main purpose:
AI-assisted autonomous monitoring and validation of a human-spaceflight-style microphone experiment.

Existing perception stack:

YOLO object detection
→ MediaPipe pose/hand tracking
→ microphone-specific derived features
→ XGBoost temporal step classification
→ temporal smoothing / expected-step validation
→ FSM / protocol validation
→ G1/G2 LED verification

Active protocol:

M0: Verify phone powered on
M1: Move phone to working station
M2: Pick microphone case
M3: Open microphone case
M4: Remove receiver
G1: Both receiver LEDs blinking
M5: Connect receiver to phone
M6: Remove one microphone
G2: Exactly one receiver LED steady and the other blinking
Complete

The final protocol ordering is:

M0 → M1 → M2 → M3 → M4 → G1 → M5 → M6 → G2 → protocol_complete

The ML pipeline has already been validated on the actual recorded experiment video.

Do not break that behavior.

==================================================
2. EXISTING ML PERFORMANCE
==================================================

Final fused XGBoost configuration uses 102 features.

Validation:
- Accuracy: 75.62%
- Macro F1: 62.34%
- Balanced Accuracy: 63.23%

Test:
- Accuracy: 71.07%
- Macro F1: 67.44%
- Balanced Accuracy: 76.60%

The final runtime has already successfully completed:

M0
M1
M2
M3
M4
G1
M5
G2
M6
protocol_complete

on:

Dataset/reshoot/re_total_experiment_v5.mp4

The final successful run completed after 912 processed frames.

==================================================
3. YOUR RESPONSIBILITY
==================================================

Your responsibility is ONLY:

Python backend / API / WebSocket bridge
↕
Existing dashboard frontend

The dashboard already exists.

First inspect:
- dashboard source code
- current dashboard entry point
- current dashboard components
- current state management
- existing mock/static data
- current API calls if any
- current run_dashboard.py
- any backend bridge already present
- package/dependency configuration
- repository documentation describing the dashboard

DO NOT rebuild the dashboard from scratch.

Reuse the existing UI.

The objective is to replace mock/static dashboard data with real live pipeline state.

==================================================
4. INSPECT THE REPOSITORY BEFORE EDITING
==================================================

Before making changes, inspect:

- src/bas_assistant/
- scripts/run_dashboard.py
- existing dashboard/frontend directory
- pipeline/factory.py
- pipeline/pipeline.py
- validation/protocol.py
- validation/protocol_evidence.py
- validation/led_estimator.py
- events/models.py
- repository/storage implementation
- video/source.py
- configs/default.yaml
- README.md
- CONTEXT.md
- AGENTS.md

Determine:

1. How ExperimentPipeline is instantiated.
2. What object currently owns the active pipeline state.
3. How process_frame() returns FrameResult.
4. How Event objects are created.
5. How sessions are started/stopped.
6. How the current dashboard expects its data.
7. Whether there is already a server or API implementation.
8. Which frontend framework and package manager are used.
9. Which existing dashboard components should consume which backend fields.

Do not guess architecture until the repository has been inspected.

==================================================
5. REQUIRED LIVE DATA CONTRACT
==================================================

Expose the current runtime state to the dashboard.

The dashboard needs at minimum:

SYSTEM STATE
- running
- paused
- stopped
- error
- completed

SESSION
- session_id
- session start time
- elapsed time
- source type
- webcam/file indicator

VIDEO
- width
- height
- source FPS
- actual/acquisition FPS
- current frame number
- inference latency

CURRENT STEP
- current classified step
- current confidence
- current accepted FSM step
- human-readable step name
- progress through protocol

PROTOCOL
- all M0-M6 states
- pending/current/completed state
- current protocol phase
- protocol completion status

GATES
- G1 status
- G2 status
- gate_status
- left LED state
- right LED state
- receiver detected
- receiver confidence
- receiver bounding box if available

EVENTS
- event type
- event timestamp
- event message
- step/gate associated with event where available
- recent event history

ERRORS
- frame errors
- pipeline errors
- source errors

==================================================
6. CAMERA FEED
==================================================

The dashboard should show the REAL webcam feed.

Prefer showing the already annotated frame produced by the Python pipeline rather than creating a second independent computer-vision implementation in the frontend.

If the existing visualization/annotation function is already available, reuse it.

The browser should receive frames through an appropriate streaming mechanism.

Preferred implementation:
- WebSocket for realtime state/events
- WebSocket or MJPEG-compatible stream for annotated video

Choose the simplest robust architecture compatible with the existing project.

Do not introduce unnecessary infrastructure such as Kafka, Redis, Kafka-like queues, or microservices.

This is a hackathon application.

==================================================
7. WEBSOCKET / API DESIGN
==================================================

Create a clean backend bridge.

Suggested architecture:

Browser
   ↕ WebSocket
Python backend bridge
   ↕
ExperimentPipeline
   ↕
OpenCV webcam
   ↕
YOLO / MediaPipe / XGBoost / FSM

The backend should be authoritative.

The frontend must NOT independently calculate:
- step transitions
- confidence
- gate state
- LED state
- protocol completion

It only displays backend state.

Suggested message structure:

{
  "type": "state",
  "timestamp": 1234567890,
  "session_id": "...",

  "system": {
    "status": "running",
    "error": null
  },

  "video": {
    "frame": 1234,
    "fps": 22.4,
    "latency_ms": 43.2,
    "width": 1280,
    "height": 720
  },

  "classification": {
    "step": "M4",
    "label": "Remove receiver",
    "confidence": 0.91
  },

  "protocol": {
    "current_step": "M4",
    "completed_steps": ["M0", "M1", "M2", "M3"],
    "complete": false
  },

  "gates": {
    "status": "G1_PENDING",
    "g1": {
      "passed": false
    },
    "g2": {
      "passed": false
    },
    "receiver_detected": true,
    "receiver_confidence": 0.77,
    "left_led": "blinking",
    "right_led": "blinking"
  },

  "events": []
}

You may improve this schema after inspecting the frontend, but keep it stable and documented.

==================================================
8. COMMANDS FROM DASHBOARD
==================================================

The dashboard should be able to control the experiment session.

At minimum implement:

START / PLAY
- start webcam
- create/start pipeline session
- begin processing
- begin streaming state/video

PAUSE
- pause processing without destroying session state

STOP
- stop webcam processing
- close/end pipeline session
- preserve completed session data

RESET / NEW SESSION
- start clean session
- reset FSM
- reset vote buffers
- reset LED history
- reset gate state

The existing pipeline's own lifecycle methods must be used.

Do not duplicate session logic in the frontend.

==================================================
9. IMPORTANT PROTOCOL RULES
==================================================

The backend must preserve these invariants exactly.

G1:
Both receiver LEDs must be blinking.

G2:
Exactly one receiver LED must be steady and the other blinking.

Never pass a gate when:
- receiver is missing
- receiver is not confidently localized
- LED state is unknown
- invalid LED combination is detected

M5:
Cannot be committed before G1.

M6:
Cannot be committed before G2.

protocol_complete:
Cannot occur before M6.

Do not create alternate protocol logic in the API layer.
The existing ExperimentPipeline/FSM remains the source of truth.

==================================================
10. EVENTS TO FORWARD TO FRONTEND
==================================================

Forward backend events such as:

session_started
step_confirmed
step_skipped
out_of_sequence
gate_g1_pending
gate_g1_passed
gate_g2_pending
gate_g2_passed
protocol_complete
pipeline_error

For example:

{
  "type": "event",
  "event": {
    "type": "gate_g1_passed",
    "timestamp": 1234567890,
    "message": "G1 passed: both receiver LEDs are blinking."
  }
}

Frontend should append these to the existing event/activity panel.

==================================================
11. DASHBOARD UI MAPPING
==================================================

Map existing dashboard components to real backend data.

Typical mapping:

Camera panel
→ annotated live frame

Current Step
→ current classifier + accepted FSM state

Confidence
→ XGBoost confidence

Protocol progress
→ accepted M0-M6 states

Gate indicator
→ gate_status

LED indicators
→ left_led / right_led

Receiver status
→ receiver_detected + receiver_confidence

System metrics
→ FPS + latency + frames

Activity log
→ Event objects

Completion indicator
→ protocol_complete

Do not create duplicate frontend-derived state when backend already provides the state.

==================================================
12. WEBCAM MODE IS THE TARGET
==================================================

The final user workflow should be:

1. Open dashboard.
2. Click PLAY / START.
3. Browser requests/activates backend webcam session.
4. Python backend starts OpenCV webcam.
5. Frames flow through:
   YOLO
   MediaPipe
   feature extraction
   XGBoost
   FSM
   G1/G2
6. Backend streams:
   annotated video
   current state
   events
   metrics
7. User physically performs the microphone experiment.
8. Dashboard updates live.
9. When G2 passes and M6 completes:
   protocol_complete
10. Dashboard clearly shows experiment completion.

==================================================
13. ERROR HANDLING
==================================================

Handle gracefully:

- webcam unavailable
- camera already in use
- model loading failure
- disconnected WebSocket
- pipeline exception
- invalid frame
- frontend refresh
- user stopping session
- browser reconnecting

The browser should show meaningful error states rather than crashing.

The backend should never die silently.

==================================================
14. PERFORMANCE
==================================================

Keep latency low.

Do not:
- run the ML pipeline twice
- run inference independently in frontend
- duplicate YOLO inference
- duplicate MediaPipe inference
- buffer unlimited frames
- block the WebSocket event loop

Use a small bounded frame buffer / latest-frame strategy if necessary.

The goal is responsive real-time operation around the existing ~20+ FPS processing performance.

==================================================
15. TESTING REQUIREMENTS
==================================================

Before finishing:

1. Run existing unit tests.
2. Run integration tests.
3. Verify existing ML pipeline still works.
4. Verify webcam backend starts.
5. Verify WebSocket connection.
6. Verify live state updates.
7. Verify events reach frontend.
8. Verify START/PAUSE/STOP.
9. Verify protocol completion.
10. Test a disconnected client/reconnect.
11. Test webcam failure.

Most importantly:
Do not regress the already-working recorded-video pipeline.

==================================================
16. DO NOT MODIFY THESE UNNECESSARILY
==================================================

Treat these as stable unless absolutely required:

- YOLO weights
- XGBoost model
- feature extraction logic
- MediaPipe logic
- LED estimator
- G1/G2 logic
- FSM protocol rules

If an integration problem appears, adapt the bridge rather than rewriting the ML pipeline.

==================================================
17. CODE QUALITY
==================================================

Keep the implementation simple and hackathon-friendly.

Prefer:
- one backend service
- one WebSocket protocol
- minimal dependencies
- clear message schema
- clean separation between ML pipeline and API layer

Add type hints and useful logging.

Document:
- how to start the backend
- how to start the dashboard
- how the WebSocket works
- message schema
- webcam requirements

==================================================
18. FINAL VALIDATION
==================================================

After implementation, demonstrate:

Browser
↓
START
↓
Webcam
↓
YOLO + MediaPipe + XGBoost
↓
live dashboard
↓
M0
↓
M1
↓
M2
↓
M3
↓
M4
↓
G1
↓
M5
↓
M6
↓
G2
↓
protocol_complete

Use the existing recorded video as a backend integration test first if webcam testing is inconvenient.

The final result should make the existing dashboard a live control/monitoring interface for the already-working BAS Assistant perception pipeline.

==================================================
19. IMPORTANT WORKING STYLE
==================================================

Do not stop after implementing one endpoint.

Inspect the entire existing architecture first, then implement the complete bridge.

Do not repeatedly ask me to fix small errors manually.

After modifying code:
- run tests
- start the backend
- connect dashboard
- exercise the flow
- fix integration errors
- verify the whole path

At the end, give me:

1. Files changed
2. Backend architecture chosen
3. WebSocket/API schema
4. How the dashboard maps to the pipeline
5. Exact commands to run backend + frontend
6. Tests performed
7. Any remaining limitation

Most importantly:

THE EXISTING ML PIPELINE IS ALREADY WORKING.
YOUR JOB IS TO CONNECT IT TO THE DASHBOARD RELIABLY, NOT TO REBUILD IT.