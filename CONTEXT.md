# BAS Experiment Assistant

Onboard AI assistant that observes an astronaut performing a predefined scientific experiment, recognizes which protocol step is happening, validates it against the expected sequence, and guides/records the outcome. Built for the SIH 2026 ISRO problem statement.

## Language

**Experiment Protocol**:
The ordered, predefined sequence of steps a scientific experiment must follow. The source of truth the system validates against.
_Avoid_: Procedure, recipe

**Experiment Step**:
A single discrete action within a protocol (e.g., "open sample tray", "pick sample", "place under scope"). The atomic unit of recognition and validation.
_Avoid_: Action, activity, stage

**Step Recognition**:
The act of classifying the current observed behavior into an Experiment Step (or a background/no-step class). Performed by a small trained model (XGBoost) over fused pose/hand/object features.
_Avoid_: Activity classification, action recognition

**Sequence Validation**:
Deterministic logic (a finite state machine) that checks whether an observed Step is allowed given the current protocol state. Produces `confirmed`, `skipped`, `repeated`, or `out-of-sequence` outcomes.
_Avoid_: Sequence model, learned validation

**Step Classifier**:
The model component that maps a window of features to a predicted Step label. Distinct from the validator: it answers "what is happening?", not "is it allowed here?".
_Avoid_: HAR model

**Protocol State**:
The FSM's current position in the Experiment Protocol, tracking which steps are done, which is expected next, and what outcomes (skipped/out-of-sequence) have occurred.
_Avoid_: Current state, session state

**Payload**:
The experiment apparatus and rack that serves as the spatial reference for the scene. Enables payload-relative spatial reasoning (hand relative to rack, tool relative to rack) instead of assuming a fixed floor/up-down orientation.
_Avoid_: Rack, experiment bench, workbench

**Hand-Object Interaction**:
The relationship between a hand and the nearest experiment object (distance, contact, which object is held). A stronger step-identity signal than pose alone.
_Avoid_: Manipulation, grasping

**Pose / Keypoints**:
Body landmarks (e.g., MediaPipe 33-point skeleton) extracted per frame, used as the geometric basis for features. Represented independently of the underlying pose model.
_Avoid_: Skeleton data, landmarks

**Observation**:
A timestamped record of what the system detected at a point in time: person detected, pose estimated, step predicted, or event raised.
_Avoid_: Detection, reading

**Event**:
A significant, logged occurrence during a session (step confirmed, step skipped, out-of-sequence detected). Drives the GUI log and the structured record.
_Avoid_: Alert, message, notification

**Session**:
One continuous run of the system against one experiment execution, from start to stop. Produces one structured record (JSON/CSV) and one video/log stream.
_Avoid_: Run, recording

**Vertical Slice**:
A single end-to-end path through the system that works with real components (webcam → pose → features → step → FSM → log → GUI), proving the whole loop rather than all subsystems at full depth.
_Avoid_: Demo, feature, module