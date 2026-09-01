# Tomorrow's Handbook — Wireless Microphone Dataset, Annotation, Training, and Prototype

**Use this as the single operational guide for the team session.** It turns the wireless-microphone Experiment Protocol into three small, cooperating perception systems:

1. **YOLO26s Payload Detector** — where are the relevant objects and visible case/screen states?
2. **Receiver LED-State Estimator** — did the receiver connect and did exactly one microphone pair?
3. **XGBoost Step Classifier** — what Experiment Step is occurring over a short window, based on Pose, Hands, object relationships, and LED state?

The deterministic FSM validates only stable, recognized Experiment Steps and Verification Gates. It must never make decisions directly from a single raw camera frame.

Related Protocol definitions are in [microphone-connection-protocol-training-runbook.md](microphone-connection-protocol-training-runbook.md).

## 0. What must be true by the end of the day

The minimum working prototype is a **recorded-video Vertical Slice**. A live webcam extension is desirable only after the recorded video works.

| Required result | Evidence |
|---|---|
| Payload objects/states detected | Annotated held-out frames show phone-screen-on, case state, receiver, and microphone boxes correctly. |
| Receiver connection verified | G1 only appears when both receiver LEDs blink. |
| One microphone pairing verified | G2 only appears when one LED is steady illuminated and the other blinks over a dwell window. |
| Experiment Steps recognized | The classifier produces meaningful M0–M5/background predictions on a held-out Session. |
| Sequence Validation works | One correct Session completes; one deliberately abnormal Session logs the right Event. |

Do not claim accuracy, FPS, or robustness until measured and written down. A stable recorded-video demonstration is a successful day-one result.

## 1. The Experiment Protocol to collect

### 1.1 Payload and camera scene

Put these on one plate, entirely inside the camera view:

- phone;
- microphone case containing receiver and microphones;
- USB thumb drives and glasses as deliberate distractors.

Use one fixed camera and one fixed plate layout for the first prototype. Frame the phone, case, receiver LED face, hands, and upper torso. The receiver LEDs must be sufficiently large, sharp, and free of glare; LED verification is impossible if they occupy only a few pixels.

### 1.2 Experiment Steps and Verification Gates

| ID | Type | Description | Required observable evidence |
|---|---|---|---|
| M0 | Verification-oriented Experiment Step | Verify phone powered on | The phone display is visibly illuminated for the chosen dwell time. This is the current assumption; replace it if the exact required display state becomes known. |
| M1 | Experiment Step | Move phone to working station | Phone remains near hand, on the workstation |
| M2 | Experiment Step | Pick microphone case | Case transitions to / remains nearest to a hand. |
| M3 | Experiment Step | Open microphone case | Open-case visual state appears. |
| M4 | Experiment Step | Remove receiver | Receiver transitions from case region to a hand. |
| M5 | Experiment Step | Connect receiver to phone | Receiver is handled at the phone connection region. |
| G1 | Verification Gate | Confirm receiver connection | Both blue receiver LEDs blink. |
| M6 | Experiment Step | Remove one microphone | A microphone transitions from case region to a hand. |
| G2 | Verification Gate | Confirm one-microphone pairing | Exactly one blue LED is steady illuminated and the other is blinking, for the chosen dwell time. |

**Important:** G1 and G2 are state checks, not ordinary action labels. They must report `pending` or `unknown` when the receiver is hidden or its LEDs cannot be resolved. Never infer success from elapsed time alone.

### 1.3 Receiver LED truth table

| Left LED | Right LED | Interpretation |
|---|---|---|
| blinking | blinking | G1 passed: receiver is connected; no microphone pairing confirmed |
| steady | blinking | G2 passed: exactly one microphone paired |
| blinking | steady | G2 passed: exactly one microphone paired |
| steady | steady | Not the required one-microphone state |
| unknown/off/hidden | any | Inconclusive — do not pass a Verification Gate |

## 2. Before recording: setup and calibration

### 2.1 Camera checklist

- [ ] Camera is fixed; do not hand-hold it.
- [ ] Capture at 1080p if possible, 30 FPS minimum.
- [ ] Phone, case, receiver, hands, and LED face remain in frame.
- [ ] Lighting is bright and even; eliminate reflections on the phone screen and blue LEDs.
- [ ] Camera auto-exposure does not make the LEDs disappear; lock exposure if the camera allows it.
- [ ] Record a 20-second trial and inspect it at full resolution before recording the dataset.

If the receiver LED region is too small or saturated, move the camera closer or use a higher-resolution crop. Do not compensate by collecting more unusable footage.

### 2.2 Environment setup

Run these from the repository root:

```bash
uv sync --extra ml
uv sync --extra ui
uv add ultralytics
uv run python scripts/download_mediapipe_models.py
```

`ultralytics` is not currently a project dependency, so the third command intentionally updates the project dependency/lock files. Run it once, commit the dependency change separately, and keep the trained weight files out of Git.

Check the baseline before modifying the pipeline:

```bash
uv run pytest -q
uv run ruff check .
uv run black --check src scripts tests
```

At the time this handbook was written, four pose tests fail because hand-observability counter fields are absent in a test-created estimator. Fix that regression before declaring the baseline healthy; do not hide or skip those tests.

### 2.3 Create the working directories

```text
data/
  raw/microphone_sessions/             # original recordings; never edit in place
  annotations/
    sessions.csv                       # one row per Session
    steps.csv                          # timestamp ranges for M0–M5/background
    led_states.csv                     # timestamp ranges for LED states
  yolo_microphone/
    images/{train,val,test}/
    labels/{train,val,test}/
  processed/microphone/                # derived feature data and reports
models/                                 # ignored trained artifacts
configs/
  microphone_protocol.yaml             # protocol/model settings when implemented
  payload_detection.yaml               # YOLO dataset definition
```

Keep raw video immutable. Derive sampled images, YOLO labels, crops, features, and reports into the other folders so mistakes are recoverable.

## 3. Record the Sessions

### 3.1 Recording rules

Each video is one **Session**. Begin with 2–3 seconds of idle and finish with 2–3 seconds where the receiver is visible. Speak the planned scenario aloud before recording if that helps later annotation; remove audio from the final demonstration if desired.

Name every file consistently:

```text
YYYYMMDD_<performer>_<scenario>_<take>.mp4
20260831_mainak_correct_01.mp4
20260831_mainak_mic_before_receiver_01.mp4
```

### 3.2 Minimum Session list

| Session family | Minimum | What to perform |
|---|---:|---|
| Correct | 10 | M0 → M1 → M2 → M3 → M4 → G1 → M5 → G2 |
| Receiver connection delay/failure | 3 | Connect receiver but delay G1 or keep LEDs unobservable/both not blinking |
| Mic-before-receiver | 3 | Perform M5 before G1 |
| Skip/repeat/out-of-sequence | 3 | Omit or repeat one Experiment Step deliberately |
| Idle/distractor | 3 | Handle thumb drives/glasses, move hands, or remain idle without protocol steps |
| LED calibration | 2 per visible state | Keep receiver visible while it shows both blinking, one-steady/one-blinking, and any other possible state |

If time permits, use a second performer and repeat the correct and abnormal Session families. Keep the physical setup fixed for this first day; variation in performer and speed is more valuable than random camera movement.

### 3.3 Session manifest — `data/annotations/sessions.csv`

Create one row per video:

```csv
session_id,video_path,split,performer,scenario,lighting,notes
correct_01,data/raw/microphone_sessions/20260831_mainak_correct_01.mp4,train,mainak,correct,bright,normal speed
mic_early_01,data/raw/microphone_sessions/20260831_mainak_mic_before_receiver_01.mp4,val,mainak,mic_before_g1,bright,M5 before G1
correct_heldout,data/raw/microphone_sessions/20260831_second_correct_01.mp4,test,second,correct,bright,never train on this Session
```

Assign the split **before** extracting images/features: approximately 70% `train`, 15% `val`, and 15% `test`. No frame, crop, augmentation, or feature window from one Session may appear in more than one split.

## 4. Annotate the dataset

Use any bounding-box annotation tool that exports standard YOLO text labels. The tool is not important; label consistency is.

### 4.1 YOLO classes

Use these classes and IDs exactly:

```text
0 phone_screen_on
1 microphone_case_closed
2 microphone_case_open
3 receiver
4 microphone
```

Label only visible objects/states. USB thumb drives and glasses are intentionally left as background.

Why `phone_screen_on` rather than `phone`? It allows M0 to require positive screen-on evidence. An unlabelled/off/occluded phone is not proof that it is powered off; it is simply not evidence that M0 passed.

### 4.2 Select frames for YOLO annotation

For each Session:

1. Start by extracting one frame every 0.5–1 second.
2. Add extra frames immediately before, during, and after M1–M5 transitions.
3. Add examples with partial occlusion, both hands, open/closed case, object near the phone, and distractors in the same frame.
4. Remove near-duplicate frames that add no new object position, hand interaction, or illumination condition.
5. Keep all extracted frames in the split assigned to their source Session.

For every object box:

- enclose only the visible object/state, as tightly as practical;
- label a partially covered object only if its class is still unambiguous;
- do not label a guessed receiver inside a closed case;
- do not label a phone with screen off as `phone_screen_on`;
- label both case and its visible contents when each is separately visible.

### 4.3 YOLO label format

Every image has a same-named `.txt` file. One object per line:

```text
<class_id> <centre_x> <centre_y> <width> <height>
```

Coordinates are normalized to `[0, 1]`. Example:

```text
0 0.310 0.486 0.200 0.310
2 0.620 0.540 0.230 0.170
3 0.638 0.555 0.055 0.083
```

Before training, inspect at least 30 random image/label overlays from every split. A label error repeated across a video is more damaging than a smaller dataset.

### 4.4 Step-time annotations — `data/annotations/steps.csv`

Annotate time ranges, not a single label for an entire video. Use `background` for idle and ambiguous periods.

```csv
session_id,start_ms,end_ms,label,confidence,notes
correct_01,0,2200,background,high,idle before protocol
correct_01,2200,4200,M0,high,illuminated phone screen visible
correct_01,4200,6300,M1,high,picking case
correct_01,6300,8600,M2,high,opening case
correct_01,8600,10500,M3,high,receiver leaves case
correct_01,10500,12800,M4,medium,receiver at phone port
correct_01,12800,15000,background,high,awaiting G1; no action
correct_01,15000,17200,M5,high,microphone leaves case
```

Rules:

- One row covers one contiguous observed Experiment Step.
- Do not label waiting time as M4 or M5 merely to fill a gap.
- Mark uncertain ranges with `confidence=low`; exclude them from first training run if needed.
- M0–M5 are Step Classifier labels. G1/G2 belong in the LED-state file, not this file.

### 4.5 LED-state annotations — `data/annotations/led_states.csv`

```csv
session_id,start_ms,end_ms,left_led,right_led,receiver_visible,notes
correct_01,10500,12750,blinking,blinking,true,G1 evidence
correct_01,17200,21000,steady,blinking,true,G2 evidence
led_calibration_01,0,5000,blinking,blinking,true,connection state
```

Use only `blinking`, `steady`, `off`, or `unknown`. When the receiver is covered, set `receiver_visible=false` and each LED to `unknown`.

### 4.6 Annotation quality gate

Do not begin training until all are true:

- [ ] Each required YOLO class has examples in training and validation splits.
- [ ] At least one correct and one abnormal Session are reserved for test.
- [ ] All correct Sessions have M0–M5 ranges and G1/G2 LED-state ranges.
- [ ] A second team member spot-checks 10% of boxes and timestamp ranges.
- [ ] `steps.csv` contains no overlapping non-background Step labels unless the overlap is explicitly justified.

## 5. Train YOLO26s

### 5.1 Dataset configuration — `configs/payload_detection.yaml`

```yaml
path: data/yolo_microphone
train: images/train
val: images/val
test: images/test
names:
  0: phone_screen_on
  1: microphone_case_closed
  2: microphone_case_open
  3: receiver
  4: microphone
```

### 5.2 First training run

```bash
yolo detect train \
  model=yolo26s.pt \
  data=configs/payload_detection.yaml \
  epochs=50 \
  patience=20 \
  imgsz=640 \
  lr0=0.001 \
  mosaic=0.5 \
  mixup=0.0 \
  freeze=10 \
  project=runs/microphone_yolo \
  name=baseline
```

Begin with pretrained `yolo26s.pt`; do not train from scratch. This small-data baseline follows the official [YOLO26 training guidance](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/yolo26-training-recipe.md). Use its saved `best.pt` only after inspecting the validation plots and sample predictions.

### 5.3 YOLO review checklist

```bash
yolo detect val \
  model=runs/microphone_yolo/baseline/weights/best.pt \
  data=configs/payload_detection.yaml \
  split=test

yolo predict \
  model=runs/microphone_yolo/baseline/weights/best.pt \
  source=data/raw/microphone_sessions/<held-out-video>.mp4 \
  save=True
```

Watch the prediction video and answer:

- Are phone-screen-on and phone-screen-off confused?
- Is the receiver detected while held near the phone and while in the open case?
- Are open/closed case states stable?
- Are glasses/thumb drives incorrectly detected as microphones or receiver?
- Are tiny/occluded objects missing? If so, improve camera framing or labels before changing hyperparameters.

Copy the chosen `best.pt` to `models/payload_detector.pt` for local use, but do not commit it.

## 6. Implement and validate the LED-state estimator

This is a lightweight deterministic subsystem. It uses the YOLO receiver box only to find the crop; it should not use a generic detector to decide blinking.

### 6.1 Per-frame processing

1. Run YOLO and select the best `receiver` box.
2. Pad and crop the receiver region.
3. Identify fixed left/right LED regions relative to the crop. Calibrate them on one clear receiver image; record the normalized region coordinates.
4. Convert each LED patch to HSV and calculate a blue brightness score per frame.
5. Keep a rolling 2–3 second score history for each LED.
6. Classify LED state:
   - `steady`: consistently bright with low variation;
   - `blinking`: repeatedly transitions between bright and dark;
   - `off`: consistently dark;
   - `unknown`: receiver/LED not visible, saturated, or too uncertain.
7. Emit G1 only for `blinking/blinking`; emit G2 only for `steady/blinking` or `blinking/steady` for a sustained dwell duration.

### 6.2 Calibrate with video, not intuition

For each LED calibration Session, plot or print the two brightness sequences. Tune thresholds against the labelled `led_states.csv`. Keep a held-out LED calibration clip to verify the chosen thresholds. If the LED patterns overlap due to exposure or reflections, improve the rig; do not silently weaken G1/G2.

### 6.3 Unit tests to add

Create small, deterministic tests from stored brightness sequences:

- both blinking → G1 pass;
- left steady/right blinking → G2 pass;
- right steady/left blinking → G2 pass;
- both steady → no G2 pass;
- receiver hidden → `unknown`, no gate pass;
- one-frame flash/noise → no pass before dwell threshold.

## 7. Build the fused Step Recognition dataset

### 7.1 Current repository limitation

`scripts/build_dataset.py` currently accepts **one label for an entire video** and extracts only MediaPipe-based features. `scripts/train_classifier.py` currently reads a hard-coded synthetic CSV and randomly splits rows. Those scripts are not sufficient for this protocol because:

- real Sessions contain several Experiment Steps;
- each video must be split by Session, never random feature rows;
- YOLO detections and LED state must become features; and
- model labels must be M0–M5/background, not the old Sample Analysis labels.

Therefore, implement these two new scripts tomorrow rather than attempting to force the old ones:

```text
scripts/build_microphone_dataset.py
scripts/train_microphone_classifier.py
```

The existing feature extractor, XGBoost wrapper, MediaPipe estimator, and training script are reference implementations to reuse—not the full solution unchanged.

### 7.2 `build_microphone_dataset.py` contract

Inputs:

```text
--sessions data/annotations/sessions.csv
--steps data/annotations/steps.csv
--led-states data/annotations/led_states.csv
--detector models/payload_detector.pt
--output data/processed/microphone/features.csv
```

For every frame/window in a Session:

1. load frame and its timestamp;
2. obtain MediaPipe Pose and hand keypoints;
3. obtain YOLO object detections;
4. obtain LED-state features when receiver is visible;
5. map the timestamp to its label in `steps.csv`, otherwise `background`;
6. wait until the configured temporal window is full;
7. write one numerical feature row with `session_id`, `split`, timestamp, label, and feature values.

Suggested fused features:

| Feature group | Examples |
|---|---|
| Existing Pose/Hands | normalized wrists, palms, joint angles, velocity, motion statistics |
| Objects | class confidence, normalized centre/size, visible/not-visible flag for every YOLO class |
| Hand–object interaction | distance/overlap from each palm to case, receiver, phone, microphone; nearest object identity |
| Case state | open confidence, closed confidence, state dwell time |
| Receiver/phone relation | receiver-to-phone distance, receiver near phone-port region dwell time |
| LEDs | left/right state scores, G1/G2 candidate flags, receiver visibility |

Do not persist MediaPipe vendor objects in the CSV. Store only standardized numerical features and metadata. Drop windows without a confident step label from the first training run, but retain them in a diagnostics report.

### 7.3 `train_microphone_classifier.py` contract

Inputs/outputs:

```text
--features data/processed/microphone/features.csv
--model-output models/microphone_step_classifier.json
--report-output data/processed/microphone/classifier_report.md
```

Rules:

1. use `split`/`session_id` supplied by the manifest; never call a random row-level split;
2. train only on `train`, choose configuration/thresholds on `val`, and report final results once on `test`;
3. write the label-to-ID map beside the model;
4. print per-class precision/recall/F1 and a confusion matrix, not only a single accuracy number;
5. fail clearly if classes are absent from train/validation or feature columns contain NaNs;
6. save the exact feature-column order beside the model so inference uses the same vector.

Start with an XGBoost multiclass classifier and conservative parameters similar to the existing baseline. Do not upgrade architecture tomorrow unless the data pipeline itself is proven correct.

## 8. Integrate into the pipeline

Implementation order:

1. add a `PayloadDetector` interface/implementation that returns model-independent object detections;
2. add the receiver LED-state estimator that returns explicit `unknown`/state output;
3. extend the feature extractor to consume object/LED observations with Pose and Hands;
4. introduce the microphone Experiment Protocol (M0–M5) and Verification Gates (G1/G2), without modifying the FSM to read raw frames;
5. configure the XGBoost wrapper to load the newly trained model and its feature schema;
6. configure smoothing, confidence, and dwell thresholds from validation footage;
7. show the current Step, expected next Step, gate state, and Events in the demo/dashboard.

The current pipeline only has a full-frame stub detector, the Sample Analysis protocol, a pose-only feature vector, and no LED subsystem. These are implementation tasks—not capabilities that already exist.

## 9. Final demonstration script

Run these two Sessions from the held-out set:

### Session A — correct protocol

1. phone display becomes visible;
2. pick and open microphone case;
3. remove receiver and connect it to phone;
4. show both blue LEDs blinking (G1);
5. remove one microphone;
6. show one blue LED steady and one blinking (G2);
7. demonstrate the final structured Event log and Protocol completion.

### Session B — intentional invalid protocol

Remove one microphone before receiver connection is verified, or skip the receiver step. The system should keep G1/G2 pending and log the appropriate out-of-sequence / skipped result only after a stable incorrect Step is recognized.

Save both annotated output videos and their JSONL Session logs. These are evidence, not performance claims.

## 10. Timeboxed plan

| Time box | Deliverable | Stop condition |
|---|---|---|
| 0:00–0:45 | Camera/LED trial, environment, folder structure | Do not record until receiver LEDs are clearly resolved. |
| 0:45–2:30 | Record all Sessions and calibration clips | Manifest filled in as each video is made. |
| 2:30–4:30 | Sample/annotate YOLO frames and step/LED time ranges | Annotation quality gate passes. |
| 4:30–5:30 | Train/review YOLO26s | Prediction video reviewed; labels/rig fixed if unusable. |
| 5:30–6:30 | LED estimator + tests | G1/G2 pass only for labelled calibration states. |
| 6:30–8:30 | Build fused dataset + train XGBoost | Held-out Session report produced. |
| 8:30–end | Pipeline integration and recorded demo | One correct + one invalid Session demonstrated. |

If time runs short, preserve the product’s core: recorded-video MediaPipe + object interactions + G1/G2 verification + FSM log. Skip streaming, voice alerts, dashboard polish, ONNX export, and advanced model tuning.

## 11. Final checklist

- [ ] Original videos, annotations, trained model files, reports, and output logs are all retained locally.
- [ ] YOLO train/validation/test sets are split by Session.
- [ ] No off/hidden phone is labelled `phone_screen_on`.
- [ ] No hidden receiver LED is treated as successful pairing.
- [ ] XGBoost uses fused temporal features, not a single-frame label.
- [ ] The FSM receives only stable Step/Gate events.
- [ ] Test Sessions were never used to choose labels, thresholds, or hyperparameters.
- [ ] Correct and invalid demonstrations are saved.
- [ ] README/status claims are updated only with results actually measured tomorrow.
