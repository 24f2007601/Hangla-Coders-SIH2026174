# Wireless Microphone Connection — Experiment Protocol and Training Runbook

**Purpose:** build one working, camera-based Vertical Slice by the end of the next dataset-and-training session. The system must observe a person connecting one wireless microphone to a phone, recognize the Experiment Steps, and validate the successful pairing condition.

## Scope and definition of success

The prototype is successful when one fixed-camera Session produces all of the following:

1. recognizes the observed Experiment Steps in order;
2. verifies the receiver connected to the phone from its LED state;
3. verifies that exactly one microphone paired to the receiver;
4. emits a structured `confirmed`, `skipped`, `repeated`, or `out-of-sequence` Event; and
5. shows the Experiment Protocol status in the existing dashboard or demo window.

This is a constrained webcam prototype, not a claim of general-purpose object recognition, field reliability, or spaceflight readiness.

## Experiment Protocol

### Payload

Place these items on a single plate within a fixed camera view:

- phone;
- USB thumb drives (distractors — they are **not** protocol objects);
- glasses (distractor — not a protocol object);
- wireless microphone case containing the receiver and microphones.

### Experiment Steps and Verification Gates

| ID | Experiment Step / Verification Gate | Observable evidence required to confirm it |
|---|---|---|
| M0 | Verify phone powered on | **Assumption:** the phone display is visibly illuminated for a sustained period. Replace this with a more exact visual state if the intended screen is known. |
| M1 | Experiment Step | M1  Move phone to working station | Phone remains near hand, on the workstation |
| M2 | Pick microphone case | Case is held/closest to a hand for a dwell period. |
| M3 | Open microphone case | Case is visibly in its open state. |
| M4 | Remove receiver from the case | Receiver is visible and transitions from case region to a hand. |
| M5 | Connect receiver to phone | Receiver is handled at the phone connection area. |
| G1 | Verify receiver connection | Both receiver blue LEDs blink. This is a **Verification Gate** following M4, not merely an Experiment Step. |
| M6 | Remove one microphone from the case | A microphone transitions from the case region to a hand. |
| G2 | Verify one-microphone pairing | Exactly one receiver blue LED is steadily illuminated while the other continues blinking, sustained for the chosen verification duration. |

`G2` is the protocol's final success condition. Do not report it as passed when both LEDs blink, when both LEDs are steady, or when the LED state cannot be seen.

### Expected and abnormal scenarios to record

Record each scenario as an independent Session; do not create a train/validation/test split by randomly mixing neighbouring frames from the same video.

| Scenario | Expected result |
|---|---|
| Correct sequence through G2 | Protocol complete |
| Phone display off | G0 fails / do not proceed |
| Case picked but never opened | M2 missing |
| Receiver removed before the case is open | M3 out-of-sequence |
| Microphone removed before receiver connection | M5 out-of-sequence |
| Receiver plugged in but both LEDs never blink | G1 fails or is inconclusive |
| Microphone removed but both receiver LEDs still blink | G2 fails or remains pending |
| Two microphones paired, if possible | G2 fails because the required one-steady/one-blinking pattern is absent |
| Idle, hands crossing the frame, handling thumb drive/glasses | `background`; no Experiment Step Event |

## System design

YOLO26s, MediaPipe, and the Step Classifier have different jobs. Do not train one model to do all three.

```text
Camera frame
  ├─ YOLO26s Payload Detector ───────> object boxes / object-state labels
  ├─ MediaPipe ──────────────────────> Pose and hand keypoints
  └─ Receiver crop + LED estimator ──> per-LED brightness over time
                                            ↓
                          fused temporal feature window
                                            ↓
                       XGBoost Step Classifier (M0–M5/background)
                                            ↓
                   Experiment Protocol FSM + Verification Gates (G1/G2)
                                            ↓
                             Event log and dashboard guidance
```

### 1. YOLO26s: Payload detection and visible object states

Fine-tune YOLO26s to locate objects and robustly visible states, not to decide an entire Experiment Step from one frame. The recommended initial classes are:

```text
phone_screen_on
microphone_case_closed
microphone_case_open
receiver
microphone
```

Only add a class when it is visually separable and has enough examples. Do **not** label USB thumb drives or glasses as protocol classes; leave them as background/distractors. If the receiver connector or phone port is not consistently visible, do not invent a `receiver_connected` bounding-box class: use the G1 LED Verification Gate as the reliable connection evidence.

Use every 10th–15th frame from each recording as a starting pool, then remove near-duplicates. Ensure the labelled set includes different hand positions, occlusion, lighting, open/closed case positions, and distractor handling. Bounding boxes must tightly enclose the visible item/state.

Dataset layout:

```text
data/yolo_microphone/
  images/{train,val,test}/
  labels/{train,val,test}/
configs/payload_detection.yaml
```

Example `payload_detection.yaml`:

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

Start from pretrained weights and establish a baseline before tuning aggressively:

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
  freeze=10
```

For a small custom dataset, lower augmentation and learning rate are a sensible starting point; revise only after inspecting the validation images and metrics. The official [YOLO26 training recipe](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/yolo26-training-recipe.md) and [Ultralytics train-mode guide](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/train.md) are the source for these commands and options.

### 2. Receiver LED-state estimator: Verification Gates G1 and G2

Do not use generic YOLO object detection to decide whether an LED is blinking. First use YOLO's `receiver` box to crop the receiver. Then estimate the blue brightness of the left and right LED regions across time.

For the fixed prototype:

1. Record a short calibration video for each receiver state: both blinking; left steady/right blinking; right steady/left blinking; both steady; not visible.
2. Define the two LED regions relative to the receiver crop, or annotate them once if their placement varies.
3. For each LED, calculate a blue-pixel brightness score per frame (HSV thresholding is adequate to start).
4. Over a 2–3 second rolling window, classify each LED as `blinking`, `steady`, `off`, or `unknown` from brightness variation.
5. Emit G1 only when both are `blinking`; emit G2 only when exactly one is `steady` and the other is `blinking` for the configured dwell duration.

If the camera cannot resolve the LEDs, change the rig before collecting more data: move closer, increase resolution, eliminate glare, and lock exposure if possible. An LED state not visible to the camera must be `unknown`, never guessed.

### 3. MediaPipe Pose and Hands: interaction evidence

Retain the existing MediaPipe Pose + Hands estimator. It contributes temporal interaction evidence that YOLO does not supply:

- left/right palm centre;
- hand visibility and motion;
- nearest detected protocol object per hand;
- hand-to-object distance and overlap;
- duration a hand remains near the case, receiver, phone, or microphone;
- pose/hand velocity features already produced by the feature extractor.

### 4. XGBoost Step Classifier: fused temporal Step Recognition

Train the XGBoost Step Classifier on one label per temporal window:

```text
background, M0, M1, M2, M3, M4, M5
```

Its input must be a fixed-length fused feature vector, not raw MediaPipe internals and not raw YOLO predictions. Add to the current feature vector:

- YOLO class confidence and normalized box centre/size for each protocol class;
- left/right hand → object distances and nearest-object identity;
- whether a case state is open/closed;
- receiver proximity to the phone;
- LED-state probabilities / G1-G2 state where available;
- existing normalized Pose, hand, velocity, and window statistics.

The classifier predicts what Experiment Step is happening. The FSM and Verification Gates determine whether it is allowed and successful. Require confidence thresholding, majority-vote smoothing, and an interaction dwell period before sending a Step Event to the FSM.

## Data collection and labelling plan for tomorrow

### Before recording

1. Fix the camera, plate, and phone in their final positions. Keep the receiver LEDs large and unobstructed in the frame.
2. Use stable lighting; avoid reflections on the phone and receiver LEDs.
3. Film a 10-second idle/background clip containing hands, USB thumb drives, and glasses.
4. Film LED calibration clips for all visible connection states.
5. Decide and write down the exact rule for M0 (what phone screen proves it is on).

### Record Sessions

Record at least:

- 10 correct complete Sessions at varied speeds;
- 3 Sessions with receiver-related failure or delayed blinking;
- 3 Sessions where M5 is performed before G1;
- 3 Sessions with one skipped/repeated/out-of-sequence Experiment Step;
- background/distractor footage.

Use at least two different performers if practical. Keep a session manifest with: filename, performer, lighting, scenario, ordered ground-truth labels, and timestamps for every Experiment Step and Verification Gate.

### Split correctly

Split at the **Session** level, approximately 70% train / 15% validation / 15% test. No frames from a single Session may occur in more than one split. Reserve at least one correct and one abnormal Session exclusively for the final live-style evaluation.

### Produce two labelled datasets

1. **YOLO dataset:** image + bounding-box labels for Payload objects/states.
2. **Step dataset:** video timestamp/window + one Experiment Step label; feature extraction later generates the fused numerical feature table for XGBoost.

The existing `scripts/build_dataset.py` and `scripts/train_classifier.py` are the intended starting points for the second dataset. Extend their input/features only after the YOLO detector and LED-state output contracts are agreed.

## End-of-day build and verification order

1. Verify MediaPipe Pose + Hands on a recorded video and retain the existing dummy fallback.
2. Train YOLO26s; inspect false positives/negatives on held-out Sessions, especially phone/receiver/microphone confusion.
3. Implement and test the receiver-crop LED estimator against calibration videos.
4. Extract fused features from only the training Sessions and train XGBoost.
5. Select thresholds using validation Sessions; evaluate only once on the held-out test Sessions.
6. Connect stable Step Recognition results and G1/G2 outputs to the FSM.
7. Run the final demonstration: a correct Session and at least one intentionally invalid Session.

## End-of-day acceptance checklist

- [ ] The phone-on rule is written and visible in the test footage.
- [ ] YOLO detects all five protocol classes in held-out footage; errors are reviewed, not ignored.
- [ ] G1 is emitted only for two blinking receiver LEDs.
- [ ] G2 is emitted only for one steady blue LED plus one blinking blue LED over the verification window.
- [ ] The Step Classifier has a held-out Session evaluation and does not default to `unknown` for all input.
- [ ] The FSM logs correct, skipped, repeated, and out-of-sequence Events from real recorded footage.
- [ ] A correct Protocol Session completes and an abnormal Session visibly fails or remains pending at the right gate.
- [ ] The README/status documentation is updated with measured results only; do not claim accuracy, FPS, or reliability until measured.

## Known constraints and decisions still needed

- **Phone-on definition:** temporarily assumed to mean a sustained illuminated display. Confirm the required screen/state before labelling.
- **LED orientation:** the G2 rule accepts either left-steady/right-blinking or right-steady/left-blinking, unless the hardware assigns a fixed microphone identity that matters.
- **Camera framing:** LED verification will fail if the receiver crop is too small or reflected; the physical rig is part of the solution.
- **Scope control:** prioritize a reliable one-camera, one-microphone prototype. YOLO fine-tuning and the LED estimator are more valuable tomorrow than streaming, voice alerts, ONNX export, or orientation research.
