# Dataset Recording Guide
## Custom Video Dataset — "Sample Analysis" Toy Protocol
### BAS Assistant | ISRO SIH 2026174

> **Status:** Planned — not yet recorded. This guide defines what to record and how.
> Follow every step in order. Consistency matters more than quantity.

---

## Step 1 — Gather Props (5 minutes)

Use any everyday objects. Keep it simple — the classifier learns gestures, not specific objects.

| Protocol Step | Suggested Prop |
|--------------|---------------|
| S1 Open tray | Cardboard box / lunchbox / any container with a lid |
| S2 Pick sample | Small object — coin, eraser, bottle cap, pen |
| S3 Place under scope | Overturned cup/glass = "scope", flat table = surface |
| S4 Adjust focus knob | Bottle cap to twist, marker cap, or any dial-like object |
| S5 Record reading | Pen + notepad / phone screen |
| S6 Close tray | Same box used for S1 |

---

## Step 2 — Camera Setup (10 minutes)

```
✅ Fixed position — do NOT move the camera between clips
✅ Use laptop webcam or phone propped against something stable
✅ Angle: slightly above eye level, looking down at hands and table
✅ Lighting: bright, even — no harsh shadows on hands
✅ Background: plain and consistent (wall, table) — not a busy background
✅ Frame must include: hands, props, upper body (shoulder to table)
✅ Resolution: minimum 720p at 30 FPS
```

**What NOT to do:**
```
❌ Do NOT hold the camera — it must be fixed and still
❌ Do NOT move the camera between clips
❌ Do NOT record in poor or uneven lighting
❌ Do NOT move props to completely different positions between clips
❌ Do NOT change your seating position significantly between clips
```

---

## Step 3 — The Protocol Sequence

Memorize this order before recording. Each step should be a clear, deliberate action:

```
S0  Start         → Sit/stand in front of the table, hands at sides or on lap (idle, ~3 sec)
S1  Open tray     → Open the box/container with both hands (clear opening motion)
S2  Pick sample   → Pick up the small object with one hand (deliberate grasp)
S3  Place sample  → Place the object under the cup/scope (set it down carefully)
S4  Adjust focus  → Twist the bottle cap/dial 2–3 times (clear rotational motion)
S5  Record reading→ Pick up pen, write on notepad (can be pretend)
S6  Close tray    → Close the box/container
S7  Complete      → Return hands to sides or lap (idle, ~3 sec)
```

---

## Step 4 — Recording Checklist (25–35 clips total)

### Correct Runs — 8 to 10 clips
> Perform all 7 steps in the correct order. Vary your speed across clips.

| Clip | Instructions |
|------|-------------|
| Correct run 1 | Normal pace |
| Correct run 2 | Slow and deliberate |
| Correct run 3 | Fast pace |
| Correct run 4 | Dominant hand only for picking/placing |
| Correct run 5 | Use both hands where possible |
| Correct runs 6–10 | Mix of speeds and minor position variation |

---

### Skip Step 3 — 4 to 5 clips (THE CORE DEMO SCENARIO)
> Perform: S0 → S1 → S2 → **SKIP S3** → S4 → S5 → S6 → S7
> After picking up the sample (S2), go directly to adjusting the knob (S4) without placing.

| Clip | Instructions |
|------|-------------|
| Skip S3 run 1 | Go directly from S2 to S4 without pausing |
| Skip S3 run 2 | Pause for 2 seconds where S3 would be, then do S4 |
| Skip S3 runs 3–5 | Vary speed |

---

### Other Skip Scenarios — 4 to 6 clips

| Clip | Steps Performed | What is Skipped |
|------|----------------|-----------------|
| Skip S5 | S0-S1-S2-S3-S4-S6-S7 | Recording reading |
| Skip S2 and S3 | S0-S1-S4-S5-S6-S7 | Picking and placing |
| Skip S4 | S0-S1-S2-S3-S5-S6-S7 | Focus adjustment |
| Skip S1 | S0-S2-S3-S4-S5-S6-S7 | Opening tray |

---

### Repeated Steps — 3 to 4 clips
> Deliberately do the same step twice in a row.

| Clip | What to Repeat |
|------|---------------|
| Repeat S2 | Pick up sample, put it back down, pick it up again |
| Repeat S4 | Adjust knob, stop, then adjust again |
| Repeat S1 | Open tray, close it, then open it again |

---

### Out-of-Order Steps — 3 to 4 clips
> Perform steps in the wrong sequence.

| Clip | Wrong Order Performed |
|------|----------------------|
| S4 before S3 | Adjust knob before placing sample |
| S6 before S5 | Close tray before recording reading |
| S2 then S6 | Pick sample then immediately close tray |
| S3 before S2 | Place without picking first |

---

### Idle / Background — 5 to 6 clips
> No protocol actions. Used to train the background/idle class.

| Clip | What to Do |
|------|-----------|
| Idle 1 | Sit still, hands in lap (10 seconds) |
| Idle 2 | Look around the room, scratch head |
| Idle 3 | Move partially out of frame |
| Idle 4 | Drink water or gesture while talking |
| Idle 5 | Walk past the camera without touching props |
| Idle 6 | Sit at desk, type on keyboard (not protocol) |

---

## Step 5 — Clip Durations

| Clip Type | Duration |
|-----------|----------|
| Full correct run | 20–40 seconds |
| Skip scenario | 15–25 seconds |
| Repeated step | 15–20 seconds |
| Out-of-order | 15–25 seconds |
| Idle / background | 8–15 seconds |

---

## Step 6 — File Organisation

Save recordings in this structure under `data/raw/`:

```
data/
└── raw/
    ├── correct/
    │   ├── run_01.mp4
    │   ├── run_02.mp4
    │   └── ...
    ├── skip_s3/
    │   ├── skip_s3_01.mp4
    │   └── ...
    ├── skip_other/
    │   ├── skip_s5_01.mp4
    │   └── ...
    ├── repeated/
    │   ├── repeat_s2_01.mp4
    │   └── ...
    ├── out_of_order/
    │   ├── oot_s4_before_s3_01.mp4
    │   └── ...
    └── idle/
        ├── idle_01.mp4
        └── ...
```

---

## Step 7 — Create the Labels CSV

After recording, create `data/raw/labels.csv`:

```csv
filename,clip_type,steps_performed,notes
correct/run_01.mp4,correct,S0-S1-S2-S3-S4-S5-S6-S7,normal speed
correct/run_02.mp4,correct,S0-S1-S2-S3-S4-S5-S6-S7,slow speed
skip_s3/skip_s3_01.mp4,skip,S0-S1-S2-S4-S5-S6-S7,S3 skipped directly
skip_s3/skip_s3_02.mp4,skip,S0-S1-S2-S4-S5-S6-S7,S3 skipped with pause
skip_other/skip_s5_01.mp4,skip,S0-S1-S2-S3-S4-S6-S7,S5 skipped
repeated/repeat_s2_01.mp4,repeated,S0-S1-S2-S2-S3-S4-S5-S6-S7,S2 repeated
out_of_order/oot_s4_before_s3_01.mp4,out_of_order,S0-S1-S2-S4-S3-S5-S6-S7,S4 before S3
idle/idle_01.mp4,idle,none,hands at sides
```

---

## Time Estimate

| Task | Time |
|------|------|
| Setting up props and camera | 10 minutes |
| Recording all 30 clips | 60–90 minutes |
| Organising files and filling labels.csv | 20 minutes |
| **Total** | **~2 hours** |

---

## After Recording — Next Steps

Once all clips are saved and `labels.csv` is filled, run:

```powershell
# 1. Extract MediaPipe pose + hand features from all clips
uv run python scripts/extract_features.py --input data/raw/ --labels data/raw/labels.csv --output data/processed/features.csv

# 2. Train the XGBoost step classifier
uv run python scripts/train_classifier.py --features data/processed/features.csv --output models/step_classifier.json

# 3. Run the full live demo with the trained model
uv run python scripts/run_demo.py
```

---

## Quality Checklist (before starting to record)

- [ ] Camera fixed and not moving
- [ ] Lighting is bright and even — no dark shadows on hands
- [ ] All 6 props are on the table within reach
- [ ] Background is plain and consistent
- [ ] Recording resolution is at least 720p at 30 FPS
- [ ] `data/raw/` subfolders are created
- [ ] `data/raw/labels.csv` header row is ready

---

*Status: Planned | Last updated: 2026-08-29 | See `docs/phase-development.md` for context*
