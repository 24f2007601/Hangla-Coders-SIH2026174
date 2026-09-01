# Space-Themed Dashboard Design & Implementation Plan

> **Status:** Planning — Awaiting design mockup/screenshots from user before implementation.
> **Theme Concept:** Spaceflight Mission Control / ISRO BAS On-board Assistant Interface.

---

## 🚀 Concept & Theme Direction

The dashboard will be redesigned into a **space-themed Mission Control HUD (Heads-Up Display)** tailored for Human Spaceflight / BAS Experiments.

### Design Aesthetics & Color Palette
- **Background:** Deep Space Charcoal / Obsidian (`#0B0E17`, `#121824`)
- **Panels & Containers:** Semi-transparent glassmorphic cards with subtle cyan/blue borders (`rgba(0, 229, 255, 0.15)`)
- **Primary Accent / Telemetry:** Electric Cyan / Mission Blue (`#00E5FF`, `#2979FF`)
- **Pass / Nominal Gates:** Emerald Orbit Green (`#00E676`)
- **Warning / Out-of-Sequence:** Solar Amber (`#FFAB00`)
- **Critical Failure / Error:** Pulsing Nova Red (`#FF5252`)
- **Typography:** Clean, high-readability monospace / technical sans-serif fonts (e.g., Segoe UI / Roboto / JetBrains Mono styling via QSS).

---

## 🛠️ Dashboard Architecture & Layout Plan

```text
+-----------------------------------------------------------------------------------+
|  🚀 ISRO BAS ASSISTANT -- MISSION CONTROL DISPLAY                       [LIVE 🟢] |
+--------------------------------------------------+--------------------------------+
|                                                  | 📊 TELEMETRY & SYSTEM STATUS   |
|                                                  |   - Status: NOMINAL / ERROR    |
|                                                  |   - Pipeline FPS: 30.0         |
|                                                  |   - Latency: 12.4 ms           |
|                                                  |                                |
|             LIVE VIDEO FEED                      +--------------------------------+
|    (with Pose Mesh & YOLO HUD Overlays)          | 📦 PAYLOAD STATE (YOLO)        |
|                                                  |   - Phone Screen: ILLUMINATED  |
|                                                  |   - Case: OPEN                 |
|                                                  |   - Receiver: PLUGGED IN       |
|                                                  |                                |
|                                                  +--------------------------------+
|                                                  | ⚡ VERIFICATION GATES (LEDs)    |
|                                                  |   - Receiver L/R: 🔵/🌟         |
|                                                  |   - Gate G1 (Conn): PASSED 🟢   |
|                                                  |   - Gate G2 (Pair): PENDING 🟡  |
+--------------------------------------------------+--------------------------------+
| 📜 EVENT LOG & PROTOCOL GUIDANCE                                                  |
| 18:14:02  [CONFIRMED] M2: Open microphone case                                    |
| 18:14:15  [GATE G1] Receiver connection confirmed                                 |
+-----------------------------------------------------------------------------------+
| [▶ START SESSION]       [⏸ PAUSE SESSION]       [⏹ STOP SESSION]                 |
+-----------------------------------------------------------------------------------+
```

---

## 📋 Key Panels & Components

### 1. Primary HUD Video Feed
- Real-time video frame display with custom dark border and corner brackets.
- Overlays MediaPipe skeleton/hands + bounding boxes for objects (phone, case, receiver, mic).

### 2. Telemetry & System Status
- Live FPS meter, inference latency, person count, and session status indicator.

### 3. Payload State Panel (YOLO Detections)
- Displays current payload states:
  - Phone Screen (`ON` / `OFF`)
  - Microphone Case (`CLOSED` / `OPEN`)
  - Receiver Presence & Port Connection

### 4. Verification Gates Panel (LED Detection)
- Real-time LED status visualization (`BLINKING`, `STEADY`, `OFF`, `UNKNOWN`).
- Verification Gate G1 (Receiver Connection) and Gate G2 (1-Mic Pairing) indicators with glowing status badges.

### 5. Event Telemetry Log
- Log of confirmed steps, skipped steps, warnings, and gate updates styled with color-coded timestamps.

### 6. Control Panel
- Mission control buttons (`START SESSION`, `PAUSE`, `STOP`).

---

## ⏸️ Next Steps (Waiting for User Input)

- [ ] User will provide reference screenshots / mockups for the visual design.
- [ ] Incorporate user's color palette, icons, and visual preferences into Qt Style Sheets (QSS).
- [ ] Proceed with implementation once design approval is given.
