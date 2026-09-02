from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
from ultralytics import YOLO

MODEL = Path("runs/detect/runs/microphone_yolo/baseline-2/weights/best.pt")

VIDEO_ROOT = Path(r"C:\Users\User\Documents\Project\BAS Assistant\Dataset\videos")

ANNOTATIONS = Path("data/annotations/led_states.csv")

OUTPUT_ROOT = Path("data/led_inspection/blink_analysis")

RECEIVER_CLASS_ID = 4

# Calibrated against the 208x145 receiver crop.
LEFT_LED = (55, 30, 78, 62)
RIGHT_LED = (120, 35, 145, 68)

BASELINE_WINDOW = 31
MIN_BLINK_DISTANCE_SECONDS = 0.35
PROMINENCE = 25.0
MIN_BLINKS = 3


def led_score(
    receiver: np.ndarray,
    roi: tuple[int, int, int, int],
) -> float:
    """Measure cyan/blue illumination inside an LED ROI."""

    receiver = cv2.resize(receiver, (208, 145))

    x1, y1, x2, y2 = roi
    region = receiver[y1:y2, x1:x2]

    if region.size == 0:
        return 0.0

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    signal = saturation * value / 255.0

    strong = signal[signal > 50]

    if strong.size == 0:
        return 0.0

    return float(np.percentile(strong, 90))


def detect_blinks(
    signal: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Detect blink events relative to the LED's local baseline."""

    if len(signal) < 10:
        return np.array([], dtype=int)

    signal = np.asarray(signal, dtype=float)

    # Smooth small frame-to-frame noise.
    smoothed = np.convolve(
        signal,
        np.ones(5) / 5,
        mode="same",
    )

    # Rolling median baseline.
    baseline = np.zeros_like(smoothed)
    half_window = BASELINE_WINDOW // 2

    for i in range(len(smoothed)):
        start = max(0, i - half_window)
        end = min(
            len(smoothed),
            i + half_window + 1,
        )
        baseline[i] = np.median(smoothed[start:end])

    residual = smoothed - baseline

    min_distance = max(
        1,
        int(fps * MIN_BLINK_DISTANCE_SECONDS),
    )

    peaks, _ = find_peaks(
        residual,
        prominence=PROMINENCE,
        distance=min_distance,
    )

    return peaks


def classify_led_state(
    left_signal: list[float],
    right_signal: list[float],
    fps: float,
) -> tuple[str, int, int]:

    left_peaks = detect_blinks(
        np.asarray(left_signal),
        fps,
    )

    right_peaks = detect_blinks(
        np.asarray(right_signal),
        fps,
    )

    left_blinking = len(left_peaks) >= MIN_BLINKS
    right_blinking = len(right_peaks) >= MIN_BLINKS

    if left_blinking and right_blinking:
        state = "LED_DOUBLE"
    elif left_blinking or right_blinking:
        state = "LED_SINGLE"
    else:
        state = "LED_OFF"

    return state, len(left_peaks), len(right_peaks)


def process_video(
    model: YOLO,
    session: dict[str, str],
) -> dict[str, object]:

    session_id = session["session_id"]

    # sessions.csv uses data/raw/microphone_sessions paths,
    # but the real MP4s are stored in the external Dataset/videos folder.
    video_name = Path(session["video_path"]).name
    video_path = VIDEO_ROOT / video_name

    expected = session["expected_led_state"]

    if not video_path.exists():
        print(f"[WARN] Missing video: {video_path}")

        return {
            "session_id": session_id,
            "expected": expected,
            "predicted": "MISSING_VIDEO",
            "left_blinks": "",
            "right_blinks": "",
            "correct": False,
        }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARN] Could not open: {video_path}")

        return {
            "session_id": session_id,
            "expected": expected,
            "predicted": "OPEN_ERROR",
            "left_blinks": "",
            "right_blinks": "",
            "correct": False,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)

    left_signal: list[float] = []
    right_signal: list[float] = []

    frame_index = 0
    receiver_observations = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        result = model.predict(
            source=frame,
            conf=0.25,
            device=0,
            verbose=False,
        )[0]

        boxes = result.boxes

        best_receiver = None

        if boxes is not None:

            for box in boxes:

                if int(box.cls.item()) != RECEIVER_CLASS_ID:
                    continue

                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                candidate = (
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                if best_receiver is None or confidence > best_receiver[0]:
                    best_receiver = candidate

        if best_receiver is not None:

            _, x1, y1, x2, y2 = best_receiver

            receiver = frame[
                max(0, y1) : min(frame.shape[0], y2),
                max(0, x1) : min(frame.shape[1], x2),
            ]

            if receiver.size:

                receiver = cv2.resize(
                    receiver,
                    (208, 145),
                )

                left_signal.append(
                    led_score(
                        receiver,
                        LEFT_LED,
                    )
                )

                right_signal.append(
                    led_score(
                        receiver,
                        RIGHT_LED,
                    )
                )

                receiver_observations += 1

        frame_index += 1

    cap.release()

    predicted, left_blinks, right_blinks = classify_led_state(
        left_signal,
        right_signal,
        fps,
    )

    correct = predicted == expected

    print()
    print("=" * 60)
    print(session_id)
    print("=" * 60)
    print(f"Expected state       : {expected}")
    print(f"Predicted state      : {predicted}")
    print(f"Left blink events    : {left_blinks}")
    print(f"Right blink events   : {right_blinks}")
    print(f"Receiver observations: {receiver_observations}")
    print(f"Correct              : {'YES' if correct else 'NO'}")

    return {
        "session_id": session_id,
        "expected": expected,
        "predicted": predicted,
        "left_blinks": left_blinks,
        "right_blinks": right_blinks,
        "receiver_observations": receiver_observations,
        "correct": correct,
    }


def main() -> None:

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    if not ANNOTATIONS.exists():
        raise FileNotFoundError(f"LED annotation file not found: {ANNOTATIONS}")

    model = YOLO(str(MODEL))

    with ANNOTATIONS.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        sessions = list(csv.DictReader(file))

    if not sessions:
        raise ValueError(f"No LED sessions found in {ANNOTATIONS}")

    results = []

    for session in sessions:
        results.append(
            process_video(
                model,
                session,
            )
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_file = OUTPUT_ROOT / "evaluation.csv"

    with evaluation_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "session_id",
                "expected",
                "predicted",
                "left_blinks",
                "right_blinks",
                "receiver_observations",
                "correct",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    valid = [
        result for result in results if result["predicted"] not in {"MISSING_VIDEO", "OPEN_ERROR"}
    ]

    correct_count = sum(bool(result["correct"]) for result in valid)

    accuracy = correct_count / len(valid) if valid else 0.0

    print()
    print("=" * 60)
    print("LED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Sessions evaluated : {len(valid)}")
    print(f"Correct            : {correct_count}")
    print(f"Accuracy           : {accuracy:.3f}")
    print(f"Evaluation CSV     : {evaluation_file}")


if __name__ == "__main__":
    main()
