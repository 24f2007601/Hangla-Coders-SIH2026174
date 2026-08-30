"""Generate a learnable synthetic feature dataset for the toy protocol."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bas_assistant.features.extractor import PoseFeatureExtractor
from bas_assistant.models import Keypoint, PoseResult

NUM_FRAMES = 30
SAMPLES_PER_CLASS = 300


STEPS = {
    "S0": "start",
    "S1": "open_tray",
    "S2": "pick_sample",
    "S3": "place_under_scope",
    "S4": "adjust_focus",
    "S5": "record_reading",
    "S6": "close_tray",
    "S7": "complete",
}


def make_pose(
    step: str,
    frame: int,
    rng: np.random.Generator,
) -> PoseResult:
    """Create a synthetic pose with strongly separated trajectories."""

    t = frame / (NUM_FRAMES - 1)

    # Smooth progress through the 30-frame sequence.
    progress = 0.5 - 0.5 * np.cos(np.pi * t)

    # Base body.
    body = np.array(
        [
            [160, 60],
            [158, 58],
            [156, 58],
            [154, 58],
            [162, 58],
            [164, 58],
            [166, 58],
            [152, 60],
            [168, 60],
            [157, 64],
            [163, 64],
            [145, 85],
            [175, 85],
            [130, 115],
            [190, 115],
            [120, 145],
            [200, 145],
            [120, 148],
            [200, 148],
            [120, 150],
            [200, 150],
            [120, 152],
            [200, 152],
            [150, 130],
            [170, 130],
            [145, 180],
            [175, 180],
            [140, 225],
            [180, 225],
            [142, 235],
            [178, 235],
            [140, 240],
            [180, 240],
        ],
        dtype=float,
    )

    left = body[15].copy()
    right = body[16].copy()

    # ---------------------------------------------------------
    # Each class gets a deliberately different trajectory.
    # ---------------------------------------------------------

    if step == "start":
        # Hands remain low and separated.
        left[0] -= 10
        right[0] += 10

    elif step == "open_tray":
        # BOTH hands move strongly outward.
        left[0] -= 100 * progress
        right[0] += 100 * progress

    elif step == "pick_sample":
        # RIGHT hand moves strongly DOWN + toward centre.
        right[0] -= 110 * progress
        right[1] += 100 * progress

    elif step == "place_under_scope":
        # RIGHT hand moves strongly UP + toward centre.
        right[0] -= 110 * progress
        right[1] -= 110 * progress

    elif step == "adjust_focus":
        # RIGHT hand remains near centre with rapid oscillation.
        right[0] -= 70
        right[1] -= 45

        # Multiple oscillations create a distinct temporal signature.
        right[0] += 20 * np.sin(6 * np.pi * t)
        right[1] += 35 * np.sin(8 * np.pi * t)

    elif step == "record_reading":
        # RIGHT hand moves toward upper-left chest.
        right[0] -= 130 * progress
        right[1] -= 70 * progress

    elif step == "close_tray":
        # BOTH hands move strongly inward.
        left[0] += 100 * progress
        right[0] -= 100 * progress

    elif step == "complete":
        # BOTH hands move strongly upward.
        left[1] -= 120 * progress
        right[1] -= 120 * progress

    # ---------------------------------------------------------
    # Add independent noise.
    # ---------------------------------------------------------

    left += rng.normal(0, 2.0, size=2)
    right += rng.normal(0, 2.0, size=2)

    body[15] = left
    body[16] = right

    # ---------------------------------------------------------
    # Body keypoints.
    # ---------------------------------------------------------

    keypoints = [
        Keypoint(
            x=float(x),
            y=float(y),
            confidence=1.0,
        )
        for x, y in body
    ]

    # ---------------------------------------------------------
    # Synthetic hand landmarks.
    # ---------------------------------------------------------

    hand_shape = [
        (0, 0),
        (-3, 2),
        (-5, 5),
        (-6, 8),
        (-7, 11),
        (-2, 3),
        (-4, 7),
        (-5, 11),
        (-6, 15),
        (1, 3),
        (0, 8),
        (0, 13),
        (0, 17),
        (4, 3),
        (5, 8),
        (6, 13),
        (7, 17),
        (7, 2),
        (9, 6),
        (10, 10),
        (11, 14),
    ]

    hands = [
        {
            "handedness": "Left",
            "confidence": 0.95,
            "keypoints": [
                {
                    "x": float(left[0] + dx),
                    "y": float(left[1] + dy),
                }
                for dx, dy in hand_shape
            ],
        },
        {
            "handedness": "Right",
            "confidence": 0.95,
            "keypoints": [
                {
                    "x": float(right[0] - dx),
                    "y": float(right[1] + dy),
                }
                for dx, dy in hand_shape
            ],
        },
    ]

    return PoseResult(
        timestamp=float(frame),
        person_id=1,
        keypoints=keypoints,
        confidence=1.0,
        metadata={"hands": hands},
    )


def generate_dataset(
    output: Path,
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> None:
    rng = np.random.default_rng(42)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extractor = PoseFeatureExtractor(
        sequence_length=NUM_FRAMES,
    )

    feature_names = extractor.feature_names

    rows: list[list[float | str]] = []

    for label, step in STEPS.items():
        print(f"Generating {step}...")

        for _ in range(samples_per_class):
            extractor.reset()

            for frame in range(NUM_FRAMES):
                pose = make_pose(
                    step,
                    frame,
                    rng,
                )
                extractor.push(pose)

            features = extractor.features()

            rows.append(
                [
                    *features.tolist(),
                    label,
                ]
            )

    rng.shuffle(rows)

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                *feature_names,
                "label",
            ]
        )

        writer.writerows(rows)

    print()
    print(f"Generated {len(rows)} samples.")
    print(f"Features per sample: {len(feature_names)}")
    print(f"Classes: {', '.join(STEPS)}")
    print(f"Output: {output}")


def main() -> None:
    generate_dataset(Path("data/processed/synthetic_features.csv"))


if __name__ == "__main__":
    main()
