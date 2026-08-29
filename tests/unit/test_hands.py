import numpy as np

from bas_assistant.features.hands import palm_center


def test_palm_center():
    keypoints = [
        {"x": float(i), "y": float(i * 2)}
        for i in range(21)
    ]

    result = palm_center(keypoints)

    expected = np.mean(
        [
            [0, 0],
            [5, 10],
            [9, 18],
            [13, 26],
            [17, 34],
        ],
        axis=0,
    )

    np.testing.assert_allclose(result, expected)
    assert result.shape == (2,)

def test_palm_center_extraction_from_both_hands():
    from bas_assistant.models import Keypoint, PoseResult
    from bas_assistant.pose.normalization import normalize_pose
    from bas_assistant.features.extractor import extract_frame_features

    body = [Keypoint(x=100, y=100, confidence=1.0) for _ in range(33)]

    # Give the landmarks used for normalization realistic positions.
    body[11] = Keypoint(x=80, y=100, confidence=1.0)   # left shoulder
    body[12] = Keypoint(x=120, y=100, confidence=1.0)  # right shoulder
    body[23] = Keypoint(x=90, y=200, confidence=1.0)   # left hip
    body[24] = Keypoint(x=110, y=200, confidence=1.0)  # right hip

    hand_keypoints = [
        {"x": float(i), "y": float(i)}
        for i in range(21)
    ]

    pose = PoseResult(
        timestamp=0.0,
        person_id=1,
        keypoints=body,
        confidence=1.0,
        metadata={
            "hands": [
                {
                    "handedness": "Left",
                    "confidence": 1.0,
                    "keypoints": hand_keypoints,
                },
                {
                    "handedness": "Right",
                    "confidence": 1.0,
                    "keypoints": [
                        {"x": float(i + 100), "y": float(i + 100)}
                        for i in range(21)
                    ],
                },
            ]
        },
    )

    normalized = normalize_pose(pose)
    assert normalized is not None

    features = extract_frame_features(normalized)

    assert features.hands is not None
    assert features.hands.shape == (10,)
    assert features.hands[4] == 1.0  # left present
    assert features.hands[5] == 1.0  # right present