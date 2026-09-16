"""MediaPipe BlazePose 33-landmark layout (3D) + projection to COCO-17.

BlazePose predicts 33 landmarks with (x, y, z) — more detail than COCO's 17
(it adds face, hands, and feet points, and depth). ``COCO_FROM_BLAZEPOSE`` maps
the 33 down to the 17 COCO keypoints so BlazePose can be scored on the same 2D
metrics as the other pose models; its 3D output (z) is what unlocks swing-plane
and rotation analysis the 2D models can't provide.
"""

from __future__ import annotations

BLAZEPOSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
BP = {name: i for i, name in enumerate(BLAZEPOSE_NAMES)}
NUM_LANDMARKS = len(BLAZEPOSE_NAMES)

# Body limb edges to draw (arms, legs, torso, feet). Face/hand detail omitted.
BLAZEPOSE_SKELETON: list[tuple[int, int]] = [
    (BP["left_shoulder"], BP["right_shoulder"]),
    (BP["left_shoulder"], BP["left_elbow"]), (BP["left_elbow"], BP["left_wrist"]),
    (BP["right_shoulder"], BP["right_elbow"]), (BP["right_elbow"], BP["right_wrist"]),
    (BP["left_shoulder"], BP["left_hip"]), (BP["right_shoulder"], BP["right_hip"]),
    (BP["left_hip"], BP["right_hip"]),
    (BP["left_hip"], BP["left_knee"]), (BP["left_knee"], BP["left_ankle"]),
    (BP["right_hip"], BP["right_knee"]), (BP["right_knee"], BP["right_ankle"]),
    (BP["left_ankle"], BP["left_heel"]), (BP["left_heel"], BP["left_foot_index"]),
    (BP["right_ankle"], BP["right_heel"]), (BP["right_heel"], BP["right_foot_index"]),
]

# BlazePose index for each COCO-17 keypoint, in COCO order. Use to down-project
# BlazePose landmarks to COCO-17 for apples-to-apples 2D OKS scoring.
COCO_FROM_BLAZEPOSE = [
    BP["nose"], BP["left_eye"], BP["right_eye"], BP["left_ear"], BP["right_ear"],
    BP["left_shoulder"], BP["right_shoulder"], BP["left_elbow"], BP["right_elbow"],
    BP["left_wrist"], BP["right_wrist"], BP["left_hip"], BP["right_hip"],
    BP["left_knee"], BP["right_knee"], BP["left_ankle"], BP["right_ankle"],
]


def to_coco17(landmarks):
    """Down-project (33, D) BlazePose landmarks to (17, D) in COCO order."""
    import numpy as np

    return np.asarray(landmarks)[COCO_FROM_BLAZEPOSE]
