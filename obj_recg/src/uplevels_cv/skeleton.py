"""The pose skeleton — the shared human keypoint layout and angle math.

Uses the standard COCO 17-keypoint person layout, which every 2D-pose model here
shares (YOLO11-pose and Keypoint R-CNN both emit exactly these 17 points).
``SKELETON`` is the list of keypoint pairs to connect; drawing a line for each
pair is the limb overlay. Per-sport biomechanics angles are defined in
``sports.py`` (each a triple of these keypoint names) and measured by
``joint_angle`` below.

The COCO-17 skeleton is a mirror of fixtures/golf_swing_skeleton.json.
"""

from __future__ import annotations

# COCO person keypoints, in index order (0..16).
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Limb edges to draw (pairs of keypoint indices). Arms, legs, and torso frame.
SKELETON: list[tuple[int, int]] = [
    (KP["left_shoulder"], KP["left_elbow"]),   (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]), (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_hip"], KP["left_knee"]),         (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]),       (KP["right_knee"], KP["right_ankle"]),
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_shoulder"], KP["left_hip"]),
    (KP["right_shoulder"], KP["right_hip"]),
]

# COCO OKS per-keypoint sigmas (constants from the COCO keypoint eval spec),
# used by the pose accuracy metric in evaluate.py.
OKS_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
]

NUM_KEYPOINTS = len(KEYPOINT_NAMES)


def joint_angle(keypoints, vertex: str, a: str, b: str) -> float:
    """Angle in degrees at `vertex` between vectors to `a` and `b`.

    `keypoints` is an (17, 2|3) array of (x, y[, conf]). Returns NaN if any of
    the three points is missing/low-confidence (caller decides the threshold).
    """
    import numpy as np

    pts = np.asarray(keypoints, dtype=float)
    v, pa, pb = pts[KP[vertex], :2], pts[KP[a], :2], pts[KP[b], :2]
    u, w = pa - v, pb - v
    denom = float(np.linalg.norm(u) * np.linalg.norm(w))
    if denom == 0:
        return float("nan")
    cos = float(np.clip(np.dot(u, w) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
