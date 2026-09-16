"""Sport registry — the per-sport domain content the pipeline is parameterized by.

Each sport defines:
  detect_classes : objects a detector locates (class 0 is the athlete)
  events         : signature actions used for event tagging and feature counts
  angles         : biomechanics joint angles, each a (vertex, a, b) triple of
                   COCO-17 keypoint names, measured by skeleton.joint_angle
  score_unit     : the unit team scoring is expressed in

Adding a sport is a one-line append to ``_SPORTS``; everything downstream
(detection class count, biomechanics features, team scoring) reads these specs.

Angle definitions assume a right-handed athlete (lead side = left). Swap left/
right for left-handed athletes. Golf and tennis are individual sports; their
"team performance" follows NCAA team scoring (aggregate strokes; dual-match points).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Sport(str, Enum):
    GOLF = "golf"
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    BASEBALL = "baseball"
    HOCKEY = "hockey"
    TENNIS = "tennis"


@dataclass(frozen=True)
class SportSpec:
    key: Sport
    detect_classes: list[str]
    events: list[str]
    angles: dict[str, tuple[str, str, str]]
    score_unit: str

    @property
    def num_detect_classes(self) -> int:
        return len(self.detect_classes)


_SPORTS = [
    SportSpec(
        Sport.GOLF,
        detect_classes=["golfer", "club", "ball"],
        events=["address", "backswing", "downswing", "impact", "follow_through"],
        angles={
            "lead_arm": ("left_elbow", "left_shoulder", "left_wrist"),
            "trail_arm": ("right_elbow", "right_shoulder", "right_wrist"),
            "lead_knee_flex": ("left_knee", "left_hip", "left_ankle"),
            "spine_tilt": ("left_hip", "left_shoulder", "right_hip"),
            "hip_hinge": ("left_hip", "left_shoulder", "left_knee"),
        },
        score_unit="strokes",
    ),
    SportSpec(
        Sport.FOOTBALL,
        detect_classes=["player", "football", "goalpost", "pylon"],
        events=["snap", "pass", "catch", "run", "tackle"],
        angles={
            "throwing_elbow": ("right_elbow", "right_shoulder", "right_wrist"),
            "throwing_shoulder": ("right_shoulder", "right_elbow", "right_hip"),
            "stride_knee": ("left_knee", "left_hip", "left_ankle"),
            "trunk_lean": ("left_hip", "left_shoulder", "right_hip"),
        },
        score_unit="points",
    ),
    SportSpec(
        Sport.BASKETBALL,
        detect_classes=["player", "basketball", "rim", "backboard"],
        events=["shot", "layup", "dribble", "rebound", "pass", "block"],
        angles={
            "shooting_elbow": ("right_elbow", "right_shoulder", "right_wrist"),
            "release_shoulder": ("right_shoulder", "right_elbow", "right_hip"),
            "knee_bend": ("right_knee", "right_hip", "right_ankle"),
            "trunk_lean": ("left_hip", "left_shoulder", "right_hip"),
        },
        score_unit="points",
    ),
    SportSpec(
        Sport.BASEBALL,
        detect_classes=["player", "baseball", "bat", "glove", "base"],
        events=["pitch", "swing", "hit", "catch", "throw", "steal"],
        angles={
            "throwing_elbow": ("right_elbow", "right_shoulder", "right_wrist"),
            "lead_arm": ("left_elbow", "left_shoulder", "left_wrist"),
            "stride_knee": ("left_knee", "left_hip", "left_ankle"),
            "hip_shoulder_sep": ("left_hip", "left_shoulder", "right_hip"),
        },
        score_unit="runs",
    ),
    SportSpec(
        Sport.HOCKEY,
        detect_classes=["skater", "puck", "stick", "goal_net"],
        events=["shot", "pass", "faceoff", "check", "save"],
        angles={
            "shot_elbow": ("right_elbow", "right_shoulder", "right_wrist"),
            "skating_knee_flex": ("left_knee", "left_hip", "left_ankle"),
            "forward_lean": ("left_hip", "left_shoulder", "left_knee"),
            "hip_hinge": ("right_hip", "right_shoulder", "right_knee"),
        },
        score_unit="goals",
    ),
    SportSpec(
        Sport.TENNIS,
        detect_classes=["player", "tennis_ball", "racket", "net"],
        events=["serve", "forehand", "backhand", "volley", "return"],
        angles={
            "serve_shoulder": ("right_shoulder", "right_elbow", "right_hip"),
            "swing_elbow": ("right_elbow", "right_shoulder", "right_wrist"),
            "trunk_rotation": ("left_hip", "left_shoulder", "right_hip"),
            "knee_load": ("right_knee", "right_hip", "right_ankle"),
        },
        score_unit="points",
    ),
]

SPORTS: dict[str, SportSpec] = {s.key.value: s for s in _SPORTS}


def sport_spec(sport: Sport | str) -> SportSpec:
    """Look up a SportSpec by Sport enum or string key."""
    return SPORTS[Sport(sport).value]
