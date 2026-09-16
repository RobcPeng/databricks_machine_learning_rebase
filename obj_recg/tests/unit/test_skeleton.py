"""Skeleton tests — verify the limb-line definition and angle math."""

import json
import math
import os

from uplevels_cv import skeleton as sk


def test_seventeen_keypoints_and_sigmas_align():
    assert len(sk.KEYPOINT_NAMES) == 17
    assert len(sk.OKS_SIGMAS) == sk.NUM_KEYPOINTS


def test_skeleton_edges_reference_valid_keypoints():
    for a, b in sk.SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17
        assert a != b


def test_fixture_matches_code():
    fixture = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures",
                           "golf_swing_skeleton.json")
    with open(fixture) as f:
        data = json.load(f)
    assert data["keypoint_names"] == sk.KEYPOINT_NAMES
    code_edges = {(sk.KEYPOINT_NAMES[a], sk.KEYPOINT_NAMES[b]) for a, b in sk.SKELETON}
    json_edges = {tuple(e) for e in data["skeleton"]}
    assert code_edges == json_edges


def test_joint_angle_right_angle():
    # A pose where the left elbow sits at the right angle of an L.
    kps = [[0.0, 0.0]] * 17
    kps[sk.KP["left_elbow"]] = [1.0, 1.0]
    kps[sk.KP["left_shoulder"]] = [1.0, 2.0]   # straight up from elbow
    kps[sk.KP["left_wrist"]] = [2.0, 1.0]      # straight right from elbow
    angle = sk.joint_angle(kps, "left_elbow", "left_shoulder", "left_wrist")
    assert math.isclose(angle, 90.0, abs_tol=1e-6)
