"""Sport registry tests — no torch required."""

from uplevels_cv.skeleton import KEYPOINT_NAMES
from uplevels_cv.sports import SPORTS, Sport, sport_spec


def test_all_six_sports_present():
    assert set(SPORTS) == {"golf", "football", "basketball", "baseball", "hockey", "tennis"}


def test_each_sport_is_fully_specified():
    for spec in SPORTS.values():
        assert spec.detect_classes, spec.key
        assert spec.events, spec.key
        assert spec.angles, spec.key
        assert spec.score_unit, spec.key
        assert spec.num_detect_classes == len(spec.detect_classes)


def test_angles_reference_valid_coco_keypoints():
    valid = set(KEYPOINT_NAMES)
    for spec in SPORTS.values():
        for angle, triple in spec.angles.items():
            assert len(triple) == 3, (spec.key, angle)
            assert all(name in valid for name in triple), (spec.key, angle, triple)


def test_sport_spec_accepts_enum_and_string():
    assert sport_spec("golf") is sport_spec(Sport.GOLF)
