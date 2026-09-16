"""Prediction-target registry tests — no ML deps required (imports are lazy)."""

from uplevels_cv.team_performance import TARGETS, Scope, TargetType, targets_for


def test_targets_cover_all_requested_use_cases():
    assert {"win_probability", "points_for", "points_against", "off_efficiency",
            "def_efficiency", "player_index", "scholarship_roi"} <= set(TARGETS)


def test_win_probability_is_classification_others_regression():
    assert TARGETS["win_probability"].target_type is TargetType.CLASSIFICATION
    assert TARGETS["points_for"].target_type is TargetType.REGRESSION
    assert TARGETS["scholarship_roi"].target_type is TargetType.REGRESSION


def test_scope_filter_partitions_targets():
    team = {t.key for t in targets_for(Scope.TEAM_GAME)}
    athlete = {t.key for t in targets_for(Scope.ATHLETE)}
    assert "win_probability" in team
    assert "player_index" in athlete
    assert team.isdisjoint(athlete)


def test_keys_match_registry():
    assert all(key == target.key for key, target in TARGETS.items())
