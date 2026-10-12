"""Tests for labelling, feature engineering, metrics and the predictor."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from nfl_analytics.ml.evaluate import LookupBaseline, classification_metrics, expected_calibration_error
from nfl_analytics.ml.features import (
    FEATURE_SETS,
    build_features,
    feature_matrix,
    label_plays,
    parse_personnel,
    qb_alignment,
    _OFFENSE_GROUPS,
    _DEFENSE_GROUPS,
)
from nfl_analytics.ml.predict import (
    STATUS_INSUFFICIENT,
    STATUS_NO_MODEL,
    STATUS_OK,
    RunPassPredictor,
    alignment_to_features,
)


def raw_play(**overrides) -> dict:
    play = {
        "game_id": "2024_01_AAA_BBB", "play_id": 1, "season": 2024, "season_type": "REG",
        "week": 1, "posteam": "AAA", "defteam": "BBB", "play_type": "pass", "pass": 1.0,
        "rush": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0, "two_point_attempt": 0.0, "down": 1.0,
        "ydstogo": 10.0, "yardline_100": 75.0, "goal_to_go": 0.0, "qtr": 1.0,
        "game_seconds_remaining": 3600.0, "half_seconds_remaining": 1800.0,
        "score_differential": 0.0, "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 3.0, "shotgun": 1.0, "no_huddle": 0.0,
        "aborted_play": 0.0, "offense_formation": "SHOTGUN",
        "offense_personnel": "1 RB, 1 TE, 3 WR", "defenders_in_box": 6.0,
        "defense_personnel": "4 DL, 2 LB, 5 DB",
    }
    play.update(overrides)
    return play


def test_qb_alignment_unifies_both_taxonomies():
    assert qb_alignment("I_FORM", 0.0) == "UNDER_CENTER"
    assert qb_alignment("UNDER CENTER", 0.0) == "UNDER_CENTER"
    assert qb_alignment("EMPTY", 1.0) == "SHOTGUN"
    assert qb_alignment("PISTOL", 1.0) == "PISTOL"  # pbp shotgun flag includes pistol
    assert qb_alignment(None, 1.0) == "SHOTGUN"
    assert qb_alignment(None, np.nan) is None


def test_personnel_parsing_old_and_new_formats():
    assert parse_personnel("1 RB, 2 TE, 2 WR", _OFFENSE_GROUPS)["off_te"] == 2
    new = parse_personnel("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR", _OFFENSE_GROUPS)
    assert (new["off_ol"], new["off_rb"], new["off_wr"]) == (5, 1, 3)
    defense = parse_personnel("2 CB, 2 DE, 1 FS, 2 MLB, 1 OLB, 1 SS, 2 DT", _DEFENSE_GROUPS)
    assert (defense["def_dl"], defense["def_lb"], defense["def_db"]) == (4, 3, 4)
    assert np.isnan(parse_personnel(None, _OFFENSE_GROUPS)["off_rb"])


def test_labels_scrambles_as_pass_and_drops_non_scrimmage_plays():
    frame = pd.DataFrame(
        [
            raw_play(play_id=1),
            raw_play(play_id=2, play_type="run", **{"pass": 1.0, "rush": 0.0}),  # scramble
            raw_play(play_id=3, play_type="run", **{"pass": 0.0, "rush": 1.0}),
            raw_play(play_id=4, play_type="qb_kneel", qb_kneel=1.0, **{"pass": 0.0}),
            raw_play(play_id=5, play_type="no_play"),
            raw_play(play_id=6, two_point_attempt=1.0),
        ]
    )
    labelled = label_plays(frame)
    assert labelled["play_id"].tolist() == [1, 2, 3]
    assert labelled["is_pass"].tolist() == [1, 1, 0]


def test_feature_matrix_drops_rather_than_imputes_missing_inputs():
    frame = pd.DataFrame([raw_play(play_id=1), raw_play(play_id=2, defenders_in_box=np.nan)])
    plays = build_features(frame)
    X, y = feature_matrix(plays, "vision")
    assert len(X) == 1 and list(X.columns) == FEATURE_SETS["vision"]
    assert X.iloc[0]["qb_shotgun"] == 1.0
    assert len(feature_matrix(plays, "situation")[0]) == 2


def test_no_outcome_columns_in_any_feature_set():
    leaky = {"yards_gained", "epa", "air_yards", "number_of_pass_rushers", "was_pressure",
             "defense_coverage_type", "is_play_action", "pass", "rush", "play_type"}
    for features in FEATURE_SETS.values():
        assert not leaky & set(features)


def test_metrics_and_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = classification_metrics(y, p)
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["roc_auc"] == 1.0
    assert expected_calibration_error(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0])) == 0.0


def test_lookup_baseline_uses_training_rates():
    X = pd.DataFrame({"down": [1, 1, 3, 3], "ydstogo": [10, 10, 8, 8]})
    baseline = LookupBaseline("dd", ["down", "ydstogo"], bucket_distance=True, min_count=1)
    baseline.fit(X, pd.Series([0, 1, 1, 1]))
    assert baseline.predict_proba(X)[:, 1].tolist() == [0.5, 0.5, 1.0, 1.0]


@pytest.fixture
def tiny_model_dir(tmp_path):
    rng = np.random.default_rng(0)
    features = FEATURE_SETS["vision"]
    X = pd.DataFrame(rng.integers(0, 2, size=(400, len(features))).astype(float), columns=features)
    X["defenders_in_box"] = rng.integers(5, 9, size=400)
    y = ((X["qb_shotgun"] == 1) | (X["defenders_in_box"] < 6)).astype(int)
    model = LogisticRegression().fit(X, y)
    joblib.dump(
        {
            "model": model, "model_name": "logistic_regression", "feature_set": "vision",
            "features": features, "trained_on_seasons": [2016], "test_seasons": [2024],
            "test_metrics": {"accuracy": 0.5}, "reference_values": X.median().to_dict(),
        },
        tmp_path / "run_pass_vision.joblib",
    )
    return tmp_path


def test_predictor_reports_missing_model(tmp_path):
    result = RunPassPredictor.load(tmp_path).predict({"defenders_in_box": 6})
    assert result.status == STATUS_NO_MODEL and result.p_pass is None


def test_predictor_refuses_incomplete_inputs(tiny_model_dir):
    result = RunPassPredictor.load(tiny_model_dir).predict({"defenders_in_box": 7})
    assert result.status == STATUS_INSUFFICIENT
    assert result.p_pass is None
    assert set(result.missing) == {"qb_shotgun", "qb_pistol", "qb_under_center"}


def test_predictor_returns_probability_and_explanation(tiny_model_dir):
    inputs = {**alignment_to_features("SHOTGUN"), "defenders_in_box": 6}
    result = RunPassPredictor.load(tiny_model_dir).predict(inputs)
    assert result.status == STATUS_OK and result.feature_set == "vision"
    assert 0.0 < result.p_pass < 1.0 and abs(result.p_run + result.p_pass - 1) < 1e-9
    assert {row["feature"] for row in result.contributions} == {"qb_alignment", "defenders_in_box"}
    assert alignment_to_features("UNKNOWN") == {}
