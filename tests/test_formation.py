"""End-to-end formation analysis on synthetic broadcast frames with known truth."""

from __future__ import annotations

import numpy as np
import pytest

import synthetic_field as sf
from nfl_analytics.formation.analysis import UNKNOWN, analyze_formation
from nfl_analytics.formation.analyzer import consensus
from nfl_analytics.formation.field_registration import TIER_HOMOGRAPHY, TIER_NONE


def analyse(spots, **kwargs):
    frame, boxes = sf.render(spots, **kwargs)
    return analyze_formation(frame, boxes)


def value(report, name):
    return report.finding(name).value


def test_registration_recovers_field_coordinates():
    report = analyse(sf.shotgun_two_by_two())
    registration = report.registration
    assert registration.tier == TIER_HOMOGRAPHY
    assert registration.reprojection_error_yd < 0.1
    # Relative distances between projected points must match the field.
    a = registration.to_field(sf.project(10.0, 1.0))[0]
    b = registration.to_field(sf.project(25.0, 6.0))[0]
    assert abs(abs(b[0] - a[0]) - 15.0) < 0.3
    assert abs(abs(b[1] - a[1]) - 5.0) < 0.3


def test_all_twenty_two_players_and_offense_identified():
    report = analyse(sf.shotgun_two_by_two())
    assert report.status == "OK"
    assert len(report.offense()) == 11 and len(report.defense()) == 11
    assert report.offense_team == "A"  # white jerseys = offense in the fixture
    assert report.offense_identification.confidence >= 0.5


def test_shotgun_two_by_two():
    report = analyse(sf.shotgun_two_by_two())
    assert value(report, "qb_alignment") == "SHOTGUN"
    assert value(report, "receiver_distribution") == "2x2"
    assert value(report, "trips") == "NO"
    assert value(report, "box_count") == 6  # 4 DL + 2 LB
    assert value(report, "safety_shell").startswith("2-HIGH")
    assert report.formation_name == "Shotgun"


def test_under_center_i_formation():
    report = analyse(sf.i_formation())
    assert value(report, "qb_alignment") == "UNDER_CENTER"
    assert value(report, "backfield") == "I_FORMATION"
    assert value(report, "box_count") == 7  # 4 DL + 3 LB, box widened by the TE
    assert report.formation_name == "Under Center · I-Formation"


def test_singleback():
    spots = [s for s in sf.i_formation() if s.role != "FB"]
    report = analyse(spots)
    assert value(report, "qb_alignment") == "UNDER_CENTER"
    assert value(report, "backfield") == "SINGLEBACK"


def test_pistol():
    report = analyse(sf.pistol())
    assert value(report, "qb_alignment") == "PISTOL"
    assert report.formation_name == "Pistol"


def test_trips_and_bunch_coexist_with_shotgun():
    report = analyse(sf.bunch_right())
    assert value(report, "trips") == "TRIPS RIGHT"
    assert value(report, "bunch") == "BUNCH RIGHT"
    assert report.formation_name == "Shotgun · Trips Right · Bunch Right"


def test_empty_backfield():
    report = analyse(sf.empty_three_by_two())
    assert value(report, "backfield") == "EMPTY"
    assert value(report, "receiver_distribution") == "2x3"
    assert "Empty" in report.formation_tags and "Shotgun" in report.formation_tags


def test_left_right_follows_offense_facing_direction():
    # Mirror the field: the offense now attacks toward -d (screen left), so
    # its right-hand side is away from the camera and the bunch is on its LEFT.
    mirrored = [sf.Spot(s.team, 2 * sf.LOS - s.d, s.l, s.role) for s in sf.bunch_right()]
    report = analyse(mirrored)
    assert value(report, "trips") == "TRIPS LEFT"
    assert value(report, "bunch") == "BUNCH LEFT"


def test_offense_hint_is_respected_and_validated():
    frame, boxes = sf.render(sf.shotgun_two_by_two())
    assert analyze_formation(frame, boxes, offense_hint="A").offense_team == "A"
    bad = analyze_formation(frame, boxes, offense_hint="RAMS")
    assert bad.status != "OK" and "RAMS" in bad.message


def test_too_few_players_is_insufficient():
    report = analyse(sf.shotgun_two_by_two()[:6])
    assert report.status == "INSUFFICIENT"
    assert report.model_inputs() == {}


def test_without_markings_depth_labels_stay_unknown():
    frame, boxes = sf.render(sf.shotgun_two_by_two(), draw_markings=False)
    report = analyze_formation(frame, boxes, offense_hint="A")
    assert report.registration.tier == TIER_NONE
    assert value(report, "box_count") == UNKNOWN
    assert "defenders_in_box" not in report.model_inputs()


def test_indistinguishable_jerseys_are_not_guessed():
    spots = [sf.Spot("O", s.d, s.l, s.role) for s in sf.shotgun_two_by_two()]  # everyone in white
    report = analyse(spots)
    assert report.status == "INSUFFICIENT"
    assert "colour" in report.message


def test_model_inputs_only_use_confident_findings():
    report = analyse(sf.pistol())
    inputs = report.model_inputs()
    assert inputs["qb_pistol"] == 1.0 and inputs["qb_shotgun"] == 0.0
    assert inputs["defenders_in_box"] == 6.0
    report.findings["box_count"].confidence = 0.2
    assert "defenders_in_box" not in report.model_inputs()


def test_consensus_prefers_agreement_and_ignores_failed_frames():
    good = analyse(sf.shotgun_two_by_two())
    failed = analyse(sf.shotgun_two_by_two()[:5])
    result = consensus([good, failed, analyse(sf.shotgun_two_by_two(), seed=1)])
    assert result.frames_ok == 2
    assert result.findings["qb_alignment"].value == "SHOTGUN"
    assert result.findings["box_count"].value == 6
    assert result.model_inputs()["qb_shotgun"] == 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_robust_to_turf_noise(seed):
    report = analyse(sf.pistol(), seed=seed)
    assert value(report, "qb_alignment") == "PISTOL"
    assert np.isfinite(report.registration.reprojection_error_yd)
