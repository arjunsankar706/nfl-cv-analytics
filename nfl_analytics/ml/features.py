"""Pre-snap feature engineering and RUN/PASS labelling.

Every feature here is knowable before the ball is snapped: game situation,
offensive alignment, personnel groupings and the defensive box count. Outcome
columns (yards gained, EPA, air yards, pass rushers, coverage, play action,
etc.) are never read by `data.py`, so they cannot leak into a model.

Feature sets
------------
``vision``           QB alignment + defenders in box: what the broadcast
                     computer-vision pipeline can estimate from one frame.
``vision_situation`` ``vision`` plus down/distance/field position/clock/score,
                     for when the viewer supplies the scoreboard context.
``qb_situation``     QB alignment plus game situation, for clips where the
                     box count cannot be measured reliably.
``situation``        Game situation only (classic down-and-distance model).
``full``             Everything above plus offensive/defensive personnel
                     counts. An upper reference using charted data the video
                     pipeline cannot reliably observe (e.g. TE vs WR).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LABEL = "is_pass"
QB_ALIGNMENTS = ("SHOTGUN", "PISTOL", "UNDER_CENTER")

SITUATION_FEATURES = [
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "no_huddle",
]
VISION_FEATURES = [
    "qb_shotgun",
    "qb_pistol",
    "qb_under_center",
    "defenders_in_box",
]
PERSONNEL_FEATURES = ["off_rb", "off_te", "off_wr", "off_ol", "def_dl", "def_lb", "def_db"]

FEATURE_SETS: dict[str, list[str]] = {
    "vision": VISION_FEATURES,
    "vision_situation": VISION_FEATURES + SITUATION_FEATURES,
    "qb_situation": ["qb_shotgun", "qb_pistol", "qb_under_center"] + SITUATION_FEATURES,
    "situation": SITUATION_FEATURES,
    "full": VISION_FEATURES + SITUATION_FEATURES + PERSONNEL_FEATURES,
    # Fallback when video gives a confident QB alignment but nothing else.
    "qb_only": ["qb_shotgun", "qb_pistol", "qb_under_center"],
}

FEATURE_DESCRIPTIONS = {
    "qb_shotgun": "QB aligned in shotgun (1/0)",
    "qb_pistol": "QB aligned in pistol (1/0)",
    "qb_under_center": "QB under center (1/0)",
    "defenders_in_box": "Defenders in the box at the snap",
    "down": "Down (1-4)",
    "ydstogo": "Yards to go for a first down",
    "yardline_100": "Yards from the opponent end zone",
    "goal_to_go": "Goal-to-go situation (1/0)",
    "qtr": "Quarter (5 = overtime)",
    "game_seconds_remaining": "Seconds remaining in the game",
    "half_seconds_remaining": "Seconds remaining in the half",
    "score_differential": "Offense score minus defense score",
    "posteam_timeouts_remaining": "Offense timeouts remaining",
    "defteam_timeouts_remaining": "Defense timeouts remaining",
    "no_huddle": "Offense in no-huddle (1/0)",
    "off_rb": "Offensive running backs/fullbacks on field",
    "off_te": "Offensive tight ends on field",
    "off_wr": "Offensive wide receivers on field",
    "off_ol": "Offensive linemen on field",
    "def_dl": "Defensive linemen on field",
    "def_lb": "Linebackers on field",
    "def_db": "Defensive backs on field",
}

_UNDER_CENTER_FORMATIONS = {"UNDER CENTER", "SINGLEBACK", "I_FORM", "JUMBO"}
_SHOTGUN_FORMATIONS = {"SHOTGUN", "EMPTY"}

_OFFENSE_GROUPS = {
    "off_rb": ("RB", "FB"),
    "off_te": ("TE",),
    "off_wr": ("WR",),
    "off_ol": ("OL", "C", "G", "T", "OG", "OT"),
}
_DEFENSE_GROUPS = {
    "def_dl": ("DL", "DE", "DT", "NT"),
    "def_lb": ("LB", "ILB", "MLB", "OLB"),
    "def_db": ("DB", "CB", "S", "FS", "SS", "SAF"),
}
_PERSONNEL_TOKEN = re.compile(r"(\d+)\s+([A-Z]+)")


def qb_alignment(formation: object, shotgun_flag: object) -> str | None:
    """Normalize the two nflverse formation taxonomies to a QB alignment.

    2016-2022 participation data uses SHOTGUN/SINGLEBACK/I_FORM/EMPTY/PISTOL/
    JUMBO/WILDCAT; 2023+ uses SHOTGUN/UNDER CENTER/PISTOL. QB depth is the
    common denominator and is directly observable on video.
    """
    formation = formation if isinstance(formation, str) else None
    if formation == "PISTOL":
        return "PISTOL"
    if formation in _SHOTGUN_FORMATIONS:
        return "SHOTGUN"
    if formation in _UNDER_CENTER_FORMATIONS:
        return "UNDER_CENTER"
    if shotgun_flag is None or (isinstance(shotgun_flag, float) and np.isnan(shotgun_flag)):
        return None
    return "SHOTGUN" if float(shotgun_flag) >= 0.5 else "UNDER_CENTER"


def parse_personnel(text: object, groups: dict[str, tuple[str, ...]]) -> dict[str, float]:
    counts = {name: np.nan for name in groups}
    if not isinstance(text, str) or not text.strip():
        return counts
    counts = {name: 0.0 for name in groups}
    for number, position in _PERSONNEL_TOKEN.findall(text.upper()):
        for name, positions in groups.items():
            if position in positions:
                counts[name] += float(number)
    return counts


def label_plays(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep scrimmage plays with a run or pass call; label dropbacks as PASS.

    Scrambles and sacks are dropbacks and therefore PASS (the defense faced a
    pass call). Kneels, spikes, two-point tries, penalties that erased the
    play (`no_play`) and aborted snaps are excluded.
    """
    plays = frame[
        frame["play_type"].isin(["run", "pass"])
        & (frame["two_point_attempt"].fillna(0) == 0)
        & (frame["qb_kneel"].fillna(0) == 0)
        & (frame["qb_spike"].fillna(0) == 0)
        & (frame["aborted_play"].fillna(0) == 0)
        & frame["down"].notna()
    ].copy()
    plays[LABEL] = (plays["pass"].fillna(0) >= 1).astype(int)
    ambiguous = (plays["pass"].fillna(0) < 1) & (plays["rush"].fillna(0) < 1)
    return plays[~ambiguous]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return labelled plays with engineered pre-snap feature columns."""
    plays = label_plays(frame)
    alignment = [
        qb_alignment(formation, shotgun)
        for formation, shotgun in zip(plays["offense_formation"], plays["shotgun"])
    ]
    plays["qb_alignment"] = alignment
    for name in QB_ALIGNMENTS:
        plays[f"qb_{name.lower()}"] = [
            np.nan if value is None else float(value == name) for value in alignment
        ]
    box = pd.to_numeric(plays["defenders_in_box"], errors="coerce")
    # 0/1 defenders in the box are missing-value codes in some seasons.
    plays["defenders_in_box"] = box.where(box >= 2)

    offense = pd.DataFrame(
        [parse_personnel(text, _OFFENSE_GROUPS) for text in plays["offense_personnel"]],
        index=plays.index,
    )
    defense = pd.DataFrame(
        [parse_personnel(text, _DEFENSE_GROUPS) for text in plays["defense_personnel"]],
        index=plays.index,
    )
    # Pre-2023 offensive strings list only skill positions; five linemen are
    # implied unless an extra "OL" token appears.
    implied_line = offense["off_ol"].eq(0) & offense["off_rb"].notna()
    offense.loc[implied_line, "off_ol"] = 5.0
    plays = pd.concat([plays, offense, defense], axis=1)
    for column in ("goal_to_go", "no_huddle"):
        plays[column] = plays[column].fillna(0).astype(float)
    return plays


def feature_matrix(plays: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.Series]:
    """Rows with every feature of the set present; missing rows are dropped, not imputed."""
    columns = FEATURE_SETS[feature_set]
    complete = plays.dropna(subset=columns)
    return complete[columns].astype(float), complete[LABEL].astype(int)
