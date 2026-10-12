"""Download and load historical NFL play-by-play data from nflverse.

Sources (public GitHub release assets of https://github.com/nflverse/nflverse-data):

* ``pbp/play_by_play_{season}.parquet`` - nflfastR play-by-play: game state,
  play type and outcome columns.
* ``pbp_participation/pbp_participation_{season}.parquet`` - per-play
  participation/alignment fields such as ``offense_formation``,
  ``offense_personnel`` and ``defenders_in_box``.

Files are cached under ``data/raw`` (git-ignored) and are never committed.
See the nflverse repository for licensing/attribution terms.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

RELEASE_ROOT = "https://github.com/nflverse/nflverse-data/releases/download"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SOURCES = {
    "pbp": "pbp/play_by_play_{season}.parquet",
    "participation": "pbp_participation/pbp_participation_{season}.parquet",
}

# Only the columns needed to build the label and pre-snap features are read.
PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
    "two_point_attempt",
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
    "shotgun",
    "no_huddle",
    "aborted_play",
]

PARTICIPATION_COLUMNS = [
    "nflverse_game_id",
    "play_id",
    "offense_formation",
    "offense_personnel",
    "defenders_in_box",
    "defense_personnel",
]


def season_path(kind: str, season: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / Path(SOURCES[kind].format(season=season)).name


def download(kind: str, season: int, data_dir: Path = DATA_DIR, force: bool = False) -> Path:
    destination = season_path(kind, season, data_dir)
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE_ROOT}/{SOURCES[kind].format(season=season)}"
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)
    return destination


def missing_files(seasons: Iterable[int], data_dir: Path = DATA_DIR) -> list[Path]:
    return [
        season_path(kind, season, data_dir)
        for season in seasons
        for kind in SOURCES
        if not season_path(kind, season, data_dir).exists()
    ]


def load_season(season: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Join play-by-play with participation for one season (inner join)."""
    pbp = pd.read_parquet(season_path("pbp", season, data_dir), columns=PBP_COLUMNS)
    participation = pd.read_parquet(
        season_path("participation", season, data_dir), columns=PARTICIPATION_COLUMNS
    ).rename(columns={"nflverse_game_id": "game_id"})
    participation["play_id"] = participation["play_id"].astype(pbp["play_id"].dtype)
    return pbp.merge(participation, on=["game_id", "play_id"], how="inner", validate="1:1")


def load_seasons(seasons: Iterable[int], data_dir: Path = DATA_DIR) -> pd.DataFrame:
    missing = missing_files(seasons, data_dir)
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing nflverse files: {names}. Run `python -m nfl_analytics.ml.train --download`."
        )
    return pd.concat([load_season(season, data_dir) for season in seasons], ignore_index=True)
