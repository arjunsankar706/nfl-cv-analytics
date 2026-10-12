"""Tunable thresholds for the broadcast tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_VIDEO = "videos/RaidersBroncos.mp4"
DEFAULT_OUTPUT = "nfl_movement_analytics_v6.mp4"
TEAM_NAMES = ("RAIDERS", "BRONCOS", "REF")


@dataclass(frozen=True)
class PipelineConfig:
    confidence: float = 0.30
    image_size: int = 640
    max_detections: int = 80
    min_box_height: int = 32
    history_decay: float = 0.96
    stale_track_frames: int = 180
    minimum_team_observations: int = 3
    minimum_vote_weight: float = 0.18
    team_candidate_frames: int = 6
    referee_candidate_frames: int = 12
    team_switch_frames: int = 14
    team_to_referee_frames: int = 22
    referee_to_team_frames: int = 14
    established_context_grace_frames: int = 30
    movement_history_frames: int = 90
    movement_max_gap_frames: int = 12
    movement_speed_window_seconds: float = 0.75
