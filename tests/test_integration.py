"""Pre-snap window detection, situation handling and the video-to-prediction path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nfl_analytics.integration.presnap import find_presnap_windows, relative_motion
from nfl_analytics.integration.video_analysis import analyze_video, situation_features
from nfl_analytics.ml.predict import STATUS_INSUFFICIENT, RunPassPredictor
from nfl_analytics.vision.pipeline import FrameAnalysis, PlayerObservation

FPS = 30.0
ROOT = Path(__file__).resolve().parents[1]


def frame_with(number: int, positions: list[tuple[float, float]], height: int = 60) -> FrameAnalysis:
    observations = tuple(
        PlayerObservation(
            track_id=index,
            box=(int(x - 10), int(y - height), int(x + 10), int(y)),
            label="UNKNOWN",
            confidence=0.5,
            on_field=True,
            player_evidence=0.8,
            field_support=0.5,
            referee_texture=0.0,
            speed_px_s=0.0,
            speed_valid=False,
        )
        for index, (x, y) in enumerate(positions)
    )
    return FrameAnalysis(number, observations, True, 1, False, True, 0.6)


def synthetic_clip(still_frames: int = 60, live_frames: int = 30, pan: float = 0.0, zoom: float = 0.0):
    rng = np.random.default_rng(0)
    base = np.column_stack([rng.uniform(200, 1000, 14), rng.uniform(250, 600, 14)])
    velocity = rng.normal(0, 1, (14, 2))
    velocity /= np.linalg.norm(velocity, axis=1, keepdims=True)
    frames = []
    for index in range(still_frames + live_frames):
        # Camera pan/zoom applied to everyone; players move only after the snap.
        scale = 1.0 + zoom * index
        offset = np.array([pan * index, 0.0])
        live = max(0, index - still_frames)
        positions = (base + velocity * 6.0 * live) * scale + offset  # 6 px/frame ~ 3 body-heights/s
        frames.append(frame_with(index + 1, [tuple(p) for p in positions]))
    return frames


def test_relative_motion_ignores_camera_pan_and_zoom():
    frames = synthetic_clip(pan=4.0, zoom=0.002)
    # Residual comes only from integer box rounding (live play is ~3 body-heights/s).
    assert relative_motion(frames[10], frames[25], 0.5) < 0.05


def test_window_found_before_snap_and_snap_located():
    frames = synthetic_clip(still_frames=60, live_frames=30, pan=2.0)
    windows, _ = find_presnap_windows(frames, FPS)
    assert len(windows) == 1
    window = windows[0]
    assert window.snap_frame is not None and 58 <= window.snap_frame <= 66
    assert max(window.analysis_frames) < window.snap_frame


def test_no_window_when_players_are_moving_or_absent():
    moving = synthetic_clip(still_frames=0, live_frames=60)
    assert find_presnap_windows(moving, FPS)[0] == []
    empty = [FrameAnalysis(i + 1, (), False, i, True, False, 0.0) for i in range(60)]
    assert find_presnap_windows(empty, FPS)[0] == []


def test_situation_features_derive_only_exact_values():
    features, missing = situation_features(
        {"down": 3, "ydstogo": 4, "yardline_100": 4, "qtr": 2, "clock_seconds": 120, "score_differential": -3}
    )
    assert features["goal_to_go"] == 1.0
    assert features["game_seconds_remaining"] == 2 * 900 + 120
    assert features["half_seconds_remaining"] == 120
    assert set(missing) == {"posteam_timeouts_remaining", "defteam_timeouts_remaining", "no_huddle"}
    assert "posteam_timeouts_remaining" not in features
    q3, _ = situation_features({"qtr": 3, "clock_seconds": 600})
    assert q3["half_seconds_remaining"] == 1500 and q3["game_seconds_remaining"] == 1500
    q1, _ = situation_features({"qtr": 1, "clock_seconds": 900})
    assert q1["half_seconds_remaining"] == 1800 and q1["game_seconds_remaining"] == 3600


@pytest.mark.slow
def test_real_clip_end_zone_presnap_window(requires_sample_video):
    predictor = RunPassPredictor.load()
    if not predictor.available:
        pytest.skip("models not trained locally")
    result = analyze_video(
        requires_sample_video,
        model_path=str(ROOT / "yolo11n.pt"),
        device="auto",
        max_frames=1320,
        predictor=predictor,
    )
    verified = result.verified_plays
    assert verified, result.messages
    play = verified[-1]
    assert 1140 <= play.window.start_frame <= 1200
    assert play.window.snap_frame is not None and 1260 <= play.window.snap_frame <= 1300
    assert play.consensus.findings["qb_alignment"].value == "SHOTGUN"
    # Shotgun is visible but below the 0.5 confidence the models require, and
    # there is no scoreboard context, so no model applies.
    assert play.prediction.status == STATUS_INSUFFICIENT and play.prediction.p_pass is None
