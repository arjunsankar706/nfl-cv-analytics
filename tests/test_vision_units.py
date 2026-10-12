"""Unit tests for the refactored broadcast-vision modules (no YOLO required)."""

from __future__ import annotations

import cv2
import numpy as np

from nfl_analytics.vision.camera import CameraMotionEstimator
from nfl_analytics.vision.classification import DetectionFeatures, TrackState, update_track
from nfl_analytics.vision.config import PipelineConfig
from nfl_analytics.vision.device import choose_device
from nfl_analytics.vision.field import build_field_context


def turf_frame(width: int = 640, height: int = 360, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 140, 50)  # BGR turf green
    noise = rng.integers(-25, 25, size=(height, width, 1))
    frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
    for x in range(30, width, 70):  # white yard lines
        cv2.line(frame, (x, 0), (x + 20, height), (240, 240, 240), 3)
    for _ in range(300):  # texture for optical flow
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        cv2.circle(frame, (x, y), 2, (90, 200, 110), -1)
    return frame


def test_field_context_finds_turf():
    frame = turf_frame()
    field = build_field_context(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    assert field.has_field
    assert field.component_ratio > 0.8
    assert field.boundary.shape == (frame.shape[1],)


def test_field_context_rejects_crowd_shot():
    rng = np.random.default_rng(1)
    # Crowd/stands: saturated reds, oranges, blues and purples, but no turf hues.
    hue = rng.choice(np.r_[0:20, 105:180], size=(360, 640)).astype(np.uint8)
    saturation = rng.integers(60, 255, size=(360, 640), dtype=np.uint8)
    value = rng.integers(40, 255, size=(360, 640), dtype=np.uint8)
    hsv_crowd = np.dstack([hue, saturation, value])
    field = build_field_context(hsv_crowd)
    assert not field.has_field


def test_camera_motion_recovers_pan():
    base = turf_frame(800, 400, seed=3)
    estimator = CameraMotionEstimator()
    shift = 0
    motion = None
    for _ in range(6):
        crop = base[:, 40 + shift : 40 + shift + 640]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        field = build_field_context(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
        motion = estimator.update(gray, field, [])
        shift += 4  # camera pans right 4 px per frame
    assert motion is not None and motion.available
    # A point at x=100 in the current image was at x=100+20 in the first frame.
    point = cv2.perspectiveTransform(
        np.array([[[100.0, 200.0]]], dtype=np.float32), motion.current_to_reference
    )[0, 0]
    assert abs(point[0] - 120.0) < 1.5
    assert abs(point[1] - 200.0) < 1.5


def strong_team_features(team: str) -> DetectionFeatures:
    scores = {"RAIDERS": 0.1, "BRONCOS": 0.1, "REF": 0.0}
    scores[team] = 2.0
    return DetectionFeatures(
        on_field=True,
        player_evidence=0.9,
        field_support=0.5,
        team_scores=scores,
        vote_weight=0.8,
        referee_texture=0.05,
    )


def test_track_state_machine_requires_repeated_evidence():
    config = PipelineConfig()
    state = TrackState()
    labels = [
        update_track(state, strong_team_features("BRONCOS"), frame, config)[0]
        for frame in range(1, 12)
    ]
    assert labels[0] == "UNKNOWN"  # one frame is never enough
    assert labels[-1] == "BRONCOS"
    assert state.classification_phase == "ESTABLISHED"


def test_established_label_resists_brief_contradiction():
    config = PipelineConfig()
    state = TrackState()
    for frame in range(1, 15):
        update_track(state, strong_team_features("RAIDERS"), frame, config)
    for frame in range(15, 20):
        label, _ = update_track(state, strong_team_features("BRONCOS"), frame, config)
    assert label == "RAIDERS"


def test_forced_nonplayer_is_labelled():
    config = PipelineConfig()
    state = TrackState()
    features = DetectionFeatures(False, 0.0, 0.0, {}, force_nonplayer=True)
    assert update_track(state, features, 1, config)[0] == "NON-PLAYER"


def test_choose_device_returns_known_backend():
    assert choose_device("auto") in ("0", "mps", "cpu")
    assert choose_device("cpu") == "cpu"
