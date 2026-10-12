"""End-to-end: broadcast clip -> tracking -> pre-snap windows -> formation -> RUN/PASS.

Every stage reports what it could and could not establish. A prediction is
produced only when a trained model's inputs are all available from confident
visual findings plus any game situation the user supplied; nothing is
imputed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

import cv2
import numpy as np

from nfl_analytics.formation.analysis import FormationReport
from nfl_analytics.formation.analyzer import ConsensusReport, FormationAnalyzer, consensus
from nfl_analytics.ml.predict import Prediction, RunPassPredictor
from nfl_analytics.vision.pipeline import FrameAnalysis, RunSummary, run_video

from .presnap import PresnapWindow, find_presnap_windows

SITUATION_FIELDS = (
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "clock_seconds",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "no_huddle",
)


def situation_features(situation: Mapping[str, Optional[float]]) -> tuple[dict[str, float], list[str]]:
    """Convert user-entered scoreboard context to model features.

    Derived values are exact functions of the entered ones (goal-to-go from
    distance and field position; game/half seconds from quarter and clock).
    Missing entries stay missing.
    """
    provided = {k: v for k, v in situation.items() if v is not None}
    missing = [name for name in SITUATION_FIELDS if name not in provided]
    features: dict[str, float] = {}
    for name in ("down", "ydstogo", "yardline_100", "qtr", "score_differential",
                 "posteam_timeouts_remaining", "defteam_timeouts_remaining", "no_huddle"):
        if name in provided:
            features[name] = float(provided[name])
    if "ydstogo" in provided and "yardline_100" in provided:
        features["goal_to_go"] = float(provided["ydstogo"] >= provided["yardline_100"])
    if "qtr" in provided and "clock_seconds" in provided:
        quarter, clock = int(provided["qtr"]), float(provided["clock_seconds"])
        if quarter <= 4:
            features["game_seconds_remaining"] = (4 - quarter) * 900 + clock
            # Q1/Q3 leave one more full quarter in the half; Q2/Q4 none.
            features["half_seconds_remaining"] = (900 if quarter in (1, 3) else 0) + clock
        else:  # overtime: the model saw overtime rows with these conventions
            features["game_seconds_remaining"] = clock
            features["half_seconds_remaining"] = clock
    return features, missing


@dataclass
class PlayAnalysis:
    window: PresnapWindow
    reports: list[FormationReport]
    consensus: ConsensusReport
    prediction: Prediction
    overlay_bgr: Optional[np.ndarray]
    overlay_frame: Optional[int]

    @property
    def verified(self) -> bool:
        return self.consensus.frames_ok > 0


@dataclass
class VideoAnalysis:
    tracking: RunSummary
    frames: list[FrameAnalysis]
    motion: np.ndarray
    plays: list[PlayAnalysis] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    annotated_video: Optional[str] = None

    @property
    def verified_plays(self) -> list[PlayAnalysis]:
        return [play for play in self.plays if play.verified]


def read_frame(path: str | Path, frame_number: int) -> Optional[np.ndarray]:
    video = cv2.VideoCapture(str(path))
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
        ok, frame = video.read()
        return frame if ok else None
    finally:
        video.release()


def tracked_labels(frame: FrameAnalysis) -> list[tuple[tuple[int, int, int, int], int, str]]:
    return [(o.box, o.track_id, o.label) for o in frame.observations]


def analyze_window(
    video_path: str | Path,
    window: PresnapWindow,
    frames: list[FrameAnalysis],
    analyzer: FormationAnalyzer,
    predictor: RunPassPredictor,
    situation: Mapping[str, Optional[float]] | None = None,
    offense_hint: Optional[str] = None,
    use_tracker_team_names: bool = False,
) -> PlayAnalysis:
    from nfl_analytics.formation.render import draw_formation_overlay

    by_number = {frame.frame_number: frame for frame in frames}
    reports = []
    images = {}
    for number in window.analysis_frames:
        image = read_frame(video_path, number)
        if image is None:
            continue
        tracked = tracked_labels(by_number[number]) if use_tracker_team_names and number in by_number else None
        report = analyzer.analyze(image, offense_hint=offense_hint, tracked=tracked)
        reports.append(report)
        images[id(report)] = (number, image)
    agreed = consensus(reports)
    overlay = overlay_frame = None
    if agreed.best is not None and id(agreed.best) in images:
        overlay_frame, image = images[id(agreed.best)]
        overlay = draw_formation_overlay(image, agreed.best)
    features, _ = situation_features(situation or {})
    prediction = predictor.predict({**agreed.model_inputs(), **features})
    return PlayAnalysis(window, reports, agreed, prediction, overlay, overlay_frame)


def analyze_video(
    video_path: str | Path,
    output_video: str | Path | None = None,
    model_path: str = "yolo11n.pt",
    device: str = "auto",
    max_frames: int = 0,
    situation: Mapping[str, Optional[float]] | None = None,
    offense_hint: Optional[str] = None,
    use_tracker_team_names: bool = False,
    predictor: RunPassPredictor | None = None,
    max_windows: int = 4,
    progress: Callable[[str, float], None] | None = None,
    codec: str = "mp4v",
) -> VideoAnalysis:
    progress = progress or (lambda _stage, _fraction: None)
    predictor = predictor or RunPassPredictor.load()
    started = time.perf_counter()

    def on_frame(number: int, total: int, _analysis: FrameAnalysis) -> None:
        if total > 0 and number % 10 == 0:
            progress("tracking", min(1.0, number / total))

    summary, frames, _ = run_video(
        video_path,
        output_video,
        model_path=model_path,
        device=device,
        max_frames=max_frames,
        codec=codec,
        on_frame=on_frame,
        verbose=False,
    )
    tracking_seconds = time.perf_counter() - started
    result = VideoAnalysis(summary, frames, np.array([]), annotated_video=str(output_video) if output_video else None)
    result.timings["tracking_s"] = round(tracking_seconds, 2)
    if not frames:
        result.messages.append("The video contained no decodable frames.")
        return result

    progress("pre-snap search", 0.0)
    windows, motion = find_presnap_windows(frames, summary.fps, max_windows=max_windows)
    result.motion = motion
    if not windows:
        result.messages.append(
            "No pre-snap window found: no stretch of at least 0.5 s with most players visible and set "
            "(the clip may be a replay close-up, start after the snap, or lack a stable wide shot)."
        )
        return result

    analyzer = FormationAnalyzer(model_path=model_path, device=device)
    formation_started = time.perf_counter()
    for index, window in enumerate(windows):
        progress("formation analysis", index / len(windows))
        result.plays.append(
            analyze_window(
                video_path,
                window,
                frames,
                analyzer,
                predictor,
                situation=situation,
                offense_hint=offense_hint,
                use_tracker_team_names=use_tracker_team_names,
            )
        )
    progress("done", 1.0)
    result.timings["formation_s"] = round(time.perf_counter() - formation_started, 2)
    result.timings["total_s"] = round(time.perf_counter() - started, 2)
    unverified = len(result.plays) - len(result.verified_plays)
    if unverified:
        result.messages.append(
            f"{unverified} low-motion window(s) were rejected: no two-team alignment across a line of scrimmage "
            "(e.g. players walking back to the huddle)."
        )
    return result
