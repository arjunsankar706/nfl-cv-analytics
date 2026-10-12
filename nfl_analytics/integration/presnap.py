"""Locate pre-snap windows in a tracked broadcast clip.

Motion is measured from tracked foot points over a 0.5 s baseline after
removing a robust similarity transform (pan + zoom + roll) fitted to all
players, so camera work does not register as movement, and expressed in
body-heights per second so it is independent of zoom level.

Measured on the sample clip (median, body-heights/s): set end-zone
alignment 0.01, set sideline alignment 0.29 (defenders creeping and box
jitter at small scale), live play 0.4-0.75. A window is a run of frames with
most players visible and motion below 0.45; it ends at the snap burst.
Windows are candidates only: the formation analyzer must confirm a two-team
alignment across a line of scrimmage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from nfl_analytics.vision.pipeline import FrameAnalysis, PlayerObservation

STILL_MOTION = 0.45
MIN_PLAYERS = 9
MIN_WINDOW_SECONDS = 0.5
BASELINE_SECONDS = 0.5


@dataclass
class PresnapWindow:
    start_frame: int  # 1-based frame numbers, inclusive
    end_frame: int
    median_motion: float
    snap_frame: Optional[int]
    analysis_frames: list[int]

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def as_dict(self, fps: float) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_seconds": round((self.start_frame - 1) / fps, 2),
            "end_seconds": round((self.end_frame - 1) / fps, 2),
            "median_motion_body_heights_per_s": round(self.median_motion, 3),
            "snap_frame": self.snap_frame,
            "analysis_frames": self.analysis_frames,
        }


def _players(frame: FrameAnalysis) -> dict[int, PlayerObservation]:
    return {
        o.track_id: o
        for o in frame.observations
        if o.on_field and o.label != "NON-PLAYER" and o.height >= 20
    }


def relative_motion(earlier: FrameAnalysis, later: FrameAnalysis, seconds: float) -> float:
    """Median player speed (body-heights/s) after removing camera similarity motion."""
    first, second = _players(earlier), _players(later)
    ids = [track for track in second if track in first]
    if len(ids) < 6 or seconds <= 0:
        return float("nan")
    a = np.array([first[t].foot_point for t in ids], dtype=np.float32)
    b = np.array([second[t].foot_point for t in ids], dtype=np.float32)
    heights = np.array([second[t].height for t in ids], dtype=float)
    transform, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.LMEDS)
    if transform is None:
        return float("nan")
    predicted = a @ transform[:, :2].T + transform[:, 2]
    speeds = np.linalg.norm(b - predicted, axis=1) / heights / seconds
    return float(np.median(speeds))


def motion_series(frames: Sequence[FrameAnalysis], fps: float) -> np.ndarray:
    lag = max(1, int(round(BASELINE_SECONDS * fps)))
    series = np.full(len(frames), np.nan)
    for index in range(lag, len(frames)):
        earlier, later = frames[index - lag], frames[index]
        if not later.has_field or earlier.camera_segment != later.camera_segment and not later.camera_available:
            continue
        series[index] = relative_motion(earlier, later, lag / fps)
    return series


def find_presnap_windows(
    frames: Sequence[FrameAnalysis],
    fps: float,
    max_windows: int = 4,
    frames_per_window: int = 5,
) -> tuple[list[PresnapWindow], np.ndarray]:
    motion = motion_series(frames, fps)
    counts = np.array([len(_players(frame)) for frame in frames])
    candidate = np.isfinite(motion) & (motion <= STILL_MOTION) & (counts >= MIN_PLAYERS)
    minimum = int(round(MIN_WINDOW_SECONDS * fps))
    gap_tolerance = int(round(0.1 * fps))
    windows: list[PresnapWindow] = []
    index = 0
    while index < len(frames):
        if not candidate[index]:
            index += 1
            continue
        start = index
        last_good = index
        cursor = index + 1
        while cursor < len(frames) and (candidate[cursor] or cursor - last_good <= gap_tolerance):
            if candidate[cursor]:
                last_good = cursor
            cursor += 1
        end = last_good
        if end - start + 1 >= minimum:
            window_motion = float(np.nanmedian(motion[start : end + 1]))
            clip_ended = end >= len(frames) - 1 - gap_tolerance
            snap = None if clip_ended else _locate_snap(frames, start, end, fps, window_motion)
            # Analyse frames comfortably before the snap (or the clip end).
            stop = (snap if snap is not None else end) - int(round(0.15 * fps))
            stop = max(start, stop)
            picks = np.linspace(start + (stop - start) * 0.4, stop, frames_per_window)
            windows.append(
                PresnapWindow(
                    start_frame=frames[start].frame_number,
                    end_frame=frames[stop].frame_number,
                    median_motion=window_motion,
                    snap_frame=None if snap is None else frames[snap].frame_number,
                    analysis_frames=sorted({frames[int(round(p))].frame_number for p in picks}),
                )
            )
        index = cursor
    # Prefer the stillest, longest windows.
    windows.sort(key=lambda w: (w.median_motion, -w.duration_frames))
    return sorted(windows[:max_windows], key=lambda w: w.start_frame), motion


def _locate_snap(
    frames: Sequence[FrameAnalysis], start: int, end: int, fps: float, baseline: float
) -> Optional[int]:
    """First burst of short-baseline (0.1 s) motion near the end of a window.

    The 0.5 s window metric reacts late, so the burst is searched from half a
    second before the window's detected end.
    """
    lag = max(1, int(round(0.1 * fps)))
    search_from = max(start + lag, end - int(round(BASELINE_SECONDS * fps)) - lag)
    threshold = max(0.4, 3.0 * baseline)
    for index in range(search_from, min(len(frames), end + int(round(0.5 * fps)))):
        value = relative_motion(frames[index - lag], frames[index], lag / fps)
        if np.isfinite(value) and value >= threshold:
            # Require the burst to persist (not a single jittery frame).
            ahead = [
                relative_motion(frames[j - lag], frames[j], lag / fps)
                for j in range(index + 1, min(len(frames), index + 1 + lag))
            ]
            if ahead and np.nanmedian(ahead) >= threshold * 0.8:
                return index
    return None
