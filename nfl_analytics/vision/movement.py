"""Camera-stabilized, shot-local player movement in image pixels."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraMotion
from .config import PipelineConfig


@dataclass
class MovementState:
    """Camera-stabilized, shot-local movement history for one tracking ID."""

    segment_id: int = -1
    first_frame: int = 0
    last_frame: int = 0
    smoothed_reference_point: np.ndarray | None = None
    last_image_point: np.ndarray | None = None
    last_camera_to_reference: np.ndarray | None = None
    smoothed_displacement: np.ndarray | None = None
    reference_history: deque[tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=90)
    )
    recent_steps: deque[tuple[int, float]] = field(default_factory=deque)
    distance_by_segment: dict[int, float] = field(default_factory=dict)
    steps_by_segment: dict[int, int] = field(default_factory=dict)
    frame_range_by_segment: dict[int, list[int]] = field(default_factory=dict)
    observations: int = 0
    accepted_steps: int = 0
    interrupted_runs: int = 0
    rejected_jumps: int = 0
    current_speed_px_s: float = 0.0


def transform_point(point: tuple[float, float], homography: np.ndarray) -> np.ndarray:
    source = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    return cv2.perspectiveTransform(source, homography).reshape(2).astype(np.float64)


def update_movement(
    state: MovementState,
    camera: CameraMotion,
    foot_point: tuple[float, float],
    box_height: int,
    frame_number: int,
    fps: float,
    config: PipelineConfig,
) -> None:
    """Update a track in shot-local, camera-stabilized pixel coordinates."""
    if not camera.available:
        return

    reference_point = transform_point(foot_point, camera.current_to_reference)
    reference_top = transform_point(
        (foot_point[0], foot_point[1] - box_height), camera.current_to_reference
    )
    if not np.all(np.isfinite(reference_point)) or not np.all(np.isfinite(reference_top)):
        return
    reference_box_height = float(np.linalg.norm(reference_point - reference_top))
    if reference_box_height < 3.0:
        return

    gap = frame_number - state.last_frame if state.last_frame else 0
    new_run = (
        state.segment_id != camera.segment_id
        or state.smoothed_reference_point is None
        or state.last_image_point is None
        or state.last_camera_to_reference is None
        or gap <= 0
        or gap > config.movement_max_gap_frames
    )
    if new_run:
        if state.last_frame:
            state.interrupted_runs += 1
        state.segment_id = camera.segment_id
        state.smoothed_reference_point = reference_point
        state.last_image_point = np.asarray(foot_point, dtype=np.float64)
        state.last_camera_to_reference = camera.current_to_reference.copy()
        state.smoothed_displacement = None
        state.reference_history = deque(maxlen=config.movement_history_frames)
        state.reference_history.append(
            (frame_number, float(reference_point[0]), float(reference_point[1]))
        )
        state.recent_steps.clear()
        state.current_speed_px_s = 0.0
        state.first_frame = state.first_frame or frame_number
        state.last_frame = frame_number
        state.observations += 1
        state.distance_by_segment.setdefault(camera.segment_id, 0.0)
        state.steps_by_segment.setdefault(camera.segment_id, 0)
        state.frame_range_by_segment.setdefault(
            camera.segment_id, [frame_number, frame_number]
        )[1] = frame_number
        return

    # Measure displacement in the last observed image after removing camera
    # motion. This avoids the slow drift that can accumulate in a long
    # frame-to-reference transform. That longer transform remains useful only
    # for rendering a short trail in the current view.
    try:
        current_to_last = (
            np.linalg.inv(state.last_camera_to_reference)
            @ camera.current_to_reference
        )
    except np.linalg.LinAlgError:
        state.last_frame = 0
        return
    current_in_last = transform_point(foot_point, current_to_last)
    current_top_in_last = transform_point(
        (foot_point[0], foot_point[1] - box_height), current_to_last
    )
    if not np.all(np.isfinite(current_in_last)):
        state.last_frame = 0
        return
    raw_displacement = current_in_last - state.last_image_point
    raw_step = float(np.linalg.norm(raw_displacement))
    local_box_height = float(np.linalg.norm(current_in_last - current_top_in_last))
    maximum_step = max(16.0, local_box_height * 0.65 * max(1, gap))
    if raw_step > maximum_step:
        state.rejected_jumps += 1
        state.interrupted_runs += 1
        state.smoothed_reference_point = reference_point
        state.last_image_point = np.asarray(foot_point, dtype=np.float64)
        state.last_camera_to_reference = camera.current_to_reference.copy()
        state.smoothed_displacement = None
        state.reference_history.clear()
        state.reference_history.append(
            (frame_number, float(reference_point[0]), float(reference_point[1]))
        )
        state.recent_steps.clear()
        state.current_speed_px_s = 0.0
        state.last_frame = frame_number
        state.observations += 1
        return

    smoothing = 0.42 if gap == 1 else min(0.72, 0.42 + 0.06 * gap)
    smoothed = (
        (1.0 - smoothing) * state.smoothed_reference_point
        + smoothing * reference_point
    )
    if state.smoothed_displacement is None:
        state.smoothed_displacement = raw_displacement
    else:
        motion_smoothing = 0.45 if gap == 1 else min(0.75, 0.45 + 0.05 * gap)
        state.smoothed_displacement = (
            (1.0 - motion_smoothing) * state.smoothed_displacement
            + motion_smoothing * raw_displacement
        )
    step_distance = float(np.linalg.norm(state.smoothed_displacement))
    state.smoothed_reference_point = smoothed
    state.last_image_point = np.asarray(foot_point, dtype=np.float64)
    state.last_camera_to_reference = camera.current_to_reference.copy()
    state.last_frame = frame_number
    state.observations += 1
    state.accepted_steps += 1
    state.distance_by_segment[camera.segment_id] = (
        state.distance_by_segment.get(camera.segment_id, 0.0) + step_distance
    )
    state.steps_by_segment[camera.segment_id] = (
        state.steps_by_segment.get(camera.segment_id, 0) + 1
    )
    state.frame_range_by_segment.setdefault(
        camera.segment_id, [frame_number, frame_number]
    )[1] = frame_number
    state.reference_history.append(
        (frame_number, float(smoothed[0]), float(smoothed[1]))
    )
    state.recent_steps.append((frame_number, step_distance))
    speed_window_frames = max(2, int(round(config.movement_speed_window_seconds * fps)))
    while (
        state.recent_steps
        and state.recent_steps[0][0] < frame_number - speed_window_frames
    ):
        state.recent_steps.popleft()
    if state.recent_steps:
        first_speed_frame = state.recent_steps[0][0]
        elapsed_seconds = max(1.0 / fps, (frame_number - first_speed_frame + 1) / fps)
        state.current_speed_px_s = (
            sum(distance for _, distance in state.recent_steps) / elapsed_seconds
        )
