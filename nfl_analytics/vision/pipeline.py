"""Frame-by-frame tracking pipeline.

`TrackingPipeline.process` is the original `main.py` loop body: YOLO11 +
ByteTrack detection, field context, camera stabilization, temporal team
classification, movement measurement and drawing. It additionally returns a
`FrameAnalysis` record so later stages (pre-snap detection, formation
recognition) can consume the per-player results without re-running YOLO.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .camera import CameraMotionEstimator
from .classification import TrackState, analyze_detection, update_track
from .config import PipelineConfig
from .device import choose_device
from .field import build_field_context
from .movement import MovementState, update_movement
from .render import draw_label, draw_movement, draw_movement_dashboard


FALLBACK_FPS = 30.0


@dataclass(frozen=True)
class PlayerObservation:
    """One tracked person in one frame, after classification."""

    track_id: int
    box: tuple[int, int, int, int]
    label: str
    confidence: float
    on_field: bool
    player_evidence: float
    field_support: float
    referee_texture: float
    speed_px_s: float
    speed_valid: bool

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) * 0.5, float(y2))

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class FrameAnalysis:
    frame_number: int  # 1-based, as printed by the original CLI
    observations: tuple[PlayerObservation, ...]
    camera_available: bool
    camera_segment: int
    camera_reset: bool
    has_field: bool
    field_ratio: float


@dataclass
class RunSummary:
    frames: int = 0
    elapsed_seconds: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    device: str = "cpu"
    label_counts: Counter = field(default_factory=Counter)
    camera_available_frames: int = 0
    camera_reset_frames: int = 0
    camera_estimated_frames: int = 0
    camera_feature_total: int = 0
    camera_inlier_total: float = 0.0
    output_path: Optional[str] = None

    @property
    def processing_fps(self) -> float:
        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def load_detector(model_path: str):
    """Load a fresh Ultralytics model.

    A new instance is used per video because `model.track(persist=True)` keeps
    ByteTrack state on the model; reusing it would carry track IDs (and their
    team evidence) from one uploaded clip into the next.
    """
    from ultralytics import YOLO

    return YOLO(model_path)


class TrackingPipeline:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "auto",
        fps: float = FALLBACK_FPS,
        config: PipelineConfig | None = None,
        hide_non_players: bool = False,
        annotate: bool = True,
    ) -> None:
        self.config = config or PipelineConfig()
        self.device = choose_device(device)
        self.model = load_detector(model_path)
        self.fps = fps if fps and math.isfinite(fps) and fps > 0 else FALLBACK_FPS
        self.hide_non_players = hide_non_players
        self.annotate = annotate
        self.track_states: dict[int, TrackState] = defaultdict(TrackState)
        self.movement_states: dict[int, MovementState] = defaultdict(MovementState)
        self.track_labels: dict[int, str] = {}
        self.camera_estimator = CameraMotionEstimator()
        self.summary = RunSummary(fps=self.fps, device=self.device)
        self.frame_number = 0

    def process(self, frame: np.ndarray) -> FrameAnalysis:
        """Analyze one BGR frame. When `annotate` is set, draws on it in place."""
        config = self.config
        summary = self.summary
        height, width = frame.shape[:2]
        self.frame_number += 1
        frame_number = self.frame_number

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=config.confidence,
            iou=0.50,
            imgsz=config.image_size,
            max_det=config.max_detections,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        field_context = build_field_context(hsv_frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coordinates = np.empty((0, 4), dtype=int)
        track_ids = np.empty((0,), dtype=int)
        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

        clipped_boxes: list[tuple[int, int, int, int]] = []
        for coordinates_for_box in coordinates:
            x1, y1, x2, y2 = coordinates_for_box.tolist()
            clipped_boxes.append(
                (max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2))
            )
        camera = self.camera_estimator.update(gray_frame, field_context, clipped_boxes)
        if camera.available:
            summary.camera_available_frames += 1
        if camera.reset:
            summary.camera_reset_frames += 1
        if camera.tracked_features:
            summary.camera_estimated_frames += 1
            summary.camera_feature_total += camera.tracked_features
            summary.camera_inlier_total += camera.inlier_ratio

        crowded_nonfield_scene = not field_context.has_field and len(coordinates) >= 5
        active_movement_rows: list[tuple[int, str, float, float]] = []
        observations: list[PlayerObservation] = []
        for box, track_id in zip(clipped_boxes, track_ids):
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            track_id = int(track_id)
            features = analyze_detection(
                hsv_frame, field_context, box, config, crowded_nonfield_scene
            )
            label, confidence = update_track(
                self.track_states[track_id], features, frame_number, config
            )
            self.track_labels[track_id] = label
            summary.label_counts[label] += 1

            # Movement is measured only on/near the field plane. It is kept
            # from the first plausible player observation so an eventual
            # established label can show the earlier trail.
            box_height = y2 - y1
            box_width = x2 - x1
            standing_ground_geometry = (
                box_height / max(1, box_width) >= 1.15 and y2 < height - 3
            )
            movement_eligible = (
                camera.available
                and features.field_support >= 0.06
                and standing_ground_geometry
                and (
                    features.on_field
                    or features.player_evidence >= 0.50
                    or self.track_states[track_id].classification_phase == "ESTABLISHED"
                )
            )
            movement_state = self.movement_states[track_id]
            if movement_eligible:
                update_movement(
                    movement_state,
                    camera,
                    ((x1 + x2) * 0.5, float(y2)),
                    box_height,
                    frame_number,
                    self.fps,
                    config,
                )
            if self.annotate and (label != "NON-PLAYER" or not self.hide_non_players):
                if movement_eligible:
                    draw_movement(frame, box, movement_state, camera, label)
                draw_label(frame, box, track_id, label, confidence)
            speed_valid = (
                movement_eligible
                and movement_state.segment_id == camera.segment_id
                and movement_state.last_frame == frame_number
                and movement_state.steps_by_segment.get(camera.segment_id, 0) >= 3
            )
            if (
                movement_eligible
                and label in ("RAIDERS", "BRONCOS")
                and movement_state.segment_id == camera.segment_id
                and movement_state.steps_by_segment.get(camera.segment_id, 0) >= 5
            ):
                active_movement_rows.append(
                    (
                        track_id,
                        label,
                        movement_state.distance_by_segment.get(camera.segment_id, 0.0),
                        movement_state.current_speed_px_s,
                    )
                )
            observations.append(
                PlayerObservation(
                    track_id=track_id,
                    box=box,
                    label=label,
                    confidence=float(confidence),
                    on_field=features.on_field,
                    player_evidence=float(features.player_evidence),
                    field_support=float(features.field_support),
                    referee_texture=float(features.referee_texture),
                    speed_px_s=float(movement_state.current_speed_px_s) if speed_valid else 0.0,
                    speed_valid=speed_valid,
                )
            )

        if self.annotate:
            draw_movement_dashboard(frame, active_movement_rows)
            draw_motion_status(frame, camera)

        if frame_number % config.stale_track_frames == 0:
            stale_before = frame_number - config.stale_track_frames
            self.track_states = defaultdict(
                TrackState,
                {
                    track_id: state
                    for track_id, state in self.track_states.items()
                    if state.last_seen >= stale_before
                },
            )

        summary.frames = frame_number
        return FrameAnalysis(
            frame_number=frame_number,
            observations=tuple(observations),
            camera_available=camera.available,
            camera_segment=camera.segment_id,
            camera_reset=camera.reset,
            has_field=field_context.has_field,
            field_ratio=float(field_context.component_ratio),
        )

    def movement_metrics(self) -> dict:
        fps = self.fps
        track_metrics: dict[str, object] = {}
        for track_id, movement in self.movement_states.items():
            if movement.observations < 2:
                continue
            segment_metrics = []
            for segment_id, distance in movement.distance_by_segment.items():
                frame_range = movement.frame_range_by_segment.get(segment_id, [0, 0])
                elapsed_seconds = max(0.0, (frame_range[1] - frame_range[0]) / max(fps, 1e-6))
                segment_metrics.append(
                    {
                        "segment_id": segment_id,
                        "first_frame": frame_range[0],
                        "last_frame": frame_range[1],
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "distance_px": round(distance, 3),
                        "accepted_steps": movement.steps_by_segment.get(segment_id, 0),
                        "mean_speed_px_s": round(
                            distance / elapsed_seconds if elapsed_seconds > 0 else 0.0, 3
                        ),
                    }
                )
            track_metrics[str(track_id)] = {
                "last_label": self.track_labels.get(track_id, "UNKNOWN"),
                "observations": movement.observations,
                "accepted_steps": movement.accepted_steps,
                "interruptions": movement.interrupted_runs,
                "rejected_jumps": movement.rejected_jumps,
                "segments": segment_metrics,
            }
        summary = self.summary
        return {
            "coordinate_system": (
                "shot-local reference-image pixels after field homography; "
                "distances are not yards and are not comparable across segments"
            ),
            "fps": fps,
            "frames": self.frame_number,
            "camera": {
                "available_frames": summary.camera_available_frames,
                "estimated_homographies": summary.camera_estimated_frames,
                "reference_resets": summary.camera_reset_frames,
                "mean_features": round(
                    summary.camera_feature_total / max(1, summary.camera_estimated_frames), 3
                ),
                "mean_inlier_ratio": round(
                    summary.camera_inlier_total / max(1, summary.camera_estimated_frames), 4
                ),
            },
            "tracks": track_metrics,
        }


def draw_motion_status(frame: np.ndarray, camera) -> None:
    height = frame.shape[0]
    motion_status = (
        f"MOTION STABILIZED  {camera.tracked_features} pts "
        f"{camera.inlier_ratio:.0%} inliers"
        if camera.available and camera.tracked_features
        else (
            "MOTION LOCKING FIELD REFERENCE"
            if camera.tracked_features
            else "MOTION PAUSED (NO FIELD LOCK)"
        )
    )
    for color, thickness in (
        ((0, 0, 0), 3),
        ((235, 235, 235) if camera.available else (80, 180, 255), 1),
    ):
        cv2.putText(
            frame,
            motion_status,
            (12, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            thickness,
            cv2.LINE_AA,
        )


class VideoOpenError(RuntimeError):
    pass


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    """Open a video, validating that it decodes at least one frame."""
    path = Path(path)
    if not path.is_file():
        raise VideoOpenError(f"Input video does not exist: {path}")
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise VideoOpenError(f"Could not open input video: {path}")
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        video.release()
        raise VideoOpenError(f"Video has no decodable frames: {path}")
    if not fps or not math.isfinite(fps) or fps <= 0:
        fps = FALLBACK_FPS
    return video, fps, width, height, expected_frames


def create_writer(path: str | Path, fps: float, width: int, height: int, codec: str = "mp4v"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise VideoOpenError(f"Could not create output video: {path}")
    return writer


def run_video(
    input_path: str | Path,
    output_path: str | Path | None,
    model_path: str = "yolo11n.pt",
    device: str = "auto",
    max_frames: int = 0,
    hide_non_players: bool = False,
    movement_metrics_path: str | Path | None = None,
    codec: str = "mp4v",
    on_frame: Callable[[int, int, FrameAnalysis], None] | None = None,
    verbose: bool = True,
) -> tuple[RunSummary, list[FrameAnalysis], dict]:
    """Track a whole clip. Returns the summary, per-frame records and movement metrics."""
    video, fps, width, height, expected_frames = open_video(input_path)
    pipeline = TrackingPipeline(
        model_path=model_path,
        device=device,
        fps=fps,
        hide_non_players=hide_non_players,
        annotate=output_path is not None,
    )
    writer = None
    try:
        if output_path is not None:
            writer = create_writer(output_path, fps, width, height, codec)
    except VideoOpenError:
        video.release()
        raise
    summary = pipeline.summary
    summary.width, summary.height = width, height
    summary.output_path = str(output_path) if output_path is not None else None
    log = print if verbose else (lambda *_args, **_kwargs: None)
    log(f"Input: {input_path} ({expected_frames} frames, {width}x{height} @ {fps:.3f} fps)")
    log(f"Detector device: {pipeline.device}; output: {output_path}")

    frames: list[FrameAnalysis] = []
    started = time.perf_counter()
    try:
        while True:
            success, frame = video.read()
            if not success:
                break
            analysis = pipeline.process(frame)
            frames.append(analysis)
            if writer is not None:
                writer.write(frame)
            if on_frame is not None:
                on_frame(analysis.frame_number, expected_frames, analysis)
            if analysis.frame_number % 100 == 0:
                elapsed = time.perf_counter() - started
                log(
                    f"Processed frame {analysis.frame_number} "
                    f"({analysis.frame_number / elapsed:.1f} fps)"
                )
            if max_frames and analysis.frame_number >= max_frames:
                break
    finally:
        video.release()
        if writer is not None:
            writer.release()

    summary.elapsed_seconds = time.perf_counter() - started
    metrics = pipeline.movement_metrics()
    if movement_metrics_path:
        metrics_path = Path(movement_metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return summary, frames, metrics
