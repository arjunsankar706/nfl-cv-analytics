"""Track and classify players in an NFL broadcast clip.

The detector is intentionally still the small YOLO11 model used by the original
project. Most accuracy work happens after detection: field-context and box-
geometry filtering, jersey-region colour/texture features, and temporal
evidence attached to each persistent tracking ID.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


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


@dataclass
class TrackState:
    seen: int = 0
    last_seen: int = 0
    player_evidence: float = 0.0
    on_field_streak: int = 0
    off_field_streak: int = 0
    team_observations: int = 0
    referee_texture: float = 0.0
    referee_observations: int = 0
    scores: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in TEAM_NAMES}
    )
    identity_scores: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in TEAM_NAMES}
    )
    best_observation_scores: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in TEAM_NAMES}
    )
    identity_support_frames: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in TEAM_NAMES}
    )
    label: str = "UNKNOWN"
    classification_phase: str = "UNKNOWN"
    candidate_label: str = "UNKNOWN"
    candidate_frames: int = 0
    candidate_weight: float = 0.0
    established_label: str = "UNKNOWN"
    challenger_label: str = "UNKNOWN"
    challenger_frames: int = 0
    challenger_weight: float = 0.0


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


@dataclass(frozen=True)
class CameraMotion:
    """Mapping from the current image into one reference image for this shot."""

    current_to_reference: np.ndarray
    segment_id: int
    available: bool
    tracked_features: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    reset: bool = False


@dataclass(frozen=True)
class DetectionFeatures:
    on_field: bool
    player_evidence: float
    field_support: float
    team_scores: dict[str, float]
    force_nonplayer: bool = False
    vote_weight: float = 0.0
    referee_texture: float = 0.0
    closeup_equipment: bool = False
    rules_out_referee: bool = False


@dataclass(frozen=True)
class FieldContext:
    mask: np.ndarray
    boundary: np.ndarray
    component_ratio: float
    has_field: bool


@dataclass(frozen=True)
class UniformAnalysis:
    scores: dict[str, float]
    vote_weight: float
    referee_texture: float
    equipment_score: float
    light_pants_ratio: float
    number_structure: float = 0.0
    player_visual_score: float = 0.0
    force_nonplayer: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_VIDEO, help="input MP4 path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output MP4 path")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path")
    parser.add_argument(
        "--device", default="auto", help="Ultralytics device (auto, cpu, mps, 0, ...)"
    )
    parser.add_argument(
        "--hide-non-players",
        action="store_true",
        help="do not draw detections rejected as crowd/non-player",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="optional short diagnostic run; 0 processes the full video",
    )
    parser.add_argument(
        "--movement-metrics",
        default="",
        help="optional JSON path for camera and per-track movement diagnostics",
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_field_mask(hsv_frame: np.ndarray) -> np.ndarray:
    """Return a conservative green-turf candidate mask."""
    mask = cv2.inRange(
        hsv_frame,
        np.array((28, 35, 32), dtype=np.uint8),
        np.array((98, 255, 255), dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def build_field_context(hsv_frame: np.ndarray) -> FieldContext:
    """Find the connected playing surface and its upper boundary by column.

    Isolated green clothing/signage is deliberately discarded.  The smoothed
    boundary lets us distinguish a player standing well inside the field from
    staff or spectators whose boxes merely overlap turf in the background.
    """
    candidate = build_field_mask(hsv_frame)
    binary = (candidate > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    height, width = binary.shape
    if count <= 1:
        return FieldContext(np.zeros_like(candidate), np.full(width, height), 0.0, False)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    component_ratio = largest_area / float(height * width)

    boundary = np.full(width, np.nan, dtype=np.float32)
    component_bool = component > 0
    populated_columns = np.any(component_bool, axis=0)
    boundary[populated_columns] = np.argmax(component_bool[:, populated_columns], axis=0)
    valid = np.flatnonzero(~np.isnan(boundary))
    if valid.size >= 2:
        boundary = np.interp(np.arange(width), valid, boundary[valid]).astype(np.float32)
        boundary = cv2.GaussianBlur(boundary.reshape(1, -1), (101, 1), 0).reshape(-1)
    else:
        boundary.fill(height)

    return FieldContext(
        component,
        boundary,
        component_ratio,
        component_ratio >= 0.035,
    )


class CameraMotionEstimator:
    """Estimate broadcast-camera motion from stable features on the field.

    The homography maps each current frame into a shot-local reference image.
    Person boxes are excluded from the feature mask so player motion cannot be
    mistaken for a pan or zoom. Failed estimates and field-less broadcast
    shots begin a new segment; movement is never bridged across those breaks.
    """

    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.previous_feature_mask: np.ndarray | None = None
        self.previous_to_reference = np.eye(3, dtype=np.float64)
        self.segment_id = 0
        self.previous_available = False
        self.lock_streak = 0

    @staticmethod
    def _feature_mask(
        field: FieldContext,
        boxes: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        mask = field.mask.copy()
        mask = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
        height, width = mask.shape
        for x1, y1, x2, y2 in boxes:
            pad_x = max(5, int((x2 - x1) * 0.20))
            pad_y = max(5, int((y2 - y1) * 0.10))
            cv2.rectangle(
                mask,
                (max(0, x1 - pad_x), max(0, y1 - pad_y)),
                (min(width - 1, x2 + pad_x), min(height - 1, y2 + pad_y)),
                0,
                -1,
            )
        return mask

    def _new_segment(
        self,
        gray: np.ndarray,
        feature_mask: np.ndarray,
        available: bool,
    ) -> CameraMotion:
        self.segment_id += 1
        self.previous_gray = gray
        self.previous_feature_mask = feature_mask
        self.previous_to_reference = np.eye(3, dtype=np.float64)
        self.previous_available = available
        self.lock_streak = 0
        return CameraMotion(
            np.eye(3, dtype=np.float64),
            self.segment_id,
            False,
            reset=True,
        )

    def update(
        self,
        gray: np.ndarray,
        field: FieldContext,
        boxes: list[tuple[int, int, int, int]],
    ) -> CameraMotion:
        # Optical flow runs at half resolution for a substantial speedup. The
        # estimated homography is conjugated back into full-image coordinates
        # before any player point uses it.
        tracking_scale = 0.5
        full_mask = self._feature_mask(field, boxes)
        tracking_size = (
            max(1, int(round(gray.shape[1] * tracking_scale))),
            max(1, int(round(gray.shape[0] * tracking_scale))),
        )
        gray = cv2.resize(gray, tracking_size, interpolation=cv2.INTER_AREA)
        current_mask = cv2.resize(
            full_mask, tracking_size, interpolation=cv2.INTER_NEAREST
        )
        enough_field = field.has_field and cv2.countNonZero(current_mask) >= 1250
        if self.previous_gray is None:
            return self._new_segment(gray, current_mask, enough_field)

        if (
            not enough_field
            or not self.previous_available
            or self.previous_feature_mask is None
            or cv2.countNonZero(self.previous_feature_mask) < 1250
        ):
            return self._new_segment(gray, current_mask, enough_field)

        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=300,
            qualityLevel=0.012,
            minDistance=8,
            mask=self.previous_feature_mask,
            blockSize=7,
        )
        if previous_points is None or len(previous_points) < 24:
            return self._new_segment(gray, current_mask, enough_field)

        current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or forward_status is None or forward_error is None:
            return self._new_segment(gray, current_mask, enough_field)

        previous_xy = previous_points.reshape(-1, 2)
        current_xy = current_points.reshape(-1, 2)
        status = (forward_status.ravel() > 0) & (forward_error.ravel() <= 18.0)
        image_height, image_width = gray.shape
        current_x = np.rint(current_xy[:, 0]).astype(int)
        current_y = np.rint(current_xy[:, 1]).astype(int)
        inside = (
            (current_x >= 0)
            & (current_x < image_width)
            & (current_y >= 0)
            & (current_y < image_height)
        )
        status &= inside
        valid_indices = np.flatnonzero(status)
        if valid_indices.size:
            on_current_field = current_mask[
                current_y[valid_indices], current_x[valid_indices]
            ] > 0
            valid_indices = valid_indices[on_current_field]
        if valid_indices.size < 20:
            return self._new_segment(gray, current_mask, enough_field)

        previous_valid = previous_xy[valid_indices]
        current_valid = current_xy[valid_indices]
        current_to_previous, inliers = cv2.findHomography(
            current_valid,
            previous_valid,
            cv2.RANSAC,
            2.5,
            maxIters=1500,
            confidence=0.995,
        )
        if current_to_previous is None or inliers is None:
            return self._new_segment(gray, current_mask, enough_field)

        inlier_mask = inliers.ravel() > 0
        inlier_count = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inlier_count / max(1, len(current_valid))
        if inlier_count < 16 or inlier_ratio < 0.45:
            return self._new_segment(gray, current_mask, enough_field)

        if abs(current_to_previous[2, 2]) < 1e-8:
            return self._new_segment(gray, current_mask, enough_field)
        current_to_previous = current_to_previous / current_to_previous[2, 2]
        if not np.all(np.isfinite(current_to_previous)):
            return self._new_segment(gray, current_mask, enough_field)
        linear_scale = float(np.sqrt(abs(np.linalg.det(current_to_previous[:2, :2]))))
        if not 0.72 <= linear_scale <= 1.38:
            return self._new_segment(gray, current_mask, enough_field)
        if np.max(np.abs(current_to_previous[2, :2])) > 0.004:
            return self._new_segment(gray, current_mask, enough_field)

        predicted_previous = cv2.perspectiveTransform(
            current_valid[inlier_mask].reshape(-1, 1, 2), current_to_previous
        ).reshape(-1, 2)
        reprojection_error = float(
            np.median(
                np.linalg.norm(
                    predicted_previous - previous_valid[inlier_mask], axis=1
                )
            )
        )
        if reprojection_error > 1.75:
            return self._new_segment(gray, current_mask, enough_field)

        scale_matrix = np.array(
            [[tracking_scale, 0.0, 0.0], [0.0, tracking_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        current_to_previous_full = (
            np.linalg.inv(scale_matrix) @ current_to_previous @ scale_matrix
        )
        current_to_reference = self.previous_to_reference @ current_to_previous_full
        current_to_reference /= current_to_reference[2, 2]
        self.previous_gray = gray
        self.previous_feature_mask = current_mask
        self.previous_to_reference = current_to_reference
        self.previous_available = True
        self.lock_streak += 1
        return CameraMotion(
            current_to_reference,
            self.segment_id,
            self.lock_streak >= 3,
            tracked_features=len(current_valid),
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
        )


def clipped_region(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    height, width = image.shape[:2]
    return image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]


def ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0


def field_and_geometry_evidence(
    field: FieldContext,
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    minimum_height: int,
) -> tuple[bool, float, float, bool, float]:
    x1, y1, x2, y2 = box
    frame_height, frame_width = frame_shape[:2]
    box_width = x2 - x1
    box_height = y2 - y1
    height_ratio = box_height / frame_height
    width_ratio = box_width / frame_width
    aspect = box_height / max(1, box_width)

    pad_x = max(5, int(box_width * 0.22))
    foot_region = clipped_region(
        field.mask,
        x1 - pad_x,
        y1 + int(box_height * 0.70),
        x2 + pad_x,
        y2 + max(6, int(box_height * 0.12)),
    )
    field_support = ratio(foot_region > 0)

    sample_x1 = max(0, x1 - pad_x)
    sample_x2 = min(frame_width, x2 + pad_x)
    if sample_x2 > sample_x1:
        field_boundary = float(np.median(field.boundary[sample_x1:sample_x2]))
    else:
        field_boundary = float(frame_height)
    field_depth = y2 - field_boundary
    depth_ratio = field_depth / max(1.0, float(box_height))

    plausible_size = (
        box_height >= minimum_height
        and width_ratio < 0.48
        and 0.78 <= aspect <= 5.8
    )
    # People on the sideline may have green directly behind their legs, but
    # their feet remain close to the upper edge of the connected field.  Real
    # on-field people extend materially inside that surface.
    sideline_scale_mismatch = field.has_field and depth_ratio < 0.62
    top_truncated_background = (
        y1 < frame_height * 0.02
        and y2 < frame_height * 0.45
        and height_ratio > 0.20
    )
    plausible_size = (
        plausible_size and not sideline_scale_mismatch and not top_truncated_background
    )
    close_up_candidate = height_ratio >= 0.23 and box_width >= 42
    on_turf = field.has_field and (
        (depth_ratio >= 0.65 and field_support >= 0.025) or depth_ratio >= 0.95
    )

    size_score = np.clip((box_height - minimum_height) / 70.0, 0.0, 1.0)
    evidence = float(
        np.clip(
            0.45 * min(field_support / 0.22, 1.0)
            + 0.40 * np.clip((depth_ratio - 0.20) / 0.90, 0.0, 1.0)
            + 0.15 * size_score,
            0.0,
            1.0,
        )
    )
    if not plausible_size:
        evidence *= 0.25

    return (
        plausible_size and on_turf,
        evidence,
        field_support,
        close_up_candidate,
        depth_ratio,
    )


def analyze_uniform(
    hsv_frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> UniformAnalysis:
    """Measure team colour, referee texture, and close-up equipment evidence."""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    head = clipped_region(
        hsv_frame,
        x1 + int(box_width * 0.14),
        y1,
        x2 - int(box_width * 0.14),
        y1 + int(box_height * 0.30),
    )
    equipment_score = 0.0
    if head.size:
        head = cv2.resize(head, (40, 40), interpolation=cv2.INTER_AREA)
        head_h, head_s, head_v = cv2.split(head)
        silver_shell = (head_s < 78) & (head_v > 125)
        coloured_shell = (
            (((head_h < 28) | (head_h > 172)) & (head_s > 90) & (head_v > 70))
            | ((head_h >= 88) & (head_h <= 132) & (head_s > 72) & (head_v > 35))
        )
        skin = (
            (head_h < 24)
            & (head_s >= 35)
            & (head_s <= 185)
            & (head_v >= 55)
        )
        edges = cv2.Canny(head_v, 55, 135)
        shell_ratio = max(ratio(silver_shell), ratio(coloured_shell))
        equipment_score = float(
            np.clip(1.55 * shell_ratio + 0.30 * ratio(edges > 0) - 0.70 * ratio(skin), 0.0, 1.0)
        )

    jersey = clipped_region(
        hsv_frame,
        x1 + int(box_width * 0.20),
        y1 + int(box_height * 0.18),
        x2 - int(box_width * 0.20),
        y1 + int(box_height * 0.58),
    )
    if jersey.size == 0:
        return UniformAnalysis({}, 0.0, 0.0, equipment_score, 0.0)

    jersey = cv2.resize(jersey, (48, 48), interpolation=cv2.INTER_AREA)
    hue, saturation, value = cv2.split(jersey)
    white = (saturation < 72) & (value > 145)
    dark = value < 108
    orange = (((hue < 28) | (hue > 172)) & (saturation > 90) & (value > 75))
    navy = (
        (hue >= 88)
        & (hue <= 132)
        & (saturation > 72)
        & (value >= 38)
        & (value <= 175)
    )
    neutral_mid = (saturation < 62) & (value >= 90) & (value <= 190)

    white_ratio = ratio(white)
    dark_ratio = ratio(dark)
    orange_ratio = ratio(orange)
    navy_ratio = ratio(navy)
    neutral_ratio = ratio(neutral_mid)

    # Jersey digits are compact, high-contrast components near the middle of
    # the torso. This does not attempt OCR; it only distinguishes number-like
    # structure from an unbroken sweatshirt or jacket.
    contrast_mask = white if dark_ratio >= white_ratio else dark
    contrast_mask = contrast_mask.astype(np.uint8)
    component_count, _, component_stats, component_centroids = (
        cv2.connectedComponentsWithStats(contrast_mask, 8)
    )
    number_components = []
    for component_index in range(1, component_count):
        component_x, component_y, component_width, component_height, component_area = (
            component_stats[component_index]
        )
        centre_x, centre_y = component_centroids[component_index]
        if (
            10 <= component_area <= 1000
            and 5 <= component_width <= 40
            and 9 <= component_height <= 42
            and 9 <= centre_x <= 39
            and 6 <= centre_y <= 43
        ):
            area_score = np.clip((component_area - 10) / 120.0, 0.0, 1.0)
            height_score = np.clip((component_height - 8) / 22.0, 0.0, 1.0)
            number_components.append(float(0.55 * area_score + 0.45 * height_score))
    number_components.sort(reverse=True)
    number_structure = float(
        np.clip(
            (number_components[0] if number_components else 0.0)
            + (0.30 * number_components[1] if len(number_components) > 1 else 0.0),
            0.0,
            1.0,
        )
    )

    pants = clipped_region(
        hsv_frame,
        x1 + int(box_width * 0.16),
        y1 + int(box_height * 0.54),
        x2 - int(box_width * 0.16),
        y1 + int(box_height * 0.84),
    )
    if pants.size:
        pants = cv2.resize(pants, (32, 24), interpolation=cv2.INTER_AREA)
        _, pants_saturation, pants_value = cv2.split(pants)
        light_pants_ratio = ratio((pants_saturation < 82) & (pants_value > 125))
    else:
        light_pants_ratio = 0.0

    # Referee stripes create alternating light/dark vertical columns. This is
    # materially different from a single jersey number on a solid shirt.
    central = slice(5, 43)
    light_by_column = np.mean(white[central, :], axis=0)
    dark_by_column = np.mean(dark[central, :], axis=0)
    light_columns = ratio(light_by_column > 0.42)
    dark_columns = ratio(dark_by_column > 0.42)
    column_balance = min(light_columns, dark_columns) * 2.0
    column_class = np.where(light_by_column > dark_by_column, 1, 0)
    alternations = float(np.mean(column_class[1:] != column_class[:-1]))
    stripe_score = 0.64 * column_balance + 0.36 * min(alternations / 0.24, 1.0)

    # True official stripes remain vertically aligned through several torso
    # bands and contain more x-oriented than y-oriented edge energy. Jersey
    # numbers can have similar colour totals, but are spatially localized.
    signed_pattern = white.astype(np.float32) - dark.astype(np.float32)
    band_profiles = [
        np.mean(signed_pattern[3:16, :], axis=0),
        np.mean(signed_pattern[17:31, :], axis=0),
        np.mean(signed_pattern[32:45, :], axis=0),
    ]
    correlations = []
    for first, second in ((0, 1), (1, 2), (0, 2)):
        if np.std(band_profiles[first]) > 0.08 and np.std(band_profiles[second]) > 0.08:
            correlations.append(
                float(np.corrcoef(band_profiles[first], band_profiles[second])[0, 1])
            )
    band_consistency = max(0.0, float(np.median(correlations))) if correlations else 0.0
    value_float = value.astype(np.float32)
    gradient_x = float(np.mean(np.abs(cv2.Sobel(value_float, cv2.CV_32F, 1, 0, ksize=3))))
    gradient_y = float(np.mean(np.abs(cv2.Sobel(value_float, cv2.CV_32F, 0, 1, ksize=3))))
    vertical_orientation = gradient_x / (gradient_x + gradient_y + 1e-6)
    referee_texture = float(
        np.clip(
            0.36 * column_balance
            + 0.24 * min(alternations / 0.18, 1.0)
            + 0.25 * band_consistency
            + 0.15 * np.clip((vertical_orientation - 0.42) / 0.25, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    repeated_stripes = (
        light_columns >= 0.32 and dark_columns >= 0.32 and alternations >= 0.08
    )
    side_view_stripes = (
        white_ratio >= 0.18
        and dark_ratio >= 0.25
        and alternations >= 0.045
        and light_pants_ratio < 0.25
        and orange_ratio < 0.04
    )
    if repeated_stripes:
        pass
    elif side_view_stripes:
        stripe_score *= 0.78
    else:
        stripe_score *= 0.38

    white_dominance = max(white_ratio - dark_ratio, 0.0)
    dark_dominance = max(dark_ratio - white_ratio, 0.0)

    height_ratio = box_height / hsv_frame.shape[0]
    referee = (
        1.10 * stripe_score
        + 1.25 * referee_texture
        + 0.75 * min(white_ratio, dark_ratio) * 2.0
        - 2.8 * orange_ratio
        - 1.05 * light_pants_ratio
        - 0.55 * max(white_ratio - 0.58, 0.0)
    )
    compact_closeup = (
        box_width / hsv_frame.shape[1] > 0.18
        and box_height / max(1, box_width) < 1.65
    )
    if compact_closeup:
        referee *= 0.10
    broncos = (
        1.55 * white_ratio
        + 3.2 * orange_ratio
        + 0.20 * dark_ratio
        + 0.72 * white_dominance
        + 0.28 * light_pants_ratio
        - 0.72 * stripe_score
    )
    raiders = (
        1.62 * dark_ratio
        + 0.32 * neutral_ratio
        + 0.18 * white_ratio
        + 0.50 * dark_dominance
        + 0.18 * light_pants_ratio
        - 2.6 * orange_ratio
        - 0.55 * stripe_score
    )

    scores = {
        "RAIDERS": max(0.0, raiders),
        "BRONCOS": max(0.0, broncos),
        "REF": max(0.0, referee),
    }
    strongest_team = max(raiders, broncos)
    team_uniform_strength = float(np.clip((strongest_team - 0.45) / 1.45, 0.0, 1.0))
    pants_consistency = float(np.clip((light_pants_ratio - 0.08) / 0.45, 0.0, 1.0))
    player_visual_score = float(
        np.clip(
            0.34 * equipment_score
            + 0.26 * number_structure
            + 0.22 * pants_consistency
            + 0.18 * team_uniform_strength,
            0.0,
            1.0,
        )
    )
    probable_uniform_kit = (
        equipment_score >= 0.58
        and light_pants_ratio >= 0.12
        and (number_structure >= 0.18 or team_uniform_strength >= 0.34)
    )
    obvious_staff_colour = (
        (orange_ratio > 0.30 and white_ratio < 0.25)
        or (navy_ratio > 0.34 and white_ratio < 0.18)
    ) and not probable_uniform_kit
    upper_sideline_dark_trousers = (
        y2 / hsv_frame.shape[0] < 0.48
        and light_pants_ratio < 0.06
        and (white_ratio > 0.40 or dark_ratio > 0.46)
        and not repeated_stripes
        and not side_view_stripes
        and not probable_uniform_kit
    )
    if obvious_staff_colour or upper_sideline_dark_trousers:
        return UniformAnalysis(
            {},
            0.0,
            referee_texture,
            equipment_score,
            light_pants_ratio,
            number_structure,
            player_visual_score,
            True,
        )
    sharpness = float(np.var(cv2.Laplacian(value, cv2.CV_32F)))
    sharpness_quality = float(np.clip((sharpness - 70.0) / 760.0, 0.0, 1.0))
    resolution_quality = float(np.clip((box_height - 38.0) / 100.0, 0.0, 1.0))
    aspect = box_height / max(1, box_width)
    geometry_quality = 1.0 if 1.05 <= aspect <= 4.8 else 0.45
    vote_weight = sharpness_quality * resolution_quality * geometry_quality
    if vote_weight < 0.12:
        vote_weight = 0.0
    return UniformAnalysis(
        scores,
        vote_weight,
        referee_texture,
        equipment_score,
        light_pants_ratio,
        number_structure,
        player_visual_score,
    )


def analyze_detection(
    hsv_frame: np.ndarray,
    field: FieldContext,
    box: tuple[int, int, int, int],
    config: PipelineConfig,
    crowded_nonfield_scene: bool = False,
) -> DetectionFeatures:
    on_field, evidence, support, closeup_candidate, depth_ratio = field_and_geometry_evidence(
        field, box, hsv_frame.shape, config.min_box_height
    )
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    aspect = (y2 - y1) / max(1, x2 - x1)
    plausible_person_crop = box_height >= config.min_box_height and (
        0.78 <= aspect <= 5.8
        or (closeup_candidate and 0.45 <= aspect < 0.78)
    )
    uniform = (
        analyze_uniform(hsv_frame, box)
        if plausible_person_crop
        else UniformAnalysis({}, 0.0, 0.0, 0.0, 0.0)
    )
    strongest_team = max(
        uniform.scores.get("RAIDERS", 0.0), uniform.scores.get("BRONCOS", 0.0)
    )
    referee_score = uniform.scores.get("REF", 0.0)
    helmet_and_uniform = (
        uniform.equipment_score >= 0.58
        and uniform.light_pants_ratio >= 0.12
        and (
            uniform.number_structure >= 0.18
            or uniform.player_visual_score >= 0.48
        )
        and strongest_team >= 0.72
    )
    numbered_closeup = (
        closeup_candidate
        and uniform.equipment_score >= 0.50
        and uniform.number_structure >= 0.38
        and uniform.player_visual_score >= 0.50
        and strongest_team >= 0.72
    )
    body_uniform_closeup = (
        closeup_candidate
        and uniform.light_pants_ratio >= 0.22
        and strongest_team >= 0.78
        and referee_score <= 0.55 * strongest_team
        and (
            uniform.number_structure >= 0.55
            or uniform.player_visual_score >= 0.33
        )
    )
    near_playing_surface = (
        field.has_field and depth_ratio >= -0.30 and support >= 0.08
    )
    visual_player_candidate = (
        not uniform.force_nonplayer
        and (helmet_and_uniform or numbered_closeup or body_uniform_closeup)
        and (
            on_field
            or near_playing_surface
            or uniform.player_visual_score >= 0.64
            or (
                body_uniform_closeup
                and box_height / hsv_frame.shape[0] >= 0.45
            )
        )
    )
    if crowded_nonfield_scene:
        # Fan clothing can match team colours. In a dense no-field shot, demand
        # the complete helmet/number/pants pattern rather than colour alone.
        visual_player_candidate = visual_player_candidate and (
            uniform.equipment_score >= 0.68
            and uniform.number_structure >= 0.30
            and uniform.light_pants_ratio >= 0.15
            and uniform.player_visual_score >= 0.66
        )
    equipment_closeup = (
        closeup_candidate
        and visual_player_candidate
        and not uniform.force_nonplayer
    )
    referee_closeup = (
        closeup_candidate
        and aspect >= 1.65
        and uniform.referee_texture >= 0.68
        and not uniform.force_nonplayer
    )
    borderline_on_field_referee = (
        field.has_field
        and depth_ratio >= 0.40
        and uniform.referee_texture >= 0.42
        and referee_score >= 1.10 * strongest_team
        and not uniform.force_nonplayer
    )
    if borderline_on_field_referee:
        on_field = True
        evidence = max(evidence, 0.62)
    # A no-field broadcast cut is commonly a crowd shot. Size alone is not
    # sufficient; require helmet/equipment evidence or very strong stripes.
    weak_field_context = field.component_ratio < 0.12
    if weak_field_context and not crowded_nonfield_scene and (equipment_closeup or referee_closeup):
        on_field = True
        evidence = max(evidence, 0.78)

    if visual_player_candidate:
        # `on_field` is the temporal pipeline's player-eligibility signal. A
        # strong uniform/equipment match can supply it outside the turf mask.
        on_field = True
        evidence = max(evidence, 0.58 + 0.36 * uniform.player_visual_score)

    force_nonplayer = uniform.force_nonplayer or (
        crowded_nonfield_scene and not (visual_player_candidate or referee_closeup)
    ) or (
        not field.has_field
        and not (visual_player_candidate or equipment_closeup or referee_closeup)
    )
    if force_nonplayer:
        on_field = False
        evidence = 0.0
    scores = uniform.scores if on_field else {}
    strong_team_over_referee = strongest_team >= 1.18 * referee_score
    obvious_player_equipment = (
        visual_player_candidate
        and uniform.equipment_score >= 0.78
        and uniform.number_structure >= 0.35
        and uniform.light_pants_ratio >= 0.12
        and uniform.player_visual_score >= 0.60
    )
    compact_equipped_player = visual_player_candidate and (
        (closeup_candidate and aspect < 1.65)
        or (uniform.equipment_score >= 0.72 and strong_team_over_referee)
    )
    rules_out_referee = (
        uniform.light_pants_ratio >= 0.28
        or compact_equipped_player
        or obvious_player_equipment
    )
    return DetectionFeatures(
        on_field,
        evidence,
        support,
        scores,
        force_nonplayer,
        uniform.vote_weight,
        uniform.referee_texture,
        equipment_closeup or referee_closeup,
        rules_out_referee,
    )


def update_track(
    state: TrackState,
    features: DetectionFeatures,
    frame_number: int,
    config: PipelineConfig,
) -> tuple[str, float]:
    state.seen += 1
    state.last_seen = frame_number
    state.player_evidence = 0.84 * state.player_evidence + 0.16 * (
        features.player_evidence if features.on_field else 0.0
    )

    if features.on_field:
        state.on_field_streak += 1
        state.off_field_streak = 0
    else:
        state.off_field_streak += 1
        state.on_field_streak = 0

    for name in TEAM_NAMES:
        state.scores[name] *= config.history_decay
    if features.rules_out_referee:
        state.scores["REF"] *= 0.62
        state.referee_texture *= 0.72

    if features.force_nonplayer:
        if (
            state.classification_phase == "ESTABLISHED"
            and state.established_label in TEAM_NAMES
            and state.off_field_streak <= config.established_context_grace_frames
        ):
            # A close crop or one bad segmentation frame must not turn a known
            # player into staff/crowd. Require sustained contradictory context.
            state.player_evidence *= 0.92
            state.label = state.established_label
            belief_scores = {
                name: (
                    state.scores[name]
                    + 0.75 * state.identity_scores[name]
                    + 1.50 * state.best_observation_scores[name]
                )
                for name in TEAM_NAMES
            }
            belief_total = sum(belief_scores.values()) + 1e-6
            return state.label, belief_scores[state.label] / belief_total
        state.player_evidence *= 0.25
        state.label = "NON-PLAYER"
        return state.label, 1.0

    reliable_observation = (
        features.on_field
        and bool(features.team_scores)
        and features.vote_weight >= config.minimum_vote_weight
    )

    # Preserve identity evidence separately from the short-term, decaying
    # scores. Clear observations contribute substantially; tiny crops can add
    # only a very small team vote when their repeated colour/context evidence
    # is decisive. Low-quality crops never add referee identity evidence.
    if features.on_field and features.team_scores:
        raiders_score = features.team_scores.get("RAIDERS", 0.0)
        broncos_score = features.team_scores.get("BRONCOS", 0.0)
        referee_score = features.team_scores.get("REF", 0.0)
        if raiders_score >= broncos_score:
            frame_team = "RAIDERS"
            frame_team_score = raiders_score
            other_team_score = broncos_score
        else:
            frame_team = "BRONCOS"
            frame_team_score = broncos_score
            other_team_score = raiders_score
        team_pair_confidence = frame_team_score / (
            frame_team_score + other_team_score + 1e-6
        )
        team_identity_weight = 0.0
        team_context_clear = (
            team_pair_confidence >= 0.68
            and (
                features.referee_texture <= 0.38
                or frame_team_score >= 1.25 * referee_score
            )
        )
        if reliable_observation and team_context_clear:
            team_clarity = float(
                np.clip((team_pair_confidence - 0.60) / 0.30, 0.15, 1.0)
            )
            team_identity_weight = features.vote_weight * team_clarity
        elif (
            team_pair_confidence >= 0.70
            and features.referee_texture <= 0.34
            and frame_team_score >= 1.25 * referee_score
        ):
            team_identity_weight = 0.035

        if team_identity_weight > 0.0:
            state.identity_scores[frame_team] = min(
                16.0,
                state.identity_scores[frame_team] + team_identity_weight,
            )
            state.identity_support_frames[frame_team] += 1
            state.best_observation_scores[frame_team] = max(
                state.best_observation_scores[frame_team], team_identity_weight
            )

        referee_identity_clear = (
            reliable_observation
            and features.vote_weight >= 0.35
            and features.referee_texture >= 0.50
            and referee_score >= 1.30 * frame_team_score
            and not features.rules_out_referee
        )
        if referee_identity_clear:
            referee_clarity = float(
                np.clip(
                    (referee_score - frame_team_score)
                    / (referee_score + frame_team_score + 1e-6),
                    0.10,
                    1.0,
                )
            )
            referee_identity_weight = features.vote_weight * referee_clarity
            state.identity_scores["REF"] = min(
                16.0,
                state.identity_scores["REF"] + referee_identity_weight,
            )
            state.identity_support_frames["REF"] += 1
            state.best_observation_scores["REF"] = max(
                state.best_observation_scores["REF"], referee_identity_weight
            )

    if reliable_observation:
        usable_scores = dict(features.team_scores)
        if features.rules_out_referee:
            usable_scores["REF"] = 0.0
        total = sum(usable_scores.values()) + 1e-6
        for name, score in usable_scores.items():
            state.scores[name] += features.vote_weight * score / total
        state.team_observations += 1
        state.referee_texture = (
            0.88 * state.referee_texture + 0.12 * features.referee_texture
        )
        if features.referee_texture >= 0.36 and not features.rules_out_referee:
            state.referee_observations += 1

    # Require repeated geometric support unless the current observation is
    # exceptionally strong. Established players survive a few occluded frames.
    accepted_player = (
        (features.on_field and features.player_evidence >= 0.72)
        or state.on_field_streak >= 2
        or features.closeup_equipment
        or (state.player_evidence >= 0.46 and state.off_field_streak <= 4)
        or (
            state.classification_phase == "ESTABLISHED"
            and state.off_field_streak <= config.established_context_grace_frames
        )
    )
    if not accepted_player:
        state.label = "NON-PLAYER" if state.off_field_streak >= 2 else "UNKNOWN"
        return state.label, max(0.0, 1.0 - state.player_evidence)

    eligible_scores = {
        name: (
            state.scores[name]
            + 0.75 * state.identity_scores[name]
            + 1.50 * state.best_observation_scores[name]
        )
        for name in TEAM_NAMES
    }
    if (
        features.rules_out_referee
        or state.referee_observations < 2
        or state.referee_texture < 0.32
    ):
        eligible_scores["REF"] = 0.0
    ranked = sorted(eligible_scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ranked[0]
    second_score = ranked[1][1]
    score_total = sum(eligible_scores.values()) + 1e-6
    confidence = best_score / score_total
    margin = (best_score - second_score) / score_total

    if best_name == "REF":
        strongest_team_score = max(
            state.scores["RAIDERS"], state.scores["BRONCOS"]
        )
        referee_pair_total = state.scores["REF"] + strongest_team_score + 1e-6
        referee_pair_confidence = state.scores["REF"] / referee_pair_total
        referee_pair_margin = (
            state.scores["REF"] - strongest_team_score
        ) / referee_pair_total
        decisive = (
            state.referee_observations >= 6
            and state.referee_texture >= 0.38
            and referee_pair_confidence >= 0.52
            and referee_pair_margin >= 0.04
            and not features.rules_out_referee
        )
    else:
        decisive = confidence >= 0.54 and margin >= 0.10
    proposed = best_name if decisive and best_score > 0.0 else "UNKNOWN"

    if state.classification_phase != "ESTABLISHED":
        # Candidate evidence is never displayed. Ambiguous and low-quality
        # frames simply leave the track UNKNOWN; a reliable contradictory
        # observation starts a new candidate instead of polluting the old one.
        if reliable_observation and proposed in TEAM_NAMES:
            if proposed == state.candidate_label:
                state.candidate_frames += 1
                state.candidate_weight += features.vote_weight
            else:
                state.candidate_label = proposed
                state.candidate_frames = 1
                state.candidate_weight = features.vote_weight
            state.classification_phase = "CANDIDATE"
        elif reliable_observation:
            # Ambiguous reliable crops erode a candidate without discarding a
            # long run of prior evidence. A different decisive class still
            # replaces the candidate immediately above.
            state.candidate_frames = max(0, state.candidate_frames - 1)
            state.candidate_weight *= 0.90
            if state.candidate_frames == 0:
                state.candidate_label = "UNKNOWN"
                state.classification_phase = "UNKNOWN"

        if state.candidate_label == "REF":
            required_frames = config.referee_candidate_frames
            required_weight = 6.0
            candidate_blocked = False
        else:
            required_frames = config.team_candidate_frames
            required_weight = 2.5
            early_referee_ambiguity = (
                state.referee_observations >= 4
                and state.scores["REF"]
                >= 0.90 * max(state.scores["RAIDERS"], state.scores["BRONCOS"])
            )
            if early_referee_ambiguity:
                # Jersey numbers, white sleeves and partial side views can
                # briefly resemble stripes. Keep the track UNKNOWN until the
                # texture evidence either persists or is disproved.
                required_frames = max(required_frames, config.referee_candidate_frames)
                required_weight = max(required_weight, 6.0)
            candidate_blocked = early_referee_ambiguity

        if (
            state.candidate_label in TEAM_NAMES
            and not candidate_blocked
            and state.candidate_frames >= required_frames
            and state.candidate_weight >= required_weight
        ):
            state.established_label = state.candidate_label
            state.classification_phase = "ESTABLISHED"
            state.candidate_label = "UNKNOWN"
            state.candidate_frames = 0
            state.candidate_weight = 0.0
            state.label = state.established_label
            return state.label, confidence

        archived_team = max(
            ("RAIDERS", "BRONCOS"),
            key=lambda name: state.identity_scores[name],
        )
        archived_other_team = (
            "BRONCOS" if archived_team == "RAIDERS" else "RAIDERS"
        )
        archived_team_ready = (
            state.candidate_label == archived_team
            and state.candidate_frames >= 5
            and state.candidate_weight >= 0.75
            and state.identity_scores[archived_team] >= 0.52
            and state.identity_support_frames[archived_team] >= 15
            and state.identity_scores[archived_team]
            >= state.identity_scores[archived_other_team] + 0.35
            and state.identity_scores[archived_team]
            >= state.identity_scores["REF"] + 0.35
        )
        if archived_team_ready:
            state.established_label = archived_team
            state.classification_phase = "ESTABLISHED"
            state.candidate_label = "UNKNOWN"
            state.candidate_frames = 0
            state.candidate_weight = 0.0
            state.label = state.established_label
            belief_total = sum(eligible_scores.values()) + 1e-6
            return state.label, eligible_scores[state.label] / belief_total

        state.label = "UNKNOWN"
        return state.label, confidence if proposed in TEAM_NAMES else 0.0

    # Established identities only consider a challenger when the current crop
    # is reliable and the aggregate evidence has a decisive different winner.
    if reliable_observation and proposed in TEAM_NAMES:
        if proposed == state.established_label:
            state.challenger_label = "UNKNOWN"
            state.challenger_frames = 0
            state.challenger_weight = 0.0
        elif proposed == state.challenger_label:
            state.challenger_frames += 1
            state.challenger_weight += features.vote_weight
        else:
            state.challenger_label = proposed
            state.challenger_frames = 1
            state.challenger_weight = features.vote_weight
    elif reliable_observation:
        state.challenger_label = "UNKNOWN"
        state.challenger_frames = 0
        state.challenger_weight = 0.0

    if state.challenger_label in TEAM_NAMES:
        if state.established_label == "REF":
            required_frames = config.referee_to_team_frames
            required_weight = 7.0
            required_confidence = 0.60
            required_margin = 0.18
        elif state.challenger_label == "REF":
            required_frames = config.team_to_referee_frames
            required_weight = 12.0
            required_confidence = 0.62
            required_margin = 0.20
        else:
            required_frames = config.team_switch_frames
            required_weight = 7.0
            required_confidence = 0.58
            required_margin = 0.16

        can_switch = (
            state.challenger_frames >= required_frames
            and state.challenger_weight >= required_weight
            and confidence >= required_confidence
            and margin >= required_margin
        )
        if state.challenger_label == "REF":
            can_switch = can_switch and (
                state.referee_texture >= 0.44
                and state.referee_observations >= 8
                and state.identity_scores["REF"]
                >= state.identity_scores[state.established_label] + 1.50
                and not features.rules_out_referee
            )
        elif state.established_label == "REF":
            can_switch = can_switch and (
                state.identity_scores[state.challenger_label]
                >= state.identity_scores["REF"] + 1.00
            )
        if can_switch:
            old_label = state.established_label
            state.established_label = state.challenger_label
            state.scores[old_label] *= 0.45
            state.challenger_label = "UNKNOWN"
            state.challenger_frames = 0
            state.challenger_weight = 0.0

    state.label = state.established_label
    belief_total = sum(eligible_scores.values()) + 1e-6
    displayed_confidence = eligible_scores.get(state.label, 0.0) / belief_total
    return state.label, displayed_confidence


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


def draw_movement(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    state: MovementState,
    camera: CameraMotion,
    label: str,
) -> None:
    if (
        label not in ("RAIDERS", "BRONCOS")
        or not camera.available
        or state.segment_id != camera.segment_id
        or len(state.reference_history) < 2
        or state.steps_by_segment.get(camera.segment_id, 0) < 5
    ):
        return
    try:
        reference_to_current = np.linalg.inv(camera.current_to_reference)
    except np.linalg.LinAlgError:
        return
    reference_points = np.asarray(
        [(x, y) for _, x, y in state.reference_history], dtype=np.float32
    ).reshape(-1, 1, 2)
    current_points = cv2.perspectiveTransform(
        reference_points, reference_to_current
    ).reshape(-1, 2)
    finite = np.all(np.isfinite(current_points), axis=1)
    current_points = current_points[finite]
    if len(current_points) < 2:
        return
    points = np.rint(current_points).astype(np.int32).reshape(-1, 1, 2)
    color = (210, 210, 210) if label == "RAIDERS" else (0, 115, 255)
    cv2.polylines(frame, [points], False, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)
    foot = tuple(points[-1, 0].tolist())
    cv2.circle(frame, foot, 4, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(frame, foot, 2, color, -1, cv2.LINE_AA)



def draw_movement_dashboard(
    frame: np.ndarray,
    rows: list[tuple[int, str, float, float]],
) -> None:
    """Show a compact summary instead of adding text beside every player."""
    if not rows:
        return
    visible_rows = sorted(rows, key=lambda row: row[3], reverse=True)[:3]
    panel_width = 282
    panel_height = 27 + 19 * len(visible_rows)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0.0, frame)
    cv2.putText(
        frame,
        "CAMERA-STABILIZED MOVEMENT (PIXELS)",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    for index, (track_id, label, distance, speed) in enumerate(visible_rows):
        team = "LV" if label == "RAIDERS" else "DEN"
        color = (210, 210, 210) if label == "RAIDERS" else (0, 140, 255)
        cv2.putText(
            frame,
            f"ID {track_id:>4} {team}   D {distance:>4.0f}px   V {speed:>4.0f}px/s",
            (15, 44 + 19 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_label(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    track_id: int,
    label: str,
    confidence: float,
) -> None:
    palette = {
        "RAIDERS": ((35, 35, 35), (255, 255, 255)),
        "BRONCOS": ((245, 245, 245), (15, 15, 15)),
        "REF": ((0, 215, 255), (0, 0, 0)),
        "NON-PLAYER": ((95, 95, 95), (255, 255, 255)),
        "UNKNOWN": ((110, 80, 40), (255, 255, 255)),
    }
    box_color, text_color = palette[label]
    x1, y1, x2, y2 = box
    thickness = 1 if label == "NON-PLAYER" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    text = f"ID:{track_id} {label} {confidence:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43 if label == "NON-PLAYER" else 0.50
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
    label_top = max(0, y1 - text_height - baseline - 7)
    cv2.rectangle(
        frame,
        (x1, label_top),
        (min(frame.shape[1] - 1, x1 + text_width + 8), y1),
        box_color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 4, max(text_height + 1, y1 - baseline - 3)),
        font,
        scale,
        text_color,
        1,
        cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    model = YOLO(args.model)
    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise SystemExit(f"Could not open input video: {input_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    output = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not output.isOpened():
        video.release()
        raise SystemExit(f"Could not create output video: {output_path}")

    track_states: dict[int, TrackState] = defaultdict(TrackState)
    movement_states: dict[int, MovementState] = defaultdict(MovementState)
    track_labels: dict[int, str] = {}
    camera_estimator = CameraMotionEstimator()
    camera_available_frames = 0
    camera_reset_frames = 0
    camera_feature_total = 0
    camera_inlier_total = 0.0
    camera_estimated_frames = 0
    label_counts: Counter[str] = Counter()
    frame_number = 0
    started = time.perf_counter()
    print(
        f"Input: {input_path} ({expected_frames} frames, {width}x{height} @ {fps:.3f} fps)"
    )
    print(f"Detector device: {device}; output: {output_path}")

    try:
        while True:
            success, frame = video.read()
            if not success:
                break
            frame_number += 1

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=config.confidence,
                iou=0.50,
                imgsz=config.image_size,
                max_det=config.max_detections,
                device=device,
                verbose=False,
            )
            boxes = results[0].boxes
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            field = build_field_context(hsv_frame)
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
                    (
                        max(0, x1),
                        max(0, y1),
                        min(width - 1, x2),
                        min(height - 1, y2),
                    )
                )
            camera = camera_estimator.update(gray_frame, field, clipped_boxes)
            if camera.available:
                camera_available_frames += 1
            if camera.reset:
                camera_reset_frames += 1
            if camera.tracked_features:
                camera_estimated_frames += 1
                camera_feature_total += camera.tracked_features
                camera_inlier_total += camera.inlier_ratio

            crowded_nonfield_scene = not field.has_field and len(coordinates) >= 5
            active_movement_rows: list[tuple[int, str, float, float]] = []
            for box, track_id in zip(clipped_boxes, track_ids):
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                features = analyze_detection(
                    hsv_frame,
                    field,
                    box,
                    config,
                    crowded_nonfield_scene,
                )
                label, confidence = update_track(
                    track_states[track_id], features, frame_number, config
                )
                track_labels[int(track_id)] = label
                label_counts[label] += 1

                # Movement is measured only on/near the calibrated field plane.
                # We retain it from the first plausible player observation so
                # an eventual established label can show the earlier trail.
                box_height = y2 - y1
                box_width = x2 - x1
                standing_ground_geometry = (
                    box_height / max(1, box_width) >= 1.15
                    and y2 < height - 3
                )
                movement_eligible = (
                    camera.available
                    and features.field_support >= 0.06
                    and standing_ground_geometry
                    and (
                        features.on_field
                        or features.player_evidence >= 0.50
                        or track_states[track_id].classification_phase
                        == "ESTABLISHED"
                    )
                )
                if movement_eligible:
                    foot_point = ((x1 + x2) * 0.5, float(y2))
                    update_movement(
                        movement_states[track_id],
                        camera,
                        foot_point,
                        box_height,
                        frame_number,
                        fps,
                        config,
                    )
                if label != "NON-PLAYER" or not args.hide_non_players:
                    if movement_eligible:
                        draw_movement(
                            frame,
                            box,
                            movement_states[track_id],
                            camera,
                            label,
                        )
                    draw_label(frame, box, track_id, label, confidence)
                movement_state = movement_states[track_id]
                if (
                    movement_eligible
                    and label in ("RAIDERS", "BRONCOS")
                    and movement_state.segment_id == camera.segment_id
                    and movement_state.steps_by_segment.get(camera.segment_id, 0) >= 5
                ):
                    active_movement_rows.append(
                        (
                            int(track_id),
                            label,
                            movement_state.distance_by_segment.get(
                                camera.segment_id, 0.0
                            ),
                            movement_state.current_speed_px_s,
                        )
                    )

            draw_movement_dashboard(frame, active_movement_rows)

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
            cv2.putText(
                frame,
                motion_status,
                (12, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                motion_status,
                (12, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (235, 235, 235) if camera.available else (80, 180, 255),
                1,
                cv2.LINE_AA,
            )

            if frame_number % config.stale_track_frames == 0:
                stale_before = frame_number - config.stale_track_frames
                track_states = defaultdict(
                    TrackState,
                    {
                        track_id: state
                        for track_id, state in track_states.items()
                        if state.last_seen >= stale_before
                    },
                )

            output.write(frame)
            if frame_number % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f"Processed frame {frame_number} ({frame_number / elapsed:.1f} fps)")
            if args.max_frames and frame_number >= args.max_frames:
                break
    finally:
        video.release()
        output.release()

    elapsed = time.perf_counter() - started
    print("DONE!")
    print(f"Processed {frame_number} frames in {elapsed:.1f}s ({frame_number / elapsed:.1f} fps)")
    print(f"Detection labels: {dict(label_counts)}")
    print(
        "Camera stabilization: "
        f"{camera_available_frames}/{frame_number} frames available; "
        f"{camera_estimated_frames} homographies; {camera_reset_frames} reference resets"
    )
    if camera_estimated_frames:
        print(
            "Camera matches: "
            f"{camera_feature_total / camera_estimated_frames:.1f} features/frame; "
            f"{camera_inlier_total / camera_estimated_frames:.1%} mean inliers"
        )
    if args.movement_metrics:
        movement_metrics_path = Path(args.movement_metrics)
        movement_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        track_metrics: dict[str, object] = {}
        for track_id, movement in movement_states.items():
            if movement.observations < 2:
                continue
            segment_metrics = []
            for segment_id, distance in movement.distance_by_segment.items():
                frame_range = movement.frame_range_by_segment.get(segment_id, [0, 0])
                elapsed_seconds = max(
                    0.0, (frame_range[1] - frame_range[0]) / max(fps, 1e-6)
                )
                segment_metrics.append(
                    {
                        "segment_id": segment_id,
                        "first_frame": frame_range[0],
                        "last_frame": frame_range[1],
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "distance_px": round(distance, 3),
                        "accepted_steps": movement.steps_by_segment.get(
                            segment_id, 0
                        ),
                        "mean_speed_px_s": round(
                            distance / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                            3,
                        ),
                    }
                )
            track_metrics[str(track_id)] = {
                "last_label": track_labels.get(track_id, "UNKNOWN"),
                "observations": movement.observations,
                "accepted_steps": movement.accepted_steps,
                "interruptions": movement.interrupted_runs,
                "rejected_jumps": movement.rejected_jumps,
                "segments": segment_metrics,
            }
        metrics = {
            "coordinate_system": (
                "shot-local reference-image pixels after field homography; "
                "distances are not yards and are not comparable across segments"
            ),
            "fps": fps,
            "frames": frame_number,
            "camera": {
                "available_frames": camera_available_frames,
                "estimated_homographies": camera_estimated_frames,
                "reference_resets": camera_reset_frames,
                "mean_features": round(
                    camera_feature_total / max(1, camera_estimated_frames), 3
                ),
                "mean_inlier_ratio": round(
                    camera_inlier_total / max(1, camera_estimated_frames), 4
                ),
            },
            "tracks": track_metrics,
        }
        movement_metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Movement metrics: {movement_metrics_path}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run(parse_args())
