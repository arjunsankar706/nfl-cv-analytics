"""OpenCV drawing helpers for the annotated output video."""

from __future__ import annotations

import cv2
import numpy as np

from .camera import CameraMotion
from .movement import MovementState


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
