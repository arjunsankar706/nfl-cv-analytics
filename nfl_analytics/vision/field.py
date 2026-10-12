"""Connected playing-surface segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldContext:
    mask: np.ndarray
    boundary: np.ndarray
    component_ratio: float
    has_field: bool


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


def clipped_region(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    height, width = image.shape[:2]
    return image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]


def ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0
