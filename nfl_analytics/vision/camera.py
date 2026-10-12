"""Shot-local camera-motion estimation from field features."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .field import FieldContext


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
