"""Register a broadcast frame to field coordinates (yards) from painted markings.

Coordinate convention: ``d`` is the downfield axis (perpendicular to yard
lines), ``l`` the lateral axis (parallel to yard lines). Both are in yards
with an arbitrary origin; the formation analyzer re-centres them on the ball.

Calibration tiers, best first:

``HOMOGRAPHY``  >=2 yard lines (5 yd apart) and both hash-mark rows
                (6.17 yd apart): a full image->field homography.
``YARD_LINES``  >=2 yard lines but no usable hash pair. Downfield distance
                follows an exact 1D projective map through the yard-line
                vanishing point; lateral distance is only known up to scale
                (callers rescale it and must treat it as approximate).
``NONE``        No reliable markings. The caller must fall back to an
                uncalibrated estimate and treat depth thresholds as unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

YARD_LINE_SPACING = 5.0
HASH_SEPARATION = 18.5 / 3.0  # NFL hash marks are 18 ft 6 in apart
TIER_HOMOGRAPHY = "HOMOGRAPHY"
TIER_YARD_LINES = "YARD_LINES"
TIER_NONE = "NONE"

Box = tuple[int, int, int, int]


@dataclass
class Line:
    """Image line a*x + b*y + c = 0 with (a, b) unit length."""

    a: float
    b: float
    c: float
    length: float = 0.0
    points: Optional[np.ndarray] = None

    @classmethod
    def through(cls, p: np.ndarray, q: np.ndarray, length: float = 0.0) -> "Line":
        direction = q - p
        normal = np.array([-direction[1], direction[0]], dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            raise ValueError("degenerate line")
        normal /= norm
        return cls(float(normal[0]), float(normal[1]), float(-normal @ p), length)

    @classmethod
    def fit(cls, points: np.ndarray, weights: Optional[np.ndarray] = None) -> "Line":
        points = np.asarray(points, dtype=float)
        weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
        centre = np.average(points, axis=0, weights=weights)
        centred = (points - centre) * np.sqrt(weights)[:, None]
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        return cls(float(normal[0]), float(normal[1]), float(-normal @ centre), points=points)

    @property
    def homogeneous(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c])

    @property
    def direction(self) -> np.ndarray:
        return np.array([self.b, -self.a])

    def angle(self) -> float:
        """Undirected angle of the line in [0, pi)."""
        return float(np.arctan2(self.direction[1], self.direction[0]) % np.pi)

    def distance(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        return points @ np.array([self.a, self.b]) + self.c


def intersect(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Homogeneous intersection of two homogeneous lines."""
    return np.cross(first, second)


@dataclass
class FieldRegistration:
    tier: str
    image_size: tuple[int, int]
    yard_lines: list[Line] = field(default_factory=list)
    yard_indices: list[int] = field(default_factory=list)
    hash_rows: list[Line] = field(default_factory=list)
    homography: Optional[np.ndarray] = None  # image -> field (d, l)
    vanishing_point: Optional[np.ndarray] = None  # homogeneous
    transversal: Optional[np.ndarray] = None  # homogeneous line
    projective_1d: Optional[np.ndarray] = None  # s -> d coefficients (a, b, c): d=(a s + b)/(c s + 1)
    lateral_ratio: Optional[float] = None  # lateral px/yd divided by downfield px/yd
    reprojection_error_yd: Optional[float] = None
    family_direction: Optional[np.ndarray] = None  # image direction of yard lines, if seen at all
    notes: list[str] = field(default_factory=list)

    @property
    def lateral_calibrated(self) -> bool:
        return self.tier == TIER_HOMOGRAPHY

    @property
    def calibrated(self) -> bool:
        return self.tier in (TIER_HOMOGRAPHY, TIER_YARD_LINES)

    @property
    def depth_uncertainty_yd(self) -> float:
        """Rough 1-sigma uncertainty used to widen classification margins."""
        if self.tier == TIER_HOMOGRAPHY:
            return 0.35
        if self.tier == TIER_YARD_LINES:
            return 0.5
        return 1.5

    # ------------------------------------------------------------------ mapping
    def to_field(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if self.tier == TIER_HOMOGRAPHY and self.homography is not None:
            return cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.homography).reshape(-1, 2)
        if self.tier == TIER_YARD_LINES:
            return np.array([self._yard_line_point(point) for point in points])
        raise ValueError("frame is not calibrated")

    def _pencil_parameter(self, point: np.ndarray) -> float:
        line_through = np.cross(self.vanishing_point, np.array([point[0], point[1], 1.0]))
        hit = intersect(line_through, self.transversal)
        return _transversal_coordinate(hit, self.transversal)

    def _downfield(self, s: float) -> float:
        a, b, c = self.projective_1d
        return float((a * s + b) / (c * s + 1.0))

    def _yard_line_point(self, point: np.ndarray) -> np.ndarray:
        s = self._pencil_parameter(point)
        d = self._downfield(s)
        # Lateral: arc length along the yard line through the point, measured
        # from the transversal and converted with the local downfield scale.
        line_through = np.cross(self.vanishing_point, np.array([point[0], point[1], 1.0]))
        foot = intersect(line_through, self.transversal)
        foot = foot[:2] / foot[2] if abs(foot[2]) > 1e-12 else np.array(point, dtype=float)
        direction = np.array([line_through[1], -line_through[0]], dtype=float)
        direction /= np.linalg.norm(direction) + 1e-12
        if direction[1] < 0:
            direction = -direction
        along = float((np.asarray(point, dtype=float) - foot) @ direction)
        # Local downfield scale (pixels per yard) at this point.
        epsilon = 1.0
        normal = np.array([direction[1], -direction[0]])
        d_plus = self._downfield(self._pencil_parameter(np.asarray(point) + epsilon * normal))
        pixels_per_yard = epsilon / max(abs(d_plus - d), 1e-6)
        lateral_scale = pixels_per_yard * (self.lateral_ratio or 1.0)
        return np.array([d, along / lateral_scale])

    def local_pixels_per_yard(self, point: Sequence[float]) -> Optional[float]:
        """Downfield pixels per yard at an image point (None when uncalibrated)."""
        if not self.calibrated:
            return None
        point = np.asarray(point, dtype=float)
        base = self.to_field(point)[0]
        best = None
        for offset in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            moved = self.to_field(point + 4.0 * offset)[0]
            yards = float(np.linalg.norm(moved - base))
            if yards > 1e-6:
                best = max(best or 0.0, 4.0 / yards)
        return best


def _transversal_coordinate(hit: np.ndarray, transversal: np.ndarray) -> float:
    if abs(hit[2]) < 1e-12:
        return float("inf")
    point = hit[:2] / hit[2]
    direction = np.array([transversal[1], -transversal[0]], dtype=float)
    direction /= np.linalg.norm(direction) + 1e-12
    return float(point @ direction)


# ---------------------------------------------------------------- detection

def white_marking_mask(frame: np.ndarray, field_mask: np.ndarray, person_boxes: Sequence[Box]) -> np.ndarray:
    """Thin bright paint on the turf, with people, graphics margins removed."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    bright = (tophat > 28) & (hsv[..., 1] < 90) & (hsv[..., 2] > 135)
    support = cv2.dilate(field_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    mask = (bright & (support > 0)).astype(np.uint8) * 255
    for x1, y1, x2, y2 in person_boxes:
        pad = max(3, int(0.08 * (x2 - x1)))
        cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)), (min(width - 1, x2 + pad), min(height - 1, y2 + pad)), 0, -1)
    return mask


def hash_mark_mask(frame: np.ndarray, person_boxes: Sequence[Box]) -> np.ndarray:
    """A more sensitive paint mask for small, compression-blurred hash dashes."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = ((tophat > 14) & (hsv[..., 2] > 110) & (hsv[..., 1] < 120)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    height, width = mask.shape
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(mask, (max(0, x1 - 3), max(0, y1 - 3)), (min(width - 1, x2 + 3), min(height - 1, y2 + 3)), 0, -1)
    return mask


def detect_segments(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    diagonal = float(np.hypot(height, width))
    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360,
        threshold=40,
        minLineLength=int(0.07 * diagonal),
        maxLineGap=int(0.02 * diagonal),
    )
    if segments is None:
        return np.empty((0, 4), dtype=float)
    return segments.reshape(-1, 4).astype(float)


def turf_flanked(segments: np.ndarray, field_mask: np.ndarray, offset: float = 7.0) -> np.ndarray:
    """True for segments with turf on both sides (paint), not graphic edges."""
    height, width = field_mask.shape
    keep = np.zeros(len(segments), dtype=bool)
    for index, (x1, y1, x2, y2) in enumerate(segments):
        direction = np.array([x2 - x1, y2 - y1], dtype=float)
        length = np.linalg.norm(direction)
        if length < 1:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        samples = np.linspace(0.1, 0.9, 9)[:, None] * direction + np.array([x1, y1])
        hits = 0
        for side in (offset, -offset):
            points = np.rint(samples + side * normal).astype(int)
            inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
            values = np.zeros(len(points), dtype=bool)
            values[inside] = field_mask[points[inside, 1], points[inside, 0]] > 0
            hits += values.mean() >= 0.6
        keep[index] = hits == 2
    return keep


def _segment_lines(segments: np.ndarray) -> np.ndarray:
    ones = np.ones((len(segments), 1))
    first = np.hstack([segments[:, :2], ones])
    second = np.hstack([segments[:, 2:], ones])
    lines = np.cross(first, second)
    return lines / (np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12)


def _consistent_with_vp(segments: np.ndarray, vp: np.ndarray, max_angle: float) -> np.ndarray:
    midpoints = (segments[:, :2] + segments[:, 2:]) / 2.0
    directions = segments[:, 2:] - segments[:, :2]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    if abs(vp[2]) < 1e-9 * np.linalg.norm(vp[:2]) or np.linalg.norm(vp[:2] / (vp[2] + 1e-300)) > 1e7:
        towards = np.tile(vp[:2] / (np.linalg.norm(vp[:2]) + 1e-12), (len(segments), 1))
    else:
        towards = vp[:2] / vp[2] - midpoints
        towards /= np.linalg.norm(towards, axis=1, keepdims=True) + 1e-12
    cosine = np.abs(np.sum(directions * towards, axis=1))
    return cosine >= np.cos(max_angle)


def merge_family(
    segments: np.ndarray,
    vp: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[list[Line], np.ndarray]:
    """Merge segments of one family into lines; returns lines and transversal."""
    width, height = image_size
    centre = np.array([width / 2.0, height / 2.0])
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    directions = (segments[:, 2:] - segments[:, :2]) / (lengths[:, None] + 1e-12)
    reference = directions[np.argmax(lengths)]
    signs = np.sign(directions @ reference)
    signs[signs == 0] = 1
    mean_direction = np.sum(directions * (signs * lengths)[:, None], axis=0)
    mean_direction /= np.linalg.norm(mean_direction) + 1e-12
    transversal = np.array([mean_direction[0], mean_direction[1], -mean_direction @ centre])
    lines = _segment_lines(segments)
    positions = np.array(
        [_transversal_coordinate(intersect(line, transversal), transversal) for line in lines]
    )
    order = np.argsort(positions)
    tolerance = max(6.0, 0.010 * np.hypot(width, height))
    groups: list[list[int]] = []
    for index in order:
        if not np.isfinite(positions[index]):
            continue
        if groups and abs(positions[index] - np.median(positions[groups[-1]])) < tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
    result = []
    minimum_span = 0.12 * np.hypot(width, height)
    for members in groups:
        points = np.concatenate([segments[members][:, :2], segments[members][:, 2:]])
        span = _span_along(points, mean_direction)
        if span < minimum_span:
            continue
        line = Line.fit(points, np.repeat(lengths[members], 2))
        if np.abs(line.distance(points)).max() > 6.0:
            continue
        line.length = span
        result.append(line)
    return result, transversal


def find_line_families(
    segments: np.ndarray,
    image_size: tuple[int, int],
    iterations: int = 600,
    max_angle: float = np.radians(1.5),
) -> list[np.ndarray]:
    """RANSAC vanishing points; returns inlier masks of distinct families, best first."""
    if len(segments) < 2:
        return []
    rng = np.random.default_rng(0)
    lines = _segment_lines(segments)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    available = np.ones(len(segments), dtype=bool)
    families = []
    for _ in range(3):
        candidates = np.flatnonzero(available)
        if len(candidates) < 2:
            break
        best = None
        best_score = 0.0
        for _ in range(iterations):
            i, j = rng.choice(candidates, 2, replace=False)
            vp = np.cross(lines[i], lines[j])
            if np.linalg.norm(vp) < 1e-12:
                continue
            inliers = available & _consistent_with_vp(segments, vp, max_angle)
            score = float(np.sqrt(lengths[inliers]).sum())
            if score > best_score:
                best, best_score = inliers, score
        if best is None or best.sum() < 2:
            break
        families.append(best)
        available &= ~best
    return families


def _span_along(points: np.ndarray, direction: np.ndarray) -> float:
    projection = points @ direction
    return float(projection.max() - projection.min())


def vanishing_point(lines: Sequence[Line]) -> np.ndarray:
    """Least-squares common point of a line family (homogeneous, may be at infinity)."""
    matrix = np.array([line.homogeneous * np.sqrt(max(line.length, 1.0)) for line in lines])
    _, _, vt = np.linalg.svd(matrix)
    point = vt[-1]
    return point / np.linalg.norm(point)


def assign_yard_indices(positions: np.ndarray) -> Optional[list[int]]:
    """Integer 5-yard indices for sorted line positions, tolerating missed lines.

    Perspective makes spacing change smoothly, so each gap is compared to its
    neighbours; a gap of ~2x (or ~3x) the local spacing means skipped lines.
    """
    if len(positions) < 2:
        return None
    gaps = np.diff(positions)
    if np.any(gaps <= 0):
        return None
    base = np.min(gaps)
    indices = [0]
    for gap in gaps:
        multiple = gap / base
        rounded = int(round(multiple))
        if rounded < 1 or abs(multiple - rounded) > 0.32 * rounded or rounded > 3:
            return None
        indices.append(indices[-1] + rounded)
    return indices


def fit_projective_1d(s: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
    """Fit d = (a s + b) / (c s + 1); falls back to affine for two points."""
    s = np.asarray(s, dtype=float)
    d = np.asarray(d, dtype=float)
    if len(s) == 2:
        a = (d[1] - d[0]) / (s[1] - s[0])
        return np.array([a, d[0] - a * s[0], 0.0])
    scale = np.max(np.abs(s)) or 1.0
    t = s / scale
    matrix = np.column_stack([t, np.ones_like(t), -t * d])
    solution, *_ = np.linalg.lstsq(matrix, d, rcond=None)
    a, b, c = solution
    # Reject fits whose pole falls inside the observed range (non-monotone).
    if abs(c) > 1e-9:
        pole = -1.0 / c
        if t.min() - 0.5 <= pole <= t.max() + 0.5:
            return None
    return np.array([a / scale, b, c / scale])


def detect_hash_marks(
    mask: np.ndarray,
    yard_lines: Sequence[Line],
    field_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Centres of short paint dashes surrounded by turf and off the yard lines."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    scale = np.hypot(height, width)
    centres = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        extent = max(w, h)
        if not (8 <= area <= 0.0005 * height * width and 3 <= extent <= 0.035 * scale):
            continue
        centre = centroids[index]
        if yard_lines and min(abs(line.distance(centre)[0]) for line in yard_lines) < 4.0:
            continue
        if field_mask is not None:
            reach = int(max(8, 0.9 * extent))
            probes = [
                (int(centre[0]) + dx, int(centre[1]) + dy)
                for dx, dy in ((reach, 0), (-reach, 0), (0, reach), (0, -reach))
            ]
            if not all(0 <= x < width and 0 <= y < height and field_mask[y, x] > 0 for x, y in probes):
                continue
        centres.append(centre)
    return np.array(centres, dtype=float).reshape(-1, 2)


def fit_hash_rows(
    centres: np.ndarray,
    vanishing: np.ndarray,
    image_size: tuple[int, int],
    minimum_marks: int = 6,
) -> list[Line]:
    """RANSAC rows of dashes running across the yard lines (one mark per yard)."""
    rows: list[Line] = []
    remaining = centres.copy()
    rng = np.random.default_rng(0)
    width, height = image_size
    vp_direction = None
    if abs(vanishing[2]) < 1e-6:
        vp_direction = vanishing[:2] / (np.linalg.norm(vanishing[:2]) + 1e-12)
    for _ in range(8):
        if len(remaining) < minimum_marks:
            break
        best_inliers = None
        for _ in range(400):
            first, second = remaining[rng.choice(len(remaining), 2, replace=False)]
            if np.linalg.norm(second - first) < 0.05 * width:
                continue
            try:
                candidate = Line.through(first, second)
            except ValueError:
                continue
            # A hash row must cross the yard-line family, not run along it.
            if vp_direction is not None:
                if abs(candidate.direction @ vp_direction) > 0.8:
                    continue
            else:
                to_vp = vanishing[:2] / vanishing[2] - first
                to_vp /= np.linalg.norm(to_vp) + 1e-12
                if abs(candidate.direction @ to_vp) > 0.8:
                    continue
            inliers = np.abs(candidate.distance(remaining)) < 3.0
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
        if best_inliers is None or best_inliers.sum() < minimum_marks:
            break
        row = Line.fit(remaining[best_inliers])
        inliers = np.abs(row.distance(remaining)) < 3.5
        row = Line.fit(remaining[inliers])
        row.length = _span_along(remaining[inliers], row.direction)
        if row.length > 0.15 * width:
            rows.append(row)
        remaining = remaining[~inliers]
    return rows


def register_field(
    frame: np.ndarray,
    field_mask: np.ndarray,
    person_boxes: Sequence[Box],
    ball_hint: Optional[np.ndarray] = None,
) -> FieldRegistration:
    height, width = frame.shape[:2]
    image_size = (width, height)
    registration = FieldRegistration(TIER_NONE, image_size)
    mask = white_marking_mask(frame, field_mask, person_boxes)
    segments = detect_segments(mask)
    if len(segments):
        segments = segments[turf_flanked(segments, field_mask)]
    best = None
    for family in find_line_families(segments, image_size):
        lines, transversal = merge_family(segments[family], np.zeros(3), image_size)
        if lines and registration.family_direction is None:
            registration.family_direction = np.array([transversal[0], transversal[1]])
        if len(lines) < 2:
            continue
        positions = np.array(
            [_transversal_coordinate(intersect(line.homogeneous, transversal), transversal) for line in lines]
        )
        order = np.argsort(positions)
        lines = [lines[index] for index in order]
        positions = positions[order]
        indices = assign_yard_indices(positions)
        if indices is None:
            continue
        projective = fit_projective_1d(positions, YARD_LINE_SPACING * np.array(indices))
        if projective is None:
            continue
        if best is None or len(lines) > len(best[0]):
            best = (lines, indices, transversal, projective)
    if best is None:
        registration.notes.append("no line family with consistent 5-yard spacing")
        return registration
    lines, indices, transversal, projective = best
    vp = vanishing_point(lines)

    registration.tier = TIER_YARD_LINES
    registration.yard_lines = lines
    registration.yard_indices = indices
    registration.vanishing_point = vp
    registration.transversal = transversal
    registration.projective_1d = projective
    # Without a hash pair the lateral scale is unknown (camera tilt cannot be
    # recovered reliably from one line family); downstream code rescales
    # lateral distances and labels them approximate.
    registration.lateral_ratio = 1.0
    registration.notes.append(f"{len(lines)} yard lines")

    centres = detect_hash_marks(hash_mark_mask(frame, person_boxes), lines, field_mask)
    rows = fit_hash_rows(centres, vp, image_size)
    hash_pair = choose_hash_pair(rows, registration, ball_hint, mask)
    if hash_pair is None:
        registration.notes.append(f"{len(rows)} hash/tick row(s); no consistent hash pair")
        return registration
    homography, error = homography_from_markings(lines, indices, hash_pair)
    if homography is None:
        registration.notes.append("homography from markings was ill-conditioned")
        return registration
    registration.hash_rows = list(hash_pair)
    registration.homography = homography
    registration.reprojection_error_yd = error
    registration.tier = TIER_HOMOGRAPHY
    registration.notes.append(f"homography from {len(lines)} yard lines x 2 hash rows (rms {error:.2f} yd)")
    return registration


def row_is_yard_regular(row: Line, registration: FieldRegistration, minimum_marks: int = 6) -> bool:
    """Hash and sideline ticks sit exactly one yard apart; text and logos do not."""
    if row.points is None or len(row.points) < minimum_marks:
        return False
    downfield = np.array([registration._downfield(registration._pencil_parameter(p)) for p in row.points])
    phase_angle = np.angle(np.mean(np.exp(2j * np.pi * downfield)))
    residual = np.abs(((downfield - phase_angle / (2 * np.pi)) + 0.5) % 1.0 - 0.5)
    on_grid = residual < 0.22
    if on_grid.sum() < minimum_marks or on_grid.mean() < 0.7:
        return False
    gaps = np.diff(np.sort(downfield[on_grid]))
    gaps = gaps[gaps > 0.5]
    return len(gaps) >= minimum_marks - 2 and float(np.median(gaps)) < 1.6


def row_beside_solid_line(row: Line, paint: np.ndarray) -> bool:
    """Sideline tick rows run right next to the solid sideline; hash rows do not."""
    if row.points is None or len(row.points) < 2:
        return False
    height, width = paint.shape
    direction = row.direction
    projection = row.points @ direction
    start = row.points[np.argmin(projection)]
    end = row.points[np.argmax(projection)]
    samples = np.linspace(0.0, 1.0, 80)[:, None] * (end - start) + start
    normal = np.array([row.a, row.b])
    for offset in range(-18, 19, 2):
        if abs(offset) < 4:
            continue
        points = np.rint(samples + offset * normal).astype(int)
        inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        if inside.sum() < 40:
            continue
        values = paint[points[inside, 1], points[inside, 0]] > 0
        if values.mean() > 0.5:
            return True
    return False


def choose_hash_pair(
    rows: Sequence[Line],
    registration: FieldRegistration,
    ball_hint: Optional[np.ndarray],
    paint: Optional[np.ndarray] = None,
) -> Optional[tuple[Line, Line]]:
    """Pick the two tick rows that are the hash marks.

    Only rows with one mark per yard qualify. Valid rows are ordered across
    the field; the ball is always on or between the hashes, so the adjacent
    pair bracketing the formation centre (or nearest to it) is chosen.
    """
    valid = [
        row
        for row in rows
        if row_is_yard_regular(row, registration) and (paint is None or not row_beside_solid_line(row, paint))
    ]
    if len(valid) < 2:
        return None
    middle = registration.yard_lines[len(registration.yard_lines) // 2]
    direction = middle.direction
    anchors = []
    for row in valid:
        hit = intersect(row.homogeneous, middle.homogeneous)
        if abs(hit[2]) < 1e-9:
            continue
        anchors.append((float((hit[:2] / hit[2]) @ direction), row))
    anchors.sort(key=lambda item: item[0])
    width, height = registration.image_size
    hint_position = None
    if ball_hint is not None:
        line_through = np.cross(registration.vanishing_point, np.array([ball_hint[0], ball_hint[1], 1.0]))
        projected = [intersect(row.homogeneous, line_through) for _, row in anchors]
        hint_position = float(np.asarray(ball_hint) @ direction)
        anchors = [
            (float((hit[:2] / hit[2]) @ direction), row) if abs(hit[2]) > 1e-9 else (position, row)
            for hit, (position, row) in zip(projected, anchors)
        ]
        anchors.sort(key=lambda item: item[0])
    candidates = []
    for (first_position, first), (second_position, second) in zip(anchors, anchors[1:]):
        separation = second_position - first_position
        if separation < 0.02 * height:
            continue  # duplicate detections of one row (e.g. sideline + tick)
        if hint_position is None:
            penalty = 0.0
        elif first_position - 0.25 * separation <= hint_position <= second_position + 0.25 * separation:
            penalty = 0.0
        else:
            penalty = min(abs(hint_position - first_position), abs(hint_position - second_position)) / separation
        candidates.append((penalty, separation, (first, second)))
    if not candidates:
        return None
    _, _, pair = min(candidates, key=lambda item: (item[0], item[1]))
    # The hint only ranks pairs: sideline rows are already excluded, and the
    # hint (median of everyone on the turf) can be pulled by sideline staff.
    return pair


def homography_from_markings(
    lines: Sequence[Line],
    indices: Sequence[int],
    hash_pair: tuple[Line, Line],
) -> tuple[Optional[np.ndarray], Optional[float]]:
    image_points = []
    field_points = []
    for line, index in zip(lines, indices):
        for lateral, row in ((0.0, hash_pair[0]), (HASH_SEPARATION, hash_pair[1])):
            hit = intersect(line.homogeneous, row.homogeneous)
            if abs(hit[2]) < 1e-9:
                continue
            image_points.append(hit[:2] / hit[2])
            field_points.append((YARD_LINE_SPACING * index, lateral))
    if len(image_points) < 4:
        return None, None
    image_points = np.array(image_points, dtype=np.float64)
    field_points = np.array(field_points, dtype=np.float64)
    homography, _ = cv2.findHomography(image_points, field_points, 0)
    if homography is None or not np.all(np.isfinite(homography)):
        return None, None
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - field_points) ** 2, axis=1))))
    # With exactly four points the fit is exact; still reject wild warps.
    if error > 0.6 or abs(np.linalg.det(homography[:2, :2])) < 1e-9:
        return None, None
    return homography, error
