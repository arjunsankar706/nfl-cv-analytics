"""Split on-field people into two teams by jersey colour (any matchup).

The Raiders/Broncos classifier in `nfl_analytics.vision.classification` is
tuned to one game. This module is team-agnostic: it clusters torso colours of
the people in one frame into two groups and removes officials using the
vertical-stripe texture measure from the original pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from nfl_analytics.vision.classification import analyze_uniform

Box = tuple[int, int, int, int]
TEAM_A = "A"
TEAM_B = "B"
OFFICIAL = "OFFICIAL"
UNASSIGNED = "UNASSIGNED"


@dataclass
class TeamAssignment:
    labels: list[str]
    colours_bgr: dict[str, tuple[int, int, int]]
    separation: float  # between-cluster distance / pooled within-cluster spread
    reliable: bool
    note: str


def jersey_feature(frame: np.ndarray, box: Box) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Torso descriptor (turf removed) and its median Lab colour.

    The descriptor is [% bright pixels, % dark pixels, 2*a, 2*b]. Bright/dark
    fractions separate white from dark jerseys even when shadows shift the
    median lightness; chroma separates coloured jerseys (e.g. red vs white).
    """
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    crop = frame[
        max(0, y1 + int(0.15 * height)) : max(0, y1 + int(0.52 * height)),
        max(0, x1 + int(0.22 * width)) : max(0, x2 - int(0.22 * width)),
    ]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    turf = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab = lab[~turf.reshape(-1)]
    if len(lab) < 8:
        return None
    median = np.median(lab, axis=0)
    descriptor = np.array(
        [
            100.0 * np.mean(lab[:, 0] > 165),
            100.0 * np.mean(lab[:, 0] < 85),
            2.0 * (median[1] - 128.0),
            2.0 * (median[2] - 128.0),
        ],
        dtype=np.float32,
    )
    return descriptor, median


def is_official(hsv_frame: np.ndarray, box: Box) -> bool:
    analysis = analyze_uniform(hsv_frame, box)
    return analysis.referee_texture >= 0.62 and analysis.light_pants_ratio < 0.25


def _kmeans_two(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    cv2.setRNGSeed(3)
    _, labels, centres = cv2.kmeans(
        features.astype(np.float32), 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS
    )
    return labels.ravel(), centres


def assign_teams(
    frame: np.ndarray,
    boxes: Sequence[Box],
    known_labels: Optional[Sequence[Optional[str]]] = None,
) -> TeamAssignment:
    """Return a team label per box.

    `known_labels` (e.g. RAIDERS/BRONCOS from the temporal tracker) are used
    when at least six people carry two distinct team names; colour clustering
    then only fills the gaps.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    labels = [UNASSIGNED] * len(boxes)
    features: dict[int, np.ndarray] = {}
    colours: dict[int, np.ndarray] = {}
    for index, box in enumerate(boxes):
        if is_official(hsv, box):
            labels[index] = OFFICIAL
            continue
        described = jersey_feature(frame, box)
        if described is not None:
            features[index], colours[index] = described

    if known_labels is not None:
        named = {
            index: label
            for index, label in enumerate(known_labels)
            if label not in (None, "UNKNOWN", "NON-PLAYER", "REF") and labels[index] != OFFICIAL
        }
        names = sorted(set(named.values()))
        if len(names) == 2 and len(named) >= 6:
            centres = {
                name: np.median([features[i] for i, label in named.items() if label == name and i in features], axis=0)
                for name in names
                if any(label == name and i in features for i, label in named.items())
            }
            lab_centres = {
                name: np.median([colours[i] for i, label in named.items() if label == name and i in colours], axis=0)
                for name in centres
            }
            for index in range(len(boxes)):
                if labels[index] == OFFICIAL:
                    continue
                if index in named:
                    labels[index] = named[index]
                elif index in features and len(centres) == 2:
                    labels[index] = min(centres, key=lambda name: np.linalg.norm(features[index] - centres[name]))
            team_colours = {name: _lab_to_bgr(centre) for name, centre in lab_centres.items()}
            return TeamAssignment(labels, team_colours, float("nan"), True, "team names from temporal tracker")

    if len(features) < 6:
        return TeamAssignment(labels, {}, 0.0, False, f"only {len(features)} usable jersey crops")
    indices = np.array(sorted(features))
    weighted = np.array([features[i] for i in indices])
    cluster, centres = _kmeans_two(weighted)
    within = [
        np.linalg.norm(weighted[cluster == k] - centres[k], axis=1) for k in (0, 1) if np.any(cluster == k)
    ]
    spread = float(np.median(np.concatenate(within))) + 1e-6
    separation = float(np.linalg.norm(centres[0] - centres[1])) / spread
    sizes = np.bincount(cluster, minlength=2)
    reliable = separation >= 2.5 and sizes.min() >= 3
    # Members far from their own centre are not trusted (sideline staff,
    # occluded players, merged boxes).
    distances = np.linalg.norm(weighted - centres[cluster], axis=1)
    cutoff = max(3.0 * spread, 25.0)
    # Name clusters so that A is the lighter jersey (stable across frames).
    order = np.argsort([-(centres[k][0] - centres[k][1]) for k in (0, 1)])
    name_of = {int(order[0]): TEAM_A, int(order[1]): TEAM_B}
    for index, k, distance in zip(indices, cluster, distances):
        labels[index] = name_of[int(k)] if distance <= cutoff else UNASSIGNED
    team_colours = {
        name_of[k]: _lab_to_bgr(np.median([colours[i] for i, c in zip(indices, cluster) if c == k], axis=0))
        for k in (0, 1)
    }
    note = f"colour clusters separated by {separation:.1f}x within-team spread"
    if not reliable:
        note += " (too similar to separate reliably)"
    return TeamAssignment(labels, team_colours, separation, reliable, note)


def _lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    pixel = np.clip(np.asarray(lab, dtype=np.float32), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bgr = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
