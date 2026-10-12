"""Render synthetic broadcast frames with known ground truth.

A field-to-image homography models a high sideline camera (perspective:
yard lines converge toward the top). Turf, yard lines every 5 yd, hash marks
every yard on both hash rows, and players (white vs dark jerseys) are drawn
at exact field coordinates, so formation analysis can be tested end to end
against known answers.

Field coordinates: d = downfield yards (offense attacks +d), l = lateral
yards (hash rows at l=0 and l=6.17; +l is toward the camera, which is the
offense's right when it faces +d).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HASH = 18.5 / 3.0
LOS = 20.0
BALL_L = 3.0
WIDTH, HEIGHT = 1280, 720

FIELD_CORNERS = np.float32([[0, -14], [40, -14], [40, 20], [0, 20]])
IMAGE_CORNERS = np.float32([[150, 60], [1130, 60], [1270, 700], [10, 700]])
FIELD_TO_IMAGE = cv2.getPerspectiveTransform(FIELD_CORNERS, IMAGE_CORNERS)

WHITE = (235, 235, 235)
DARK = (35, 30, 30)


@dataclass
class Spot:
    team: str  # "O" (white) or "D" (dark)
    d: float
    l: float
    role: str = ""


def project(d: float, l: float) -> np.ndarray:
    return cv2.perspectiveTransform(np.array([[[d, l]]], dtype=np.float64), FIELD_TO_IMAGE)[0, 0]


def render(spots: list[Spot], draw_markings: bool = True, seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    rng = np.random.default_rng(seed)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (60, 140, 60)
    noise = rng.integers(-12, 12, size=(HEIGHT, WIDTH, 1))
    frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
    if draw_markings:
        for d in range(0, 41, 5):
            p, q = project(d, -14), project(d, 20)
            cv2.line(frame, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), (235, 235, 235), 3, cv2.LINE_AA)
        for row in (0.0, HASH):
            for d in range(1, 40):
                if d % 5 == 0:
                    continue
                p, q = project(d, row - 0.35), project(d, row + 0.35)
                cv2.line(frame, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), (230, 230, 230), 2, cv2.LINE_AA)
    boxes = []
    # Draw far players first so nearer ones occlude them.
    for spot in sorted(spots, key=lambda s: s.l):
        foot = project(spot.d, spot.l)
        scale = np.linalg.norm(project(spot.d + 0.5, spot.l) - project(spot.d - 0.5, spot.l))
        height = int(round(2.0 * scale * 0.95))
        width = max(8, int(round(0.42 * height)))
        x1, y2 = int(round(foot[0] - width / 2)), int(round(foot[1]))
        x2, y1 = x1 + width, y2 - height
        jersey = WHITE if spot.team == "O" else DARK
        cv2.rectangle(frame, (x1 + width // 4, y1), (x2 - width // 4, y1 + height // 6), (90, 90, 90), -1)  # helmet
        cv2.rectangle(frame, (x1, y1 + height // 6), (x2, y1 + int(0.55 * height)), jersey, -1)  # torso
        cv2.rectangle(frame, (x1 + 2, y1 + int(0.55 * height)), (x2 - 2, y2), (200, 200, 200) if spot.team == "O" else (60, 60, 60), -1)
        boxes.append((x1, y1, x2, y2))
    return frame, boxes


def offensive_line(depth: float = 1.0) -> list[Spot]:
    return [Spot("O", LOS - depth, BALL_L + offset, "OL") for offset in (-2.9, -1.45, 0.0, 1.45, 2.9)]


def base_defense(box_backers: int = 2) -> list[Spot]:
    spots = [Spot("D", LOS + 1.0, BALL_L + offset, "DL") for offset in (-3.6, -1.0, 1.0, 3.6)]
    backers = [(-2.0,), (2.0,), (0.0,)][:box_backers]
    spots += [Spot("D", LOS + 4.5, BALL_L + offset[0], "LB") for offset in backers]
    spots += [Spot("D", LOS + 12.0, BALL_L - 6.0, "S"), Spot("D", LOS + 12.5, BALL_L + 9.0, "S")]
    return spots


def shotgun_two_by_two() -> list[Spot]:
    offense = offensive_line() + [
        Spot("O", LOS - 5.0, BALL_L, "QB"),
        Spot("O", LOS - 5.0, BALL_L + 1.6, "RB"),
        Spot("O", LOS - 1.0, BALL_L - 12.0, "WR"),
        Spot("O", LOS - 1.5, BALL_L - 7.0, "WR"),
        Spot("O", LOS - 1.5, BALL_L + 8.0, "WR"),
        Spot("O", LOS - 1.0, BALL_L + 14.0, "WR"),
    ]
    defense = base_defense(2) + [
        Spot("D", LOS + 6.0, BALL_L - 12.0, "CB"),
        Spot("D", LOS + 6.0, BALL_L + 14.0, "CB"),
        Spot("D", LOS + 5.0, BALL_L + 8.0, "NB"),
    ]
    return offense + defense


def i_formation() -> list[Spot]:
    offense = offensive_line() + [
        Spot("O", LOS - 1.6, BALL_L, "QB"),
        Spot("O", LOS - 4.5, BALL_L, "FB"),
        Spot("O", LOS - 7.0, BALL_L, "RB"),
        Spot("O", LOS - 1.0, BALL_L + 4.4, "TE"),
        Spot("O", LOS - 1.0, BALL_L - 12.0, "WR"),
        Spot("O", LOS - 1.5, BALL_L + 13.0, "WR"),
    ]
    defense = base_defense(3) + [
        Spot("D", LOS + 6.0, BALL_L - 12.0, "CB"),
        Spot("D", LOS + 6.0, BALL_L + 13.0, "CB"),
    ]
    return offense + defense


def pistol() -> list[Spot]:
    offense = offensive_line() + [
        Spot("O", LOS - 3.8, BALL_L, "QB"),
        Spot("O", LOS - 6.8, BALL_L, "RB"),
        Spot("O", LOS - 1.0, BALL_L + 4.4, "TE"),
        Spot("O", LOS - 1.0, BALL_L - 12.0, "WR"),
        Spot("O", LOS - 1.5, BALL_L - 7.0, "WR"),
        Spot("O", LOS - 1.5, BALL_L + 12.0, "WR"),
    ]
    defense = base_defense(2) + [
        Spot("D", LOS + 6.0, BALL_L - 12.0, "CB"),
        Spot("D", LOS + 6.0, BALL_L + 12.0, "CB"),
        Spot("D", LOS + 5.0, BALL_L - 7.0, "NB"),
    ]
    return offense + defense


def bunch_right() -> list[Spot]:
    offense = offensive_line() + [
        Spot("O", LOS - 5.0, BALL_L, "QB"),
        Spot("O", LOS - 5.0, BALL_L - 1.6, "RB"),
        Spot("O", LOS - 1.0, BALL_L + 6.5, "WR"),
        Spot("O", LOS - 2.3, BALL_L + 7.6, "WR"),
        Spot("O", LOS - 1.0, BALL_L + 8.7, "WR"),
        Spot("O", LOS - 1.0, BALL_L - 13.0, "WR"),
    ]
    defense = base_defense(2) + [
        Spot("D", LOS + 6.0, BALL_L - 13.0, "CB"),
        Spot("D", LOS + 5.0, BALL_L + 8.0, "CB"),
        Spot("D", LOS + 5.5, BALL_L + 11.0, "NB"),
    ]
    return offense + defense


def empty_three_by_two() -> list[Spot]:
    offense = offensive_line() + [
        Spot("O", LOS - 5.0, BALL_L, "QB"),
        Spot("O", LOS - 1.0, BALL_L - 13.0, "WR"),
        Spot("O", LOS - 1.5, BALL_L - 7.0, "WR"),
        Spot("O", LOS - 1.5, BALL_L + 6.5, "WR"),
        Spot("O", LOS - 1.5, BALL_L + 10.5, "WR"),
        Spot("O", LOS - 1.0, BALL_L + 15.0, "WR"),
    ]
    defense = base_defense(1) + [
        Spot("D", LOS + 6.0, BALL_L - 13.0, "CB"),
        Spot("D", LOS + 6.0, BALL_L + 15.0, "CB"),
        Spot("D", LOS + 5.0, BALL_L + 9.0, "NB"),
        Spot("D", LOS + 5.0, BALL_L - 7.0, "NB"),
    ]
    return offense + defense
