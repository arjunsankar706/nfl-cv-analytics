"""Draw formation analysis on the broadcast frame and as a top-down diagram."""

from __future__ import annotations

import cv2
import numpy as np

from .analysis import BOX_DEPTH, FormationReport

OFFENSE_BGR = (60, 200, 255)  # amber
DEFENSE_BGR = (255, 170, 60)  # cyan-blue
LOS_BGR = (40, 220, 255)
BOX_BGR = (90, 90, 255)


def draw_formation_overlay(frame: np.ndarray, report: FormationReport) -> np.ndarray:
    canvas = frame.copy()
    if report.box_polygon_image is not None:
        polygon = np.rint(report.box_polygon_image).astype(np.int32)
        shade = canvas.copy()
        cv2.fillPoly(shade, [polygon], BOX_BGR)
        cv2.addWeighted(shade, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [polygon], True, BOX_BGR, 2, cv2.LINE_AA)
    if report.los_image is not None:
        start, end = (tuple(int(round(v)) for v in point) for point in report.los_image)
        cv2.line(canvas, start, end, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(canvas, start, end, LOS_BGR, 2, cv2.LINE_AA)
    for player in report.players:
        x1, y1, x2, y2 = player.box
        colour = OFFENSE_BGR if player.side == "OFFENSE" else DEFENSE_BGR
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = player.role if player.role != "UNKNOWN" else "?"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - height - 6), (x1 + width + 6, y1), colour, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    lines = [
        f"FORMATION: {report.formation_name}",
        f"BOX: {_value(report, 'box_count')}   SHELL: {_value(report, 'safety_shell')}",
        f"CALIBRATION: {report.registration.tier if report.registration else 'NONE'}",
    ]
    panel_height = 22 * len(lines) + 12
    shade = canvas.copy()
    cv2.rectangle(shade, (8, 8), (520, 8 + panel_height), (12, 14, 18), -1)
    cv2.addWeighted(shade, 0.78, canvas, 0.22, 0, canvas)
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (18, 30 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def _value(report: FormationReport, name: str) -> str:
    finding = report.finding(name)
    return str(finding.value)


def tactical_diagram(report: FormationReport, theme: dict | None = None):
    """Top-down matplotlib figure in yards: offense at the bottom facing up."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    theme = theme or {}
    background = theme.get("background", "#0b0f14")
    turf = theme.get("turf", "#12301f")
    ink = theme.get("ink", "#d7dde5")
    offense_colour = theme.get("offense", "#ffc83d")
    defense_colour = theme.get("defense", "#3db4ff")

    figure, axis = plt.subplots(figsize=(6.4, 6.0), dpi=140)
    figure.patch.set_facecolor(background)
    axis.set_facecolor(turf)
    players = [p for p in report.players if not np.isnan(p.lateral)]
    laterals = [p.lateral for p in players] or [0.0]
    span = max(14.0, min(27.0, max(abs(v) for v in laterals) + 3.0))
    axis.set_xlim(-span, span)
    axis.set_ylim(-10, 17)
    for yard in range(-10, 18, 5):
        axis.axhline(yard, color="#ffffff", alpha=0.12, linewidth=1)
    axis.axhline(0, color="#ffd400", linewidth=2.2, label="line of scrimmage")
    if report.box_lateral is not None:
        left, right = report.box_lateral
        axis.add_patch(
            Rectangle((left, 0), right - left, BOX_DEPTH, facecolor="#ff5a5a", alpha=0.12, edgecolor="#ff5a5a", linewidth=1.2)
        )
    for player in players:
        y = -player.depth if player.side == "OFFENSE" else player.depth
        colour = offense_colour if player.side == "OFFENSE" else defense_colour
        marker = "o" if player.side == "OFFENSE" else "X"
        axis.scatter(player.lateral, y, s=230, color=colour, marker=marker, edgecolors="#000000", linewidths=0.8, zorder=3)
        axis.annotate(
            player.role if player.role != "UNKNOWN" else "?",
            (player.lateral, y),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            color=ink,
            fontsize=7.5,
            fontweight="bold",
        )
    axis.set_xlabel("Lateral (yd, offense's right is +)" + (" ~approx." if report.lateral_approximate else ""), color=ink)
    axis.set_ylabel("Depth from line of scrimmage (yd)", color=ink)
    axis.tick_params(colors=ink)
    for spine in axis.spines.values():
        spine.set_color("#2a3440")
    axis.set_title(report.formation_name, color=ink, fontsize=11, loc="left")
    figure.tight_layout()
    return figure
