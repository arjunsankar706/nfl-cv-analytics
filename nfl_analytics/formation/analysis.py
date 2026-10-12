"""Pre-snap offensive formation and defensive alignment from one frame.

Pipeline: on-field people -> team split -> field registration -> line of
scrimmage -> offense/defense -> roles (OL, QB, backs, receivers; front,
second level, deep) -> labelled findings.

Every finding carries a value, a heuristic confidence in [0, 1] and a
human-readable reason. When the evidence is insufficient the value is
``UNKNOWN`` (or ``None`` for counts); nothing is guessed to fill a gap.

All distances are yards. ``depth`` is measured from the line of scrimmage
into each team's own backfield (positive for both teams); ``lateral`` is
measured from the ball, positive toward the offense's right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from nfl_analytics.vision.field import build_field_context

from .field_registration import (
    HASH_SEPARATION,
    TIER_HOMOGRAPHY,
    TIER_NONE,
    FieldRegistration,
    register_field,
)
from .teams import OFFICIAL, UNASSIGNED, TeamAssignment, assign_teams

Box = tuple[int, int, int, int]
UNKNOWN = "UNKNOWN"

# Geometry thresholds (yards). Values follow standard alignment conventions
# and are deliberately conservative; see README "Formation rules".
ON_LINE_DEPTH = 1.6  # used before the offense is known (both teams)
ON_LINE_BAND = 1.1  # on the line: within this of the offense's front-most feet
# Adjacent linemen stand ~1.3-1.6 yd apart; 3.2 tolerates one undetected
# lineman while slot receivers (4+ yd outside) stay separate.
LINE_GAP_MAX = 3.2
UNDER_CENTER_MAX = 1.5
OFF_BALL_MIN = 2.3
STACK_LATERAL = 1.0
BOX_DEPTH = 7.0
BOX_MARGIN = 1.0
DEEP_SAFETY_DEPTH = 9.0
# Half the field width is 26.7 yd; extrapolating the homography from the
# 6.2 yd hash baseline compresses distant laterals by ~5-10%, so the cut-off
# sits inside the true sideline.
SIDELINE_LIMIT = 23.5
TYPICAL_OL_SPACING = 1.45  # centre-to-centre between adjacent linemen' feet


@dataclass
class Finding:
    value: object
    confidence: float
    reason: str

    @property
    def known(self) -> bool:
        return self.value not in (None, UNKNOWN)

    def as_dict(self) -> dict:
        return {"value": self.value, "confidence": round(float(self.confidence), 2), "reason": self.reason}


def unknown(reason: str) -> Finding:
    return Finding(UNKNOWN, 0.0, reason)


@dataclass
class Player:
    index: int
    box: Box
    team: str
    foot: tuple[float, float]
    field_xy: Optional[np.ndarray] = None
    side: str = UNKNOWN  # OFFENSE / DEFENSE
    depth: float = float("nan")
    lateral: float = float("nan")
    role: str = UNKNOWN
    track_id: Optional[int] = None
    body_scale: float = 1.0  # local size relative to the median player (uncalibrated mode)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "track_id": self.track_id,
            "team": self.team,
            "side": self.side,
            "role": self.role,
            "depth_yd": None if np.isnan(self.depth) else round(float(self.depth), 2),
            "lateral_yd": None if np.isnan(self.lateral) else round(float(self.lateral), 2),
            "box": list(self.box),
        }


@dataclass
class FormationReport:
    status: str
    message: str
    registration: Optional[FieldRegistration] = None
    teams: Optional[TeamAssignment] = None
    players: list[Player] = field(default_factory=list)
    offense_team: Optional[str] = None
    defense_team: Optional[str] = None
    offense_identification: Finding = field(default_factory=lambda: unknown("not analysed"))
    lateral_approximate: bool = True
    los_image: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    ball_image: Optional[tuple[float, float]] = None
    findings: dict[str, Finding] = field(default_factory=dict)
    formation_tags: list[str] = field(default_factory=list)
    box_polygon_image: Optional[np.ndarray] = None
    line_extent: Optional[tuple[float, float]] = None  # inline width used for the box
    box_lateral: Optional[tuple[float, float]] = None
    ball_shift: float = 0.0
    trimmed_teams: set = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    def finding(self, name: str) -> Finding:
        return self.findings.get(name, unknown("not analysed"))

    @property
    def formation_name(self) -> str:
        return " · ".join(self.formation_tags) if self.formation_tags else UNKNOWN

    def offense(self) -> list[Player]:
        return [player for player in self.players if player.side == "OFFENSE"]

    def defense(self) -> list[Player]:
        return [player for player in self.players if player.side == "DEFENSE"]

    def model_inputs(self, min_confidence: float = 0.5) -> dict[str, float]:
        """Vision features for the RUN/PASS model; only confident findings."""
        inputs: dict[str, float] = {}
        alignment = self.finding("qb_alignment")
        if alignment.known and alignment.confidence >= min_confidence:
            for name in ("SHOTGUN", "PISTOL", "UNDER_CENTER"):
                inputs[f"qb_{name.lower()}"] = float(alignment.value == name)
        box = self.finding("box_count")
        if box.known and box.confidence >= min_confidence:
            inputs["defenders_in_box"] = float(box.value)
        return inputs

    def summary(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "calibration": None if self.registration is None else self.registration.tier,
            "calibration_notes": [] if self.registration is None else self.registration.notes,
            "offense_team": self.offense_team,
            "defense_team": self.defense_team,
            "offense_identification": self.offense_identification.as_dict(),
            "formation": self.formation_name,
            "findings": {name: finding.as_dict() for name, finding in self.findings.items()},
            "lateral_approximate": self.lateral_approximate,
            "warnings": self.warnings,
            "players": [player.as_dict() for player in self.players],
        }


# --------------------------------------------------------------------- inputs

def on_field_people(frame: np.ndarray, boxes: Sequence[Box]) -> tuple[list[int], np.ndarray]:
    """Indices of boxes whose feet stand inside the connected playing surface."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    field_context = build_field_context(hsv)
    mask = field_context.mask
    height, width = mask.shape
    keep = []
    heights = np.array([box[3] - box[1] for box in boxes], dtype=float)
    # Playing surface = turf or *thin* paint (yard lines, hashes). The wide
    # white sideline border is excluded, so people standing on it still fail.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thin_paint = (
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 28
    ) & (hsv[..., 1] < 90) & (hsv[..., 2] > 135)
    turf = (cv2.inRange(hsv, np.array((30, 50, 40), np.uint8), np.array((90, 255, 255), np.uint8)) > 0) | thin_paint
    occupied = np.zeros(mask.shape, dtype=np.int32)
    for x1, y1, x2, y2 in boxes:
        occupied[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] += 1
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        box_height = y2 - y1
        if box_height < 14 or (x2 - x1) < 5:
            continue
        if y2 >= height - 3:
            continue  # feet out of frame: ground position unknown
        # Real turf (not the closed field mask) must surround the lower legs;
        # this rejects logos and people drawn inside broadcast graphics.
        # Pixels covered by other detections (a clump of linemen) are ignored.
        pad = max(4, (x2 - x1) // 3)
        regions = [
            (slice(y1 + int(0.6 * box_height), y2), slice(max(0, x1 - pad), max(0, x1))),
            (slice(y1 + int(0.6 * box_height), y2), slice(min(width, x2), min(width, x2 + pad))),
            (slice(y2, min(height, y2 + pad)), slice(max(0, x1 - pad), min(width, x2 + pad))),
        ]
        fractions = []
        for rows, columns in regions:
            free = occupied[rows, columns] == 0
            if free.sum() >= 6:
                fractions.append(float(turf[rows, columns][free].mean()))
        if fractions and (min(fractions) < 0.3 or np.mean(fractions) < 0.5):
            continue
        centre_x = int(np.clip((x1 + x2) // 2, 0, width - 1))
        if field_context.has_field and y2 < field_context.boundary[centre_x] + 0.25 * box_height:
            continue  # sideline/crowd: feet at the far edge of the turf
        keep.append(index)
    # Remove implausibly tall boxes relative to neighbours (merged/close-up).
    if len(keep) >= 5:
        median = np.median(heights[keep])
        keep = [index for index in keep if heights[index] <= 2.6 * median]
    return keep, mask


def _separation_quality(feet: np.ndarray, teams: np.ndarray, axis: np.ndarray) -> tuple[int, float]:
    """(misplaced players, normalised gap) for the best threshold along an axis."""
    values = feet @ axis
    order = np.argsort(values)
    values, labels = values[order], teams[order]
    best = (len(values), 0.0)
    spread = np.std(values) + 1e-6
    for low in np.unique(labels):
        for cut in range(1, len(values)):
            errors = int(np.sum(labels[:cut] != low) + np.sum(labels[cut:] == low))
            gap = (values[cut] - values[cut - 1]) / spread
            if (errors, -gap) < (best[0], -best[1]):
                best = (errors, gap)
    return best


def _approximate_coordinates(players: list[Player], registration: FieldRegistration) -> str:
    """Uncalibrated fallback: image axes and a body-height scale.

    The downfield axis is perpendicular to any visible yard-line family; if no
    lines were seen, the direction that best separates the two teams is used.
    Coordinates are in "body units" (box height / 2 ~ 1 yd) and are only
    used for ordering and coarse, wide-margin thresholds.
    """
    feet = np.array([p.foot for p in players])
    teams = np.array([p.team for p in players])
    candidates = []
    for degrees in range(0, 180, 3):
        theta = np.radians(degrees)
        candidates.append(np.array([np.cos(theta), np.sin(theta)]))
    scored = [(_separation_quality(feet, teams, axis), axis) for axis in candidates]
    (best_errors, _), best_axis = min(scored, key=lambda item: (item[0][0], -item[0][1]))
    source = "team separation"
    if registration.family_direction is not None:
        direction = registration.family_direction / (np.linalg.norm(registration.family_direction) + 1e-12)
        normal = np.array([-direction[1], direction[0]])
        errors, _ = _separation_quality(feet, teams, normal)
        if errors <= best_errors + 1:
            best_axis, source = normal, "perpendicular to visible yard lines"
    lateral_axis = np.array([-best_axis[1], best_axis[0]])
    scale = float(np.median([p.box[3] - p.box[1] for p in players])) / 2.0
    for player in players:
        point = np.asarray(player.foot)
        player.field_xy = np.array([point @ best_axis, point @ lateral_axis]) / max(scale, 1e-6)
        player.body_scale = (player.box[3] - player.box[1]) / 2.0 / max(scale, 1e-6)
    return source


# ------------------------------------------------------------- the analysis

def analyze_formation(
    frame: np.ndarray,
    boxes: Sequence[Box],
    track_ids: Optional[Sequence[Optional[int]]] = None,
    known_labels: Optional[Sequence[Optional[str]]] = None,
    offense_hint: Optional[str] = None,
) -> FormationReport:
    """Analyse one pre-snap frame given person boxes (x1, y1, x2, y2)."""
    boxes = [tuple(int(v) for v in box) for box in boxes]
    keep, field_mask = on_field_people(frame, boxes)
    if len(keep) < 8:
        return FormationReport(
            "INSUFFICIENT",
            f"Only {len(keep)} people detected on the field; a pre-snap analysis needs most of the 22 players.",
        )
    # 1) Register the field first (team-agnostic). Every detected person is
    #    masked out of the paint search.
    kept_feet = np.array([((boxes[i][0] + boxes[i][2]) / 2.0, float(boxes[i][3])) for i in keep])
    registration = register_field(frame, field_mask, boxes, ball_hint=np.median(kept_feet, axis=0))
    report = FormationReport("INSUFFICIENT", "", registration=registration)
    # 2) With a full homography, drop people outside the sidelines *before*
    #    colour clustering, so staff and photographers cannot distort it.
    if registration.tier == TIER_HOMOGRAPHY:
        lateral = registration.to_field(kept_feet)[:, 1]
        inside = np.abs(lateral - HASH_SEPARATION / 2.0) <= SIDELINE_LIMIT
        removed = int(np.sum(~inside))
        keep = [index for index, flag in zip(keep, inside) if flag]
        if removed:
            report.warnings.append(f"{removed} people outside the sidelines ignored")
        if len(keep) < 8:
            report.message = f"Only {len(keep)} people inside the sidelines; a pre-snap analysis needs most of the 22 players."
            return report

    # 3) Split teams by jersey colour.
    kept_boxes = [boxes[i] for i in keep]
    kept_labels = None if known_labels is None else [known_labels[i] for i in keep]
    teams = assign_teams(frame, kept_boxes, kept_labels)
    report.teams = teams
    if not teams.reliable:
        report.message = f"Could not separate the two teams by jersey colour: {teams.note}."
        return report

    players: list[Player] = []
    for index, label in zip(keep, teams.labels):
        if label in (OFFICIAL, UNASSIGNED):
            continue
        x1, y1, x2, y2 = boxes[index]
        players.append(
            Player(
                index=index,
                box=boxes[index],
                team=label,
                foot=((x1 + x2) / 2.0, float(y2)),
                track_id=None if track_ids is None else track_ids[index],
            )
        )

    def group(players_: list[Player]) -> dict[str, list[Player]]:
        grouped: dict[str, list[Player]] = {}
        for player in players_:
            grouped.setdefault(player.team, []).append(player)
        return grouped

    by_team = group(players)
    if len(by_team) != 2 or min(len(members) for members in by_team.values()) < 4:
        report.message = "Need at least four players from each team on screen."
        report.players = players
        return report

    # 4) Field coordinates.
    feet = np.array([player.foot for player in players])
    if registration.calibrated:
        for player, xy in zip(players, registration.to_field(feet)):
            player.field_xy = xy
        players = _merge_duplicates(players, 0.7, report)
        by_team = group(players)
    else:
        source = _approximate_coordinates(players, registration)
        report.warnings.append(f"Uncalibrated downfield axis from {source}.")
        report.warnings.append(
            "No field calibration (yard lines not measurable): depths are approximate and "
            "depth-based labels are reported only when unambiguous."
        )
    # A team with 12-14 people usually includes sideline/end-line staff.
    # Trim the members farthest from the team's centre (at most three) and
    # mark the team incomplete so completeness-based calls are withheld.
    for name, members in list(by_team.items()):
        if 11 < len(members) <= 14:
            centre = np.median([p.field_xy for p in members], axis=0)
            ranked = sorted(members, key=lambda p: -np.linalg.norm(p.field_xy - centre))
            dropped = ranked[: len(members) - 11]
            players = [p for p in players if p not in dropped]
            report.trimmed_teams.add(name)
            report.warnings.append(
                f"{len(dropped)} outlying person(s) removed from team {name} (likely staff off the field)"
            )
    by_team = group(players)
    oversized = [name for name, members in by_team.items() if len(members) > 12]
    if oversized:
        report.message = (
            f"More than 12 people assigned to one team ({len(by_team[oversized[0]])}): "
            "non-players are mixed in, so the alignment cannot be trusted."
        )
        report.players = players
        return report
    report.lateral_approximate = registration.tier != TIER_HOMOGRAPHY

    # ----------------------------------------------------- line of scrimmage
    names = sorted(by_team)
    split = _split_line_of_scrimmage(players, names)
    if split is None:
        report.message = "The two teams are not separated along the field: not a pre-snap alignment."
        report.players = players
        return report
    los, low_team, errors = split
    high_team = names[1] if low_team == names[0] else names[0]
    if errors > 2:
        report.warnings.append(f"{errors} players are on the wrong side of the estimated line (motion or occlusion).")

    near_line = sorted(players, key=lambda p: abs(p.field_xy[0] - los))[:6]
    line_scale = float(np.median([p.body_scale for p in near_line]))

    def set_depths(offense: str) -> None:
        for player in players:
            downfield = player.field_xy[0] - los
            if not registration.calibrated:
                # Perspective: express the gap in the average local body size
                # of the player and the line.
                downfield /= (player.body_scale + line_scale) / 2.0
            toward_low = player.team == low_team
            player.depth = -downfield if toward_low else downfield
            player.side = "OFFENSE" if player.team == offense else "DEFENSE"

    # --------------------------------------------------- offense vs defense
    identification = _identify_offense(players, low_team, high_team, los, offense_hint, teams)
    report.offense_identification = identification
    if not identification.known:
        report.message = identification.reason
        report.players = players
        return report
    offense_team = identification.value
    defense_team = high_team if offense_team == low_team else low_team
    report.offense_team, report.defense_team = offense_team, defense_team
    set_depths(offense_team)
    # Before the snap nobody may be beyond the line; a player measured well
    # past it was mis-coloured (or is a mis-detection) and is excluded.
    wrong_side = [p for p in players if p.depth < -1.5]
    if wrong_side:
        players = [p for p in players if p.depth >= -1.5]
        report.warnings.append(f"{len(wrong_side)} player(s) on the wrong side of the line excluded (colour mix-up)")

    offense = [p for p in players if p.side == "OFFENSE"]
    defense = [p for p in players if p.side == "DEFENSE"]
    if not registration.lateral_calibrated:
        _rescale_lateral(players, offense)

    # Lateral coordinate relative to the ball, positive = offense's right.
    line_players = _offensive_line(offense)
    if len(line_players) < 3:
        report.message = "Could not find the offensive line (fewer than three linemen visible)."
        report.players = players
        _set_lateral_raw(players)
        return report
    raw_lateral = np.array([p.field_xy[1] for p in line_players])
    centre_raw = float(np.median(raw_lateral))
    right_sign = _right_hand_sign(registration, players, offense_team, low_team, los)
    for player in players:
        player.lateral = right_sign * (player.field_xy[1] - centre_raw)

    report.players = players
    report.status = "OK"
    _assign_offense_roles(report, offense, line_players)
    _assign_defense_roles(report, offense, defense, line_players)
    _image_overlays(report, los, low_team, centre_raw, right_sign)
    report.message = (
        f"{len(offense)} offensive and {len(defense)} defensive players located; "
        f"calibration {registration.tier}."
    )
    return report


def _box_iou(first: Box, second: Box) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - overlap
    return overlap / union if union > 0 else 0.0


def _merge_duplicates(players: list[Player], radius: float, report: FormationReport) -> list[Player]:
    """Collapse same-team detections of one person (heavy overlap, same feet).

    Feet distance alone is not enough: an under-center QB stands ~0.6 yd
    behind the centre, so both box overlap and foot distance must agree.
    """
    kept: list[Player] = []
    for player in sorted(players, key=lambda p: -(p.box[3] - p.box[1]) * (p.box[2] - p.box[0])):
        duplicate = any(
            other.team == player.team
            and np.linalg.norm(other.field_xy - player.field_xy) < radius
            and _box_iou(other.box, player.box) > 0.55
            for other in kept
        )
        if not duplicate:
            kept.append(player)
    if len(kept) < len(players):
        report.warnings.append(f"{len(players) - len(kept)} duplicate detection(s) merged")
    return kept


def _split_line_of_scrimmage(players: list[Player], names: list[str]) -> Optional[tuple[float, str, int]]:
    """Best 1D threshold on downfield position separating the two teams."""
    values = np.array([p.field_xy[0] for p in players])
    team = np.array([p.team for p in players])
    order = np.argsort(values)
    values, team = values[order], team[order]
    best = None
    for low in names:
        for cut in range(1, len(values)):
            errors = int(np.sum(team[:cut] != low) + np.sum(team[cut:] == low))
            margin = values[cut] - values[cut - 1]
            key = (errors, -margin)
            if best is None or key < best[0]:
                best = (key, cut, low)
    (errors, _), cut, low = best
    if errors > max(3, 0.2 * len(values)):
        return None
    low_values = np.array([p.field_xy[0] for p in players if p.team == low])
    high_values = np.array([p.field_xy[0] for p in players if p.team != low])
    # Midpoint between the nearest players of each team across the line.
    low_edge = np.sort(low_values)[-min(3, len(low_values)) :].mean()
    high_edge = np.sort(high_values)[: min(3, len(high_values))].mean()
    los = float((np.max(low_values[low_values <= values[cut]]) if np.any(low_values <= values[cut]) else low_edge)
                + (np.min(high_values[high_values >= values[cut - 1]]) if np.any(high_values >= values[cut - 1]) else high_edge)) / 2.0
    return los, low, errors


def _identify_offense(
    players: list[Player],
    low_team: str,
    high_team: str,
    los: float,
    hint: Optional[str],
    teams: TeamAssignment,
) -> Finding:
    if hint:
        if hint in (low_team, high_team):
            return Finding(hint, 1.0, "offense specified by the user")
        return unknown(f"offense team '{hint}' not present in this frame")

    def team_stats(name: str) -> dict:
        depths = np.array([abs(p.field_xy[0] - los) for p in players if p.team == name])
        laterals = np.array([p.field_xy[1] for p in players if p.team == name])
        front = np.sort(depths)[min(1, len(depths) - 1)]
        near = np.sort(laterals[depths <= front + ON_LINE_BAND])
        run = _longest_run(near, LINE_GAP_MAX)
        return {
            "near": int(len(near)),
            "run": run,
            "deep": int(np.sum(depths >= DEEP_SAFETY_DEPTH + 1.0)),
            "max_depth": float(depths.max()) if len(depths) else 0.0,
        }

    stats = {name: team_stats(name) for name in (low_team, high_team)}
    # Offense: a compact interior line and >= 7 men on the line; defense:
    # deep safeties beyond any offensive alignment depth.
    score = {
        name: 1.2 * min(s["run"], 6) + 0.4 * min(s["near"], 8) - 1.5 * s["deep"]
        for name, s in stats.items()
    }
    first, second = sorted(score, key=score.get, reverse=True)
    margin = score[first] - score[second]
    if margin < 1.0:
        return unknown(
            "Offense and defense look alike in this frame (no clearly tighter line or deep safeties); "
            "select the offense manually."
        )
    confidence = float(np.clip(0.5 + margin / 8.0, 0.5, 0.95))
    s = stats[first]
    return Finding(
        first,
        confidence,
        f"tightest line ({s['run']} adjacent on the line, {s['near']} within {ON_LINE_DEPTH} yd) "
        f"vs {stats[second]['run']}/{stats[second]['near']}; opponents have {stats[second]['deep']} deep players",
    )


def _longest_run(sorted_values: np.ndarray, max_gap: float) -> int:
    if len(sorted_values) == 0:
        return 0
    best = current = 1
    for gap in np.diff(sorted_values):
        current = current + 1 if gap <= max_gap else 1
        best = max(best, current)
    return best


def _rescale_lateral(players: list[Player], offense: list[Player]) -> None:
    """Uncalibrated lateral axis: scale so adjacent linemen are ~1.45 yd apart."""
    front = _line_reference(offense)
    near = sorted((p for p in offense if p.depth <= front + ON_LINE_BAND), key=lambda p: p.field_xy[1])
    gaps = np.diff([p.field_xy[1] for p in near])
    gaps = np.abs(gaps[np.abs(gaps) > 1e-6])
    if len(gaps) < 2:
        return
    # The smallest gaps are interior-line splits.
    reference = float(np.median(np.sort(gaps)[: max(2, min(4, len(gaps)))]))
    factor = TYPICAL_OL_SPACING / max(reference, 1e-6)
    for player in players:
        player.field_xy = np.array([player.field_xy[0], player.field_xy[1] * factor])


def _set_lateral_raw(players: list[Player]) -> None:
    for player in players:
        player.lateral = float(player.field_xy[1])


def _line_reference(offense: list[Player]) -> float:
    """Depth of the offense's front: robust to one mis-placed detection."""
    depths = np.sort([p.depth for p in offense if p.depth >= -0.5] or [p.depth for p in offense])
    return float(depths[min(1, len(depths) - 1)])


def _offensive_line(offense: list[Player]) -> list[Player]:
    front = _line_reference(offense)
    near = [p for p in offense if p.depth <= front + ON_LINE_BAND]
    # Linemen stand side by side, never directly behind one another: of two
    # players at the same lateral spot, the deeper one (e.g. an under-center
    # QB behind the centre) is not on the line.
    near = [
        p
        for p in near
        if not any(
            abs(q.field_xy[1] - p.field_xy[1]) < 0.45 and q.depth < p.depth - 0.4 for q in near if q is not p
        )
    ]
    near.sort(key=lambda p: p.field_xy[1])
    if not near:
        return []
    runs: list[list[Player]] = [[near[0]]]
    for previous, player in zip(near, near[1:]):
        if player.field_xy[1] - previous.field_xy[1] <= LINE_GAP_MAX:
            runs[-1].append(player)
        else:
            runs.append([player])
    run = max(runs, key=len)
    if len(run) <= 5:
        return run
    # Longer runs include attached tight ends: keep the tightest five.
    windows = [run[i : i + 5] for i in range(len(run) - 4)]
    return min(windows, key=lambda window: window[-1].field_xy[1] - window[0].field_xy[1])


def _right_hand_sign(
    registration: FieldRegistration,
    players: list[Player],
    offense_team: str,
    low_team: str,
    los: float,
) -> float:
    """+1 if the raw lateral axis points to the offense's right (see module doc).

    Uses on-screen geometry: the camera views the field from above, so the
    image is not mirrored. The offense faces toward the defense; its right on
    screen is the facing vector rotated 90 degrees clockwise (y axis down).
    """
    offense_points = np.array([p.foot for p in players if p.team == offense_team])
    defense_points = np.array([p.foot for p in players if p.team != offense_team])
    facing = defense_points.mean(axis=0) - offense_points.mean(axis=0)
    right_on_screen = np.array([-facing[1], facing[0]])
    # Image direction in which the raw lateral coordinate increases.
    anchor = offense_points.mean(axis=0)
    if registration.calibrated:
        base = registration.to_field(anchor)[0]
        jacobian = np.column_stack(
            [registration.to_field(anchor + step)[0] - base for step in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))]
        )
        try:
            lateral_direction = np.linalg.solve(jacobian, np.array([0.0, 1.0]))
        except np.linalg.LinAlgError:
            lateral_direction = np.array([0.0, 1.0])
    else:
        lateral_direction = _uncalibrated_lateral_direction(players)
    return 1.0 if float(lateral_direction @ right_on_screen) >= 0 else -1.0


def _uncalibrated_lateral_direction(players: list[Player]) -> np.ndarray:
    feet = np.array([p.foot for p in players])
    xy = np.array([p.field_xy for p in players])
    # Least-squares image direction of increasing lateral coordinate.
    design = np.column_stack([feet, np.ones(len(feet))])
    solution, *_ = np.linalg.lstsq(design, xy[:, 1], rcond=None)
    gradient = solution[:2]
    return gradient / (np.linalg.norm(gradient) + 1e-12)


# -------------------------------------------------------------- offense roles

def _assign_offense_roles(report: FormationReport, offense: list[Player], line: list[Player]) -> None:
    registration = report.registration
    calibrated = registration is not None and registration.calibrated
    sigma = registration.depth_uncertainty_yd if registration is not None else 1.5
    findings = report.findings
    for player in line:
        player.role = "OL"
    if len(line) == 5:
        line[2].role = "C"
    line_depth = float(np.median([p.depth for p in line]))
    line_left = min(p.lateral for p in line)
    line_right = max(p.lateral for p in line)
    completeness = len(offense)
    if report.offense_team in report.trimmed_teams:
        completeness = min(completeness, 10)  # count is not trustworthy after trimming
    findings["offense_players_visible"] = Finding(
        completeness, 1.0 if completeness == 11 else 0.6, f"{completeness} of 11 offensive players located"
    )
    findings["offensive_line"] = Finding(
        len(line),
        (0.9 if len(line) == 5 else 0.5) if calibrated else 0.35,
        f"{len(line)} linemen in a contiguous group "
        f"({line_right - line_left:.1f} yd wide); line depth {line_depth:.1f} yd behind the ball",
    )

    # QB: shallowest player directly behind the ball. With an undetected
    # lineman the ball estimate (median lineman) can be off by ~1 yd, so the
    # search window widens.
    line_complete = len(line) == 5
    window = 1.6 if line_complete else 2.6
    behind = [p for p in offense if p.role == UNKNOWN and p.depth > line_depth + 0.2 and abs(p.lateral) <= window]
    qb = min(behind, key=lambda p: p.depth + 2.0 * abs(p.lateral), default=None)
    if qb is None:
        findings["qb_alignment"] = unknown("no player found directly behind the centre")
    else:
        qb.role = "QB"
    # Re-centre on the ball: the centre lineman when all five are seen,
    # otherwise the QB (who aligns on the ball in every QB alignment).
    if line_complete:
        shift = line[2].lateral
    elif qb is not None:
        shift = qb.lateral
    else:
        shift = 0.0
    for player in report.players:
        player.lateral -= shift
    report.ball_shift = shift
    line_left = min(p.lateral for p in line)
    line_right = max(p.lateral for p in line)
    if not line_complete:
        # Expected tackle positions (2 x ~1.45 yd from the centre).
        line_left = min(line_left, -2 * TYPICAL_OL_SPACING)
        line_right = max(line_right, 2 * TYPICAL_OL_SPACING)
    report.line_extent = (line_left, line_right)
    others = [p for p in offense if p.role == UNKNOWN]
    core_width = max(abs(line_left), abs(line_right)) + 2.5
    backs = [p for p in others if p.depth > line_depth + 1.0 and abs(p.lateral) <= core_width]
    for back in backs:
        back.role = "RB"
    receivers = [p for p in others if p.role == UNKNOWN]
    for receiver in receivers:
        attached = receiver.depth <= line_depth + ON_LINE_BAND and (
            line_left - 2.2 <= receiver.lateral <= line_right + 2.2
        )
        receiver.role = "TE" if attached else "WR"

    if qb is not None:
        relative = qb.depth - line_depth
        stacked = [b for b in backs if abs(b.lateral - qb.lateral) <= STACK_LATERAL and b.depth > qb.depth + 1.0]
        if calibrated:
            under_max, off_min = UNDER_CENTER_MAX, OFF_BALL_MIN
            measure = f"QB {relative:.1f} yd behind the line's feet"
        else:
            # Body-height units: ambiguous foreshortening, so wider dead zone.
            under_max, off_min = 0.6, 1.1
            measure = f"QB {relative:.1f} body-lengths behind the line (uncalibrated)"
        gun_spot = [
            p
            for p in offense
            if p is not qb and abs(p.lateral - qb.lateral) <= 1.0 and p.depth - line_depth >= off_min
        ]
        if relative <= under_max and not line_complete and gun_spot and len(stacked) < 2:
            # With a lineman missing, a merged lineman can look like an
            # under-center QB while the real QB stands in the gun spot. A
            # shotgun QB stands ~3.5-4.8 yd behind the linemen's feet; a
            # singleback RB ~5.5-6.5 yd. Resolve only the unambiguous case.
            gun = min(gun_spot, key=lambda p: p.depth)
            gun_relative = gun.depth - line_depth
            behind_gun = [p for p in offense if p is not gun and abs(p.lateral - gun.lateral) <= 1.0 and p.depth > gun.depth + 1.0]
            if gun_relative <= 4.8 and not behind_gun and calibrated:
                qb.role = "OL"
                line.append(qb)
                if gun in backs:
                    backs.remove(gun)
                gun.role = "QB"
                qb = gun
                findings["qb_alignment"] = Finding(
                    "SHOTGUN",
                    0.6,
                    f"QB {gun_relative:.1f} yd behind the line's feet with nobody stacked behind; the shallow player "
                    f"behind the centre is treated as an unseparated lineman ({len(line) - 1}/5 linemen detected)",
                )
            else:
                findings["qb_alignment"] = unknown(
                    measure + f", but only {len(line)}/5 linemen were separated and a player stands in the shotgun spot"
                )
        elif relative <= under_max:
            margin = (under_max - relative) / sigma
            findings["qb_alignment"] = Finding(
                "UNDER_CENTER", float(np.clip(0.55 + 0.2 * margin, 0.5, 0.95)), measure
            )
        elif relative >= off_min:
            margin = (relative - off_min) / sigma
            confidence = float(np.clip(0.55 + 0.2 * margin, 0.5, 0.95))
            if stacked and (not calibrated or relative <= 4.6):
                findings["qb_alignment"] = Finding(
                    "PISTOL", confidence * 0.9, measure + f"; back stacked {stacked[0].depth - qb.depth:.1f} yd directly behind"
                )
            else:
                findings["qb_alignment"] = Finding("SHOTGUN", confidence, measure + "; no back stacked behind QB")
        else:
            findings["qb_alignment"] = unknown(measure + ": between under-center and shotgun depths")

    # Backfield structure.
    complete = completeness >= 10
    if not backs:
        if complete and qb is not None:
            findings["backfield"] = Finding("EMPTY", 0.75 if completeness == 11 else 0.6, "no back behind the line besides the QB")
        else:
            findings["backfield"] = unknown("no back visible, but too few offensive players located to call EMPTY")
    else:
        description = f"{len(backs)} back(s)"
        value = f"{len(backs)}-BACK"
        alignment = findings.get("qb_alignment", unknown(""))
        if qb is not None and alignment.value == "UNDER_CENTER":
            stacked_all = [b for b in backs if abs(b.lateral - qb.lateral) <= STACK_LATERAL]
            if len(backs) == 2 and len(stacked_all) == 2 and abs(backs[0].depth - backs[1].depth) >= 1.2:
                value, description = "I_FORMATION", "two backs stacked directly behind an under-center QB"
            elif len(backs) == 1 and backs[0].depth - line_depth >= 3.5:
                value, description = "SINGLEBACK", "one back deep behind an under-center QB"
        confidence = 0.7 if complete else 0.45
        findings["backfield"] = Finding(value, confidence, description)

    # Receiver distribution (offense's left x right).
    left = sum(1 for p in receivers if p.lateral < 0)
    right = sum(1 for p in receivers if p.lateral > 0)
    if len(receivers) + len(backs) > 5:
        # Only five eligible players besides the QB exist: extra "receivers"
        # are unseparated linemen or mis-detections.
        reason = (
            f"{len(receivers)} receivers + {len(backs)} backs exceeds the 5 eligible players; "
            f"linemen were not fully separated ({len(line)}/5 found)"
        )
        findings["receiver_distribution"] = unknown(reason)
        findings["trips"] = unknown(reason)
        findings["bunch"] = unknown(reason)
    else:
        line_penalty = 1.0 if line_complete else 0.6
        findings["receiver_distribution"] = Finding(
            f"{left}x{right}",
            (0.75 if completeness == 11 else (0.5 if complete else 0.3)) * line_penalty,
            f"{left} receiver(s) to the offense's left, {right} to the right"
            + ("" if completeness == 11 else f" ({11 - completeness} offensive player(s) not located)"),
        )
        strong = max(left, right)
        side = "RIGHT" if right > left else "LEFT"
        if strong >= 3:
            findings["trips"] = Finding(
                f"{'QUADS' if strong >= 4 else 'TRIPS'} {side}",
                0.8 * line_penalty,
                f"{strong} receivers aligned to the {side.lower()}",
            )
            group = sorted((p for p in receivers if (p.lateral > 0) == (side == "RIGHT")), key=lambda p: abs(p.lateral))
            cluster = _tightest_group(group, 3)
            if cluster is not None:
                lateral_span = max(p.lateral for p in cluster) - min(p.lateral for p in cluster)
                depth_span = max(p.depth for p in cluster) - min(p.depth for p in cluster)
                if lateral_span <= 3.5 and depth_span >= 0.6:
                    findings["bunch"] = Finding(
                        f"BUNCH {side}",
                        (0.6 if report.lateral_approximate else 0.75) * line_penalty,
                        f"three receivers within {lateral_span:.1f} yd laterally, staggered {depth_span:.1f} yd",
                    )
                else:
                    findings["bunch"] = Finding(
                        "NO", 0.6, f"closest three receivers span {lateral_span:.1f} yd (bunch needs <= 3.5, staggered)"
                    )
        elif completeness == 11 and line_complete:
            findings["trips"] = Finding("NO", 0.7, f"at most {strong} receivers to one side")
            findings["bunch"] = Finding("NO", 0.7, "no three-receiver side")
        else:
            findings["trips"] = unknown(
                f"max {strong} receivers seen to a side, but {11 - completeness} offensive player(s) missing"
                if completeness < 11
                else f"only {len(line)}/5 linemen separated"
            )
            findings["bunch"] = unknown("receiver set incomplete")

    # Formation tags (coexisting labels).
    tags = []
    alignment = findings.get("qb_alignment", unknown(""))
    if alignment.known:
        tags.append(alignment.value.replace("_", " ").title())
    backfield = findings.get("backfield", unknown(""))
    if backfield.value in ("I_FORMATION", "SINGLEBACK", "EMPTY"):
        tags.append({"I_FORMATION": "I-Formation", "SINGLEBACK": "Singleback", "EMPTY": "Empty"}[backfield.value])
    if findings.get("trips", unknown("")).value not in (UNKNOWN, "NO", None):
        tags.append(findings["trips"].value.title())
    if findings.get("bunch", unknown("")).value not in (UNKNOWN, "NO", None):
        tags.append(findings["bunch"].value.title())
    report.formation_tags = tags


def _tightest_group(players: list[Player], size: int) -> Optional[list[Player]]:
    if len(players) < size:
        return None
    ordered = sorted(players, key=lambda p: p.lateral)
    windows = [ordered[i : i + size] for i in range(len(ordered) - size + 1)]
    return min(windows, key=lambda w: w[-1].lateral - w[0].lateral)


# -------------------------------------------------------------- defense roles

def _assign_defense_roles(
    report: FormationReport,
    offense: list[Player],
    defense: list[Player],
    line: list[Player],
) -> None:
    findings = report.findings
    registration = report.registration
    calibrated = registration is not None and registration.calibrated
    inline = [p for p in offense if p.role in ("OL", "C", "TE")]
    extent_left, extent_right = report.line_extent or (
        min(p.lateral for p in inline),
        max(p.lateral for p in inline),
    )
    box_left = min([extent_left] + [p.lateral for p in inline]) - BOX_MARGIN
    box_right = max([extent_right] + [p.lateral for p in inline]) + BOX_MARGIN
    report.box_lateral = (box_left, box_right)
    trimmed = report.defense_team in report.trimmed_teams
    findings["defense_players_visible"] = Finding(
        len(defense),
        1.0 if len(defense) == 11 and not trimmed else 0.6,
        f"{len(defense)} of 11 defenders located" + (" (after removing outliers)" if trimmed else ""),
    )
    for player in defense:
        inside = box_left <= player.lateral <= box_right
        if player.depth >= DEEP_SAFETY_DEPTH:
            player.role = "S"
        elif inside and player.depth <= 2.0:
            player.role = "DL"
        elif inside and player.depth <= BOX_DEPTH:
            player.role = "LB"
        else:
            player.role = "CB" if player.depth <= 5.0 else "DB"

    if not calibrated:
        findings["box_count"] = unknown("box count needs a calibrated field (yard lines)")
    elif len(line) < 4:
        findings["box_count"] = unknown("offensive line edges not located")
    else:
        in_box = [p for p in defense if box_left <= p.lateral <= box_right and 0 <= p.depth <= BOX_DEPTH]
        border = [
            p
            for p in defense
            if (min(abs(p.lateral - box_left), abs(p.lateral - box_right)) <= 0.5 and 0 <= p.depth <= BOX_DEPTH + 0.5)
            or (box_left <= p.lateral <= box_right and abs(p.depth - BOX_DEPTH) <= 0.5)
        ]
        lateral_note = " (lateral scale approximate)" if report.lateral_approximate else ""
        confidence = 0.85 - 0.12 * len(border) - (0.15 if report.lateral_approximate else 0.0)
        if len(defense) < 9:
            confidence -= 0.2
        if len(line) < 5:
            confidence -= 0.15
            lateral_note += f"; box edges assume standard tackle splits ({len(line)}/5 linemen seen)"
        findings["box_count"] = Finding(
            len(in_box),
            float(np.clip(confidence, 0.2, 0.9)),
            f"defenders within {BOX_MARGIN:.0f} yd outside the end men on the line and {BOX_DEPTH:.0f} yd deep; "
            f"{len(border)} on the boundary{lateral_note}",
        )
    front = sum(1 for p in defense if p.role == "DL")
    second = sum(1 for p in defense if p.role == "LB")
    secondary = len(defense) - front - second
    if calibrated:
        shell_visible = _deep_zone_visible(report)
        deep = [p for p in defense if p.role == "S"]
        if len(defense) >= 10 and shell_visible and not trimmed:
            name = {0: "0-HIGH", 1: "1-HIGH (MOFC)", 2: "2-HIGH (MOFO)"}.get(len(deep), f"{len(deep)}-DEEP")
            findings["safety_shell"] = Finding(
                name, 0.6, f"{len(deep)} defender(s) deeper than {DEEP_SAFETY_DEPTH:.0f} yd before the snap (alignment only, not coverage)"
            )
        else:
            findings["safety_shell"] = unknown("deep field not fully visible or defenders missing")
        findings["defensive_front"] = Finding(
            f"{front} on line / {second} second level / {secondary} secondary",
            0.55 if len(defense) >= 10 else 0.35,
            "alignment-based (stance and personnel are not identified)",
        )
        if len(defense) >= 10:
            name = {4: "Base (4 DB)", 5: "Nickel-like (5 DB)", 6: "Dime-like (6 DB)"}.get(secondary, f"{secondary} DB")
            findings["defensive_structure"] = Finding(
                f"{front + second} box / {name}", 0.4, "approximate: secondary = defenders outside the box or deep"
            )
    else:
        findings["defensive_front"] = unknown("needs field calibration")


def _deep_zone_visible(report: FormationReport) -> bool:
    registration = report.registration
    if registration is None or registration.tier != TIER_HOMOGRAPHY or registration.homography is None:
        # Without an inverse map, require the deepest defender to be well past 9 yd.
        return any(p.depth >= DEEP_SAFETY_DEPTH + 2 for p in report.defense()) or len(report.defense()) == 11
    return len(report.defense()) == 11 or any(p.depth >= DEEP_SAFETY_DEPTH + 2 for p in report.defense())


def _image_overlays(report: FormationReport, los: float, low_team: str, centre_raw: float, right_sign: float) -> None:
    registration = report.registration
    offense = report.offense()
    centre = min((p for p in offense if p.role in ("OL", "C", "QB")), key=lambda p: abs(p.lateral), default=None)
    if centre is not None:
        report.ball_image = centre.foot
    if registration is None or registration.tier != TIER_HOMOGRAPHY:
        return
    inverse = np.linalg.inv(registration.homography)
    direction = 1.0 if report.offense_team == low_team else -1.0

    def to_image(depth: float, lateral: float) -> np.ndarray:
        downfield = los + direction * depth  # positive depth = defense side
        raw_lateral = centre_raw + right_sign * (lateral + report.ball_shift)
        return cv2.perspectiveTransform(np.array([[[downfield, raw_lateral]]], dtype=np.float64), inverse)[0, 0]

    report.los_image = (tuple(to_image(0.0, -26.0)), tuple(to_image(0.0, 26.0)))
    if report.box_lateral is not None:
        left, right = report.box_lateral
        report.box_polygon_image = np.array(
            [to_image(0, left), to_image(0, right), to_image(BOX_DEPTH, right), to_image(BOX_DEPTH, left)]
        )
