"""Player eligibility and Raiders/Broncos/referee classification.

The thresholds were tuned on the original Raiders-Broncos broadcast clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import TEAM_NAMES, PipelineConfig
from .field import FieldContext, clipped_region, ratio


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
class UniformAnalysis:
    scores: dict[str, float]
    vote_weight: float
    referee_texture: float
    equipment_score: float
    light_pants_ratio: float
    number_structure: float = 0.0
    player_visual_score: float = 0.0
    force_nonplayer: bool = False


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
