"""High-resolution detection + formation analysis, and multi-frame consensus."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from nfl_analytics.vision.device import choose_device

from .analysis import UNKNOWN, Finding, FormationReport, analyze_formation

Box = tuple[int, int, int, int]


class FormationAnalyzer:
    """Runs a single-image YOLO pass tuned for crowded pre-snap alignments.

    Tracking uses 640 px input for speed; a pre-snap frame is analysed once,
    so it is worth 1920 px input, where the detector separates linemen in a
    clump far better (measured on the sample clip: 6 vs 4 boxes in the line).
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "auto",
        image_size: int = 1920,
        confidence: float = 0.2,
        model=None,
    ) -> None:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_path)
        self.model = model
        self.device = choose_device(device)
        self.image_size = image_size
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Box]:
        result = self.model.predict(
            frame,
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return [tuple(int(v) for v in box) for box in result.boxes.xyxy.cpu().numpy()]

    def analyze(
        self,
        frame: np.ndarray,
        offense_hint: Optional[str] = None,
        tracked: Optional[Sequence[tuple[Box, int, str]]] = None,
    ) -> FormationReport:
        boxes = self.detect(frame)
        labels = track_ids = None
        if tracked:
            labels, track_ids = match_tracked_labels(boxes, tracked)
        return analyze_formation(frame, boxes, track_ids=track_ids, known_labels=labels, offense_hint=offense_hint)


def iou(first: Box, second: Box) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - overlap
    return overlap / union if union > 0 else 0.0


def match_tracked_labels(
    boxes: Sequence[Box], tracked: Sequence[tuple[Box, int, str]]
) -> tuple[list[Optional[str]], list[Optional[int]]]:
    """Carry temporal team labels/IDs from the tracker onto new detections (IoU >= 0.5)."""
    labels: list[Optional[str]] = []
    ids: list[Optional[int]] = []
    for box in boxes:
        best = max(tracked, key=lambda item: iou(box, item[0]), default=None)
        if best is not None and iou(box, best[0]) >= 0.5:
            labels.append(best[2])
            ids.append(best[1])
        else:
            labels.append(None)
            ids.append(None)
    return labels, ids


@dataclass
class ConsensusReport:
    """Findings agreed across several frames of one pre-snap window."""

    frames_used: int
    frames_ok: int
    findings: dict[str, Finding]
    formation_tags: list[str]
    best: Optional[FormationReport]
    offense_team: Optional[str]

    @property
    def formation_name(self) -> str:
        return " · ".join(self.formation_tags) if self.formation_tags else UNKNOWN

    def model_inputs(self, min_confidence: float = 0.5) -> dict[str, float]:
        inputs: dict[str, float] = {}
        alignment = self.findings.get("qb_alignment")
        if alignment is not None and alignment.known and alignment.confidence >= min_confidence:
            for name in ("SHOTGUN", "PISTOL", "UNDER_CENTER"):
                inputs[f"qb_{name.lower()}"] = float(alignment.value == name)
        box = self.findings.get("box_count")
        if box is not None and box.known and box.confidence >= min_confidence:
            inputs["defenders_in_box"] = float(box.value)
        return inputs


def consensus(reports: Sequence[FormationReport]) -> ConsensusReport:
    ok = [report for report in reports if report.status == "OK"]
    if not ok:
        return ConsensusReport(len(reports), 0, {}, [], reports[0] if reports else None, None)
    offense_votes = Counter(report.offense_team for report in ok)
    offense_team, _ = offense_votes.most_common(1)[0]
    ok = [report for report in ok if report.offense_team == offense_team]
    by_name: dict[str, list[Finding]] = defaultdict(list)
    for report in ok:
        for name, finding in report.findings.items():
            by_name[name].append(finding)
    agreed: dict[str, Finding] = {}
    for name, findings in by_name.items():
        known = [f for f in findings if f.known]
        if not known:
            agreed[name] = Finding(UNKNOWN, 0.0, findings[-1].reason)
            continue
        if name == "box_count":
            values = np.array([f.value for f in known], dtype=float)
            median = int(round(float(np.median(values))))
            agreement = float(np.mean(np.abs(values - median) <= 0.5))
            coverage = 0.5 + 0.5 * len(known) / len(findings)
            confidence = float(np.mean([f.confidence for f in known])) * agreement * coverage
            agreed[name] = Finding(
                median,
                confidence,
                f"median of {len(known)} frames ({', '.join(str(int(v)) for v in values)}); " + known[-1].reason,
            )
            continue
        weights: dict[object, float] = defaultdict(float)
        for finding in known:
            weights[finding.value] += finding.confidence
        value = max(weights, key=weights.get)
        supporting = [f for f in known if f.value == value]
        # Contradicting frames cost more than frames that could not tell.
        agreement = len(supporting) / len(known)
        coverage = 0.5 + 0.5 * len(known) / len(findings)
        confidence = float(np.mean([f.confidence for f in supporting])) * agreement * coverage
        reason = supporting[-1].reason
        if len(findings) > 1:
            reason += f" (agreed in {len(supporting)}/{len(findings)} frames)"
        agreed[name] = Finding(value, confidence, reason)
    best = max(ok, key=lambda report: (len(report.offense()) + len(report.defense()), report.registration.tier == "HOMOGRAPHY"))
    tags = _tags_from(agreed)
    return ConsensusReport(len(reports), len(ok), agreed, tags, best, offense_team)


def _tags_from(findings: dict[str, Finding]) -> list[str]:
    tags = []
    alignment = findings.get("qb_alignment")
    if alignment is not None and alignment.known and alignment.confidence >= 0.4:
        tags.append(str(alignment.value).replace("_", " ").title())
    backfield = findings.get("backfield")
    if backfield is not None and backfield.value in ("I_FORMATION", "SINGLEBACK", "EMPTY") and backfield.confidence >= 0.4:
        tags.append({"I_FORMATION": "I-Formation", "SINGLEBACK": "Singleback", "EMPTY": "Empty"}[backfield.value])
    for name in ("trips", "bunch"):
        finding = findings.get(name)
        if finding is not None and finding.value not in (UNKNOWN, "NO", None) and finding.confidence >= 0.5:
            tags.append(str(finding.value).title())
    return tags
