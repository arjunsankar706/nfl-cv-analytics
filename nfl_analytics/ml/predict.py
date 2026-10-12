"""Serve the trained RUN/PASS models with explicit status reporting.

The predictor never fills in a missing input. It picks the richest trained
feature set whose inputs are *all* supplied; if none qualifies, it returns a
status explaining exactly which inputs are missing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_DESCRIPTIONS, FEATURE_SETS

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
# Most informative first. `full` needs personnel, which video cannot supply
# reliably, but a user may enter it manually.
PREFERENCE = ("full", "vision_situation", "vision", "situation")
QB_GROUP = ("qb_shotgun", "qb_pistol", "qb_under_center")

STATUS_OK = "OK"
STATUS_NO_MODEL = "MODEL_NOT_TRAINED"
STATUS_INSUFFICIENT = "INSUFFICIENT_FEATURES"


@dataclass
class Prediction:
    status: str
    message: str
    p_pass: Optional[float] = None
    feature_set: Optional[str] = None
    model_name: Optional[str] = None
    inputs: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    contributions: list[dict] = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)

    @property
    def p_run(self) -> Optional[float]:
        return None if self.p_pass is None else 1.0 - self.p_pass

    @property
    def call(self) -> Optional[str]:
        if self.p_pass is None:
            return None
        return "PASS" if self.p_pass >= 0.5 else "RUN"

    @property
    def confidence(self) -> Optional[float]:
        return None if self.p_pass is None else max(self.p_pass, 1.0 - self.p_pass)


def alignment_to_features(alignment: Optional[str]) -> dict[str, float]:
    """Map a QB alignment label to the one-hot model inputs (empty if unknown)."""
    if alignment not in ("SHOTGUN", "PISTOL", "UNDER_CENTER"):
        return {}
    return {
        "qb_shotgun": float(alignment == "SHOTGUN"),
        "qb_pistol": float(alignment == "PISTOL"),
        "qb_under_center": float(alignment == "UNDER_CENTER"),
    }


class RunPassPredictor:
    def __init__(self, artifacts: Mapping[str, dict]):
        self.artifacts = dict(artifacts)

    @classmethod
    def load(cls, model_dir: Path | str = MODEL_DIR) -> "RunPassPredictor":
        artifacts = {}
        for feature_set in FEATURE_SETS:
            path = Path(model_dir) / f"run_pass_{feature_set}.joblib"
            if path.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    artifacts[feature_set] = joblib.load(path)
        return cls(artifacts)

    @property
    def available(self) -> bool:
        return bool(self.artifacts)

    def describe(self) -> list[dict]:
        rows = []
        for feature_set in PREFERENCE:
            artifact = self.artifacts.get(feature_set)
            if artifact is None:
                continue
            metrics = artifact.get("test_metrics", {})
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": artifact["model_name"],
                    "trained_on": f"{min(artifact['trained_on_seasons'])}-{max(artifact['trained_on_seasons'])}",
                    "test_seasons": ", ".join(str(s) for s in artifact["test_seasons"]),
                    "test_accuracy": metrics.get("accuracy"),
                    "test_roc_auc": metrics.get("roc_auc"),
                    "test_log_loss": metrics.get("log_loss"),
                }
            )
        return rows

    def predict(self, inputs: Mapping[str, Optional[float]]) -> Prediction:
        supplied = {
            name: float(value)
            for name, value in inputs.items()
            if value is not None and not (isinstance(value, float) and np.isnan(value))
        }
        if not self.artifacts:
            return Prediction(
                STATUS_NO_MODEL,
                "No trained model found in models/. Run `python -m nfl_analytics.ml.train --download`.",
                inputs=supplied,
            )
        best_missing: list[str] | None = None
        for feature_set in PREFERENCE:
            artifact = self.artifacts.get(feature_set)
            if artifact is None:
                continue
            missing = [name for name in artifact["features"] if name not in supplied]
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing
            if missing:
                continue
            row = pd.DataFrame([[supplied[name] for name in artifact["features"]]], columns=artifact["features"])
            model = artifact["model"]
            p_pass = float(model.predict_proba(row)[0, 1])
            return Prediction(
                STATUS_OK,
                f"Predicted with the `{feature_set}` model ({artifact['model_name']}).",
                p_pass=p_pass,
                feature_set=feature_set,
                model_name=artifact["model_name"],
                inputs={name: supplied[name] for name in artifact["features"]},
                contributions=self._contributions(artifact, row, p_pass),
                test_metrics=artifact.get("test_metrics", {}),
            )
        readable = [FEATURE_DESCRIPTIONS.get(name, name) for name in (best_missing or [])]
        return Prediction(
            STATUS_INSUFFICIENT,
            "Not enough pre-snap information for any trained model. Missing: "
            + "; ".join(readable),
            inputs=supplied,
            missing=best_missing or [],
        )

    @staticmethod
    def _contributions(artifact: dict, row: pd.DataFrame, p_pass: float) -> list[dict]:
        """Counterfactual sensitivity: change in P(PASS) when one input (or the
        QB-alignment group) is reset to the training median. This explains the
        model's response around this play; it is not a causal attribution."""
        reference = artifact.get("reference_values", {})
        features = artifact["features"]
        groups: list[tuple[str, tuple[str, ...]]] = []
        if all(name in features for name in QB_GROUP):
            groups.append(("qb_alignment", QB_GROUP))
        groups += [(name, (name,)) for name in features if name not in QB_GROUP]
        rows = []
        for label, members in groups:
            if not all(name in reference for name in members):
                continue
            counterfactual = row.copy()
            for name in members:
                counterfactual[name] = reference[name]
            if counterfactual.equals(row):
                delta = 0.0
            else:
                delta = p_pass - float(artifact["model"].predict_proba(counterfactual)[0, 1])
            rows.append(
                {
                    "feature": label,
                    "value": ", ".join(f"{row[name].iloc[0]:g}" for name in members),
                    "reference": ", ".join(f"{reference[name]:g}" for name in members),
                    "delta_p_pass": round(delta, 4),
                }
            )
        return sorted(rows, key=lambda item: abs(item["delta_p_pass"]), reverse=True)
