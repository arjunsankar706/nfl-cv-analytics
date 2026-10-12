"""Classification and calibration metrics, plus simple reference baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        members = indices == index
        if members.any():
            error += members.mean() * abs(y_true[members].mean() - probability[members].mean())
    return float(error)


def reliability_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    rows = []
    for index in range(bins):
        members = indices == index
        if members.any():
            rows.append(
                {
                    "bin_low": round(float(edges[index]), 2),
                    "bin_high": round(float(edges[index + 1]), 2),
                    "count": int(members.sum()),
                    "mean_predicted": round(float(probability[members].mean()), 4),
                    "observed_pass_rate": round(float(y_true[members].mean()), 4),
                }
            )
    return rows


def classification_metrics(y_true, probability, threshold: float = 0.5) -> dict:
    """Metrics with PASS (1) as the positive class."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    predicted = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    has_both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "pass_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "precision_pass": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall_pass": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1_pass": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "precision_run": round(
            float(precision_score(y_true, predicted, pos_label=0, zero_division=0)), 4
        ),
        "recall_run": round(float(recall_score(y_true, predicted, pos_label=0, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, predicted, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4) if has_both else None,
        "log_loss": round(float(log_loss(y_true, probability, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_true, probability)), 4),
        "ece": round(expected_calibration_error(y_true, probability), 4),
        # rows = actual [RUN, PASS], columns = predicted [RUN, PASS]
        "confusion_matrix": matrix.tolist(),
    }


class MajorityBaseline:
    """Always predicts the training PASS rate."""

    name = "baseline_majority"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityBaseline":
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rate = np.full(len(X), self.rate_)
        return np.column_stack([1 - rate, rate])


class LookupBaseline:
    """Empirical PASS rate per group of a few columns (e.g. down x distance)."""

    def __init__(self, name: str, columns: list[str], bucket_distance: bool = False, min_count: int = 30):
        self.name = name
        self.columns = columns
        self.bucket_distance = bucket_distance
        self.min_count = min_count

    def _keys(self, X: pd.DataFrame) -> pd.Series:
        parts = []
        for column in self.columns:
            values = X[column]
            if self.bucket_distance and column == "ydstogo":
                values = pd.cut(values, [0, 1, 2, 3, 6, 10, 15, 100], labels=False)
            parts.append(values.astype(int).astype(str))
        return pd.Series(["|".join(row) for row in zip(*parts)], index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LookupBaseline":
        keys = self._keys(X)
        stats = pd.DataFrame({"key": keys, "y": np.asarray(y)}).groupby("key")["y"].agg(["mean", "count"])
        self.table_ = stats.loc[stats["count"] >= self.min_count, "mean"].to_dict()
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rate = np.array([self.table_.get(key, self.rate_) for key in self._keys(X)])
        return np.column_stack([1 - rate, rate])
