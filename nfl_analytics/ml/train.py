"""Train and evaluate pre-snap RUN/PASS classifiers on nflverse data.

Season-based split (no game appears in more than one split):

* train:      2016-2022 (model fitting)
* validation: 2023      (model selection by log loss)
* test:       2024-2025 (reported once; never used for selection)

The selected model for each feature set is refit on train+validation and
saved to ``models/``. Metrics are written to ``reports/``.

    python -m nfl_analytics.ml.train --download
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import SOURCES, download, load_seasons
from .evaluate import (
    LookupBaseline,
    MajorityBaseline,
    classification_metrics,
    reliability_table,
)
from .features import FEATURE_SETS, build_features, feature_matrix

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 7

# numpy 2.0 linked against Apple Accelerate emits spurious "divide by zero /
# overflow encountered in matmul" warnings even for well-conditioned inputs
# (reproducible with LogisticRegression on standard-normal data). Results are
# finite and correct, so only that specific message is silenced.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
CATEGORICAL = ("down", "qtr")


def make_models(features: list[str]) -> dict[str, object]:
    categorical = [name for name in features if name in CATEGORICAL]
    numeric = [name for name in features if name not in CATEGORICAL]
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            *(
                [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)]
                if categorical
                else []
            ),
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [("preprocess", preprocess), ("model", LogisticRegression(C=1.0, max_iter=2000))]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=25,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    }


def make_baselines(features: list[str]) -> dict[str, object]:
    baselines: dict[str, object] = {"baseline_majority": MajorityBaseline()}
    if "down" in features:
        baselines["baseline_down_distance"] = LookupBaseline(
            "baseline_down_distance", ["down", "ydstogo"], bucket_distance=True
        )
    if "qb_shotgun" in features:
        baselines["baseline_qb_alignment"] = LookupBaseline(
            "baseline_qb_alignment", ["qb_shotgun", "qb_pistol"]
        )
    return baselines


def parse_seasons(text: str) -> list[int]:
    seasons: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = part.split("-")
            seasons.extend(range(int(start), int(end) + 1))
        elif part.strip():
            seasons.append(int(part))
    return seasons


def evaluate_feature_set(
    feature_set: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, object, str]:
    features = FEATURE_SETS[feature_set]
    X_train, y_train = feature_matrix(train, feature_set)
    X_val, y_val = feature_matrix(validation, feature_set)
    X_test, y_test = feature_matrix(test, feature_set)
    X_final = pd.concat([X_train, X_val])
    y_final = pd.concat([y_train, y_val])

    results: dict[str, dict] = {}
    candidates = {**make_baselines(features), **make_models(features)}
    for name, estimator in candidates.items():
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        val_metrics = classification_metrics(y_val, estimator.predict_proba(X_val)[:, 1])
        fit_seconds = time.perf_counter() - started
        results[name] = {"validation": val_metrics, "fit_seconds_train": round(fit_seconds, 2)}
        print(f"  [{feature_set}] {name:<24} val acc={val_metrics['accuracy']:.4f} "
              f"logloss={val_metrics['log_loss']:.4f} ({fit_seconds:.1f}s)")

    learned = [name for name in candidates if not name.startswith("baseline")]
    selected = min(learned, key=lambda name: results[name]["validation"]["log_loss"])

    # Refit every candidate on train+validation and score the untouched test seasons.
    final_estimators = {**make_baselines(features), **make_models(features)}
    test_probabilities = {}
    for name, estimator in final_estimators.items():
        estimator.fit(X_final, y_final)
        probability = estimator.predict_proba(X_test)[:, 1]
        test_probabilities[name] = probability
        results[name]["test"] = classification_metrics(y_test, probability)

    selected_model = final_estimators[selected]
    sample = X_test.sample(n=min(6000, len(X_test)), random_state=RANDOM_STATE)
    importance = permutation_importance(
        selected_model,
        sample,
        y_test.loc[sample.index],
        scoring="neg_log_loss",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_importance = sorted(
        (
            {"feature": name, "log_loss_increase": round(float(mean), 5), "std": round(float(std), 5)}
            for name, mean, std in zip(features, importance.importances_mean, importance.importances_std)
        ),
        key=lambda row: row["log_loss_increase"],
        reverse=True,
    )
    summary = {
        "features": features,
        "rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "selected_model": selected,
        "selection_metric": "validation log loss",
        "models": results,
        "test_reliability_selected": reliability_table(
            y_test.to_numpy(), test_probabilities[selected]
        ),
        "permutation_importance_test": feature_importance,
        "reference_values": {
            # Medians/modes of the training data, used only by the explanation
            # routine as a neutral counterfactual; never used to fill inputs.
            name: float(X_final[name].median()) for name in features
        },
    }
    return summary, selected_model, selected


def write_reliability_plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 5.0), dpi=130)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="perfect")
    colors = {"vision": "#2b8cbe", "vision_situation": "#e6550d", "situation": "#31a354", "full": "#756bb1"}
    for feature_set, summary in report["feature_sets"].items():
        rows = summary["test_reliability_selected"]
        axis.plot(
            [row["mean_predicted"] for row in rows],
            [row["observed_pass_rate"] for row in rows],
            marker="o",
            markersize=3.5,
            color=colors.get(feature_set),
            label=f"{feature_set} ({summary['selected_model']})",
        )
    axis.set_xlabel("Predicted P(PASS)")
    axis.set_ylabel("Observed PASS rate")
    axis.set_title("Test-season calibration (2024-2025)")
    axis.legend(fontsize=7, loc="upper left")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# RUN/PASS model evaluation",
        "",
        f"Generated {report['created_utc']} by `python -m nfl_analytics.ml.train`.",
        "",
        f"* Train seasons: {report['split']['train']}",
        f"* Validation season (model selection): {report['split']['validation']}",
        f"* Test seasons (held out, reported once): {report['split']['test']}",
        "* Positive class: PASS (dropbacks, including sacks and scrambles)",
        "",
    ]
    for feature_set, summary in report["feature_sets"].items():
        lines += [
            f"## Feature set `{feature_set}`",
            "",
            f"Features: {', '.join(summary['features'])}",
            "",
            f"Rows: train {summary['rows']['train']:,} / validation {summary['rows']['validation']:,} "
            f"/ test {summary['rows']['test']:,}. Selected: **{summary['selected_model']}** "
            "(lowest validation log loss).",
            "",
            "| Model | Test acc | Precision (pass) | Recall (pass) | F1 (pass) | F1 macro | ROC AUC | Log loss | Brier | ECE |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, result in summary["models"].items():
            metrics = result["test"]
            marker = " ✔" if name == summary["selected_model"] else ""
            auc = "-" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}"
            lines.append(
                f"| {name}{marker} | {metrics['accuracy']:.3f} | {metrics['precision_pass']:.3f} | "
                f"{metrics['recall_pass']:.3f} | {metrics['f1_pass']:.3f} | {metrics['f1_macro']:.3f} | "
                f"{auc} | {metrics['log_loss']:.3f} | {metrics['brier']:.3f} | {metrics['ece']:.3f} |"
            )
        matrix = summary["models"][summary["selected_model"]]["test"]["confusion_matrix"]
        lines += [
            "",
            f"Confusion matrix for the selected model (rows actual, columns predicted): "
            f"RUN→RUN {matrix[0][0]:,}, RUN→PASS {matrix[0][1]:,}, "
            f"PASS→RUN {matrix[1][0]:,}, PASS→PASS {matrix[1][1]:,}.",
            "",
            "Permutation importance on test (increase in log loss): "
            + ", ".join(
                f"{row['feature']} {row['log_loss_increase']:.4f}"
                for row in summary["permutation_importance_test"][:6]
            ),
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-seasons", default="2016-2022")
    parser.add_argument("--val-season", default="2023")
    parser.add_argument("--test-seasons", default="2024-2025")
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS))
    parser.add_argument("--download", action="store_true", help="fetch missing nflverse files")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)

    train_seasons = parse_seasons(args.train_seasons)
    val_seasons = parse_seasons(args.val_season)
    test_seasons = parse_seasons(args.test_seasons)
    overlap = set(train_seasons) & set(val_seasons) | set(train_seasons) & set(test_seasons) | set(val_seasons) & set(test_seasons)
    if overlap:
        raise SystemExit(f"Seasons appear in more than one split: {sorted(overlap)}")
    all_seasons = train_seasons + val_seasons + test_seasons
    if args.download:
        for season in all_seasons:
            for kind in SOURCES:
                download(kind, season)

    print("Loading and engineering features...")
    train = build_features(load_seasons(train_seasons))
    validation = build_features(load_seasons(val_seasons))
    test = build_features(load_seasons(test_seasons))
    shared_games = set(train["game_id"]) & set(test["game_id"]) | set(validation["game_id"]) & set(test["game_id"])
    assert not shared_games, "game leakage across splits"

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sklearn_version": sklearn.__version__,
        "split": {"train": train_seasons, "validation": val_seasons, "test": test_seasons},
        "data_source": "nflverse-data play_by_play + pbp_participation release assets",
        "feature_sets": {},
    }
    for feature_set in [name.strip() for name in args.feature_sets.split(",") if name.strip()]:
        print(f"Feature set: {feature_set}")
        summary, model, selected = evaluate_feature_set(feature_set, train, validation, test)
        report["feature_sets"][feature_set] = summary
        joblib.dump(
            {
                "model": model,
                "model_name": selected,
                "feature_set": feature_set,
                "features": FEATURE_SETS[feature_set],
                "trained_on_seasons": train_seasons + val_seasons,
                "test_seasons": test_seasons,
                "test_metrics": summary["models"][selected]["test"],
                "reference_values": summary["reference_values"],
                "sklearn_version": sklearn.__version__,
                "created_utc": report["created_utc"],
            },
            model_dir / f"run_pass_{feature_set}.joblib",
            compress=3,
        )
        selected_test = summary["models"][selected]["test"]
        print(f"  -> selected {selected}: test acc={selected_test['accuracy']:.4f} "
              f"auc={selected_test['roc_auc']} logloss={selected_test['log_loss']:.4f}")

    (report_dir / "ml_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, report_dir / "ml_evaluation.md")
    write_reliability_plot(report, report_dir / "calibration_test.png")
    print(f"Wrote {report_dir / 'ml_evaluation.md'} and models in {model_dir}")


if __name__ == "__main__":
    main()
