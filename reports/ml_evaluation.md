# RUN/PASS model evaluation

Generated 2026-10-12 17:47 UTC by `python -m nfl_analytics.ml.train`.

* Train seasons: [2016, 2017, 2018, 2019, 2020, 2021, 2022]
* Validation season (model selection): [2023]
* Test seasons (held out, reported once): [2024, 2025]
* Positive class: PASS (dropbacks, including sacks and scrambles)

## Feature set `vision`

Features: qb_shotgun, qb_pistol, qb_under_center, defenders_in_box

Rows: train 237,686 / validation 35,294 / test 69,110. Selected: **gradient_boosting** (lowest validation log loss).

| Model | Test acc | Precision (pass) | Recall (pass) | F1 (pass) | F1 macro | ROC AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| baseline_majority | 0.604 | 0.604 | 1.000 | 0.753 | 0.377 | 0.500 | 0.672 | 0.239 | 0.011 |
| baseline_qb_alignment | 0.761 | 0.796 | 0.812 | 0.804 | 0.749 | 0.745 | 0.547 | 0.181 | 0.022 |
| logistic_regression | 0.762 | 0.791 | 0.823 | 0.807 | 0.749 | 0.784 | 0.541 | 0.180 | 0.055 |
| random_forest | 0.760 | 0.794 | 0.814 | 0.804 | 0.747 | 0.782 | 0.533 | 0.177 | 0.035 |
| gradient_boosting ✔ | 0.760 | 0.794 | 0.814 | 0.804 | 0.747 | 0.782 | 0.533 | 0.177 | 0.034 |

Confusion matrix for the selected model (rows actual, columns predicted): RUN→RUN 18,538, RUN→PASS 8,829, PASS→RUN 7,766, PASS→PASS 33,977.

Permutation importance on test (increase in log loss): qb_shotgun 0.1601, defenders_in_box 0.0464, qb_under_center 0.0005, qb_pistol -0.0007

## Feature set `vision_situation`

Features: qb_shotgun, qb_pistol, qb_under_center, defenders_in_box, down, ydstogo, yardline_100, goal_to_go, qtr, game_seconds_remaining, half_seconds_remaining, score_differential, posteam_timeouts_remaining, defteam_timeouts_remaining, no_huddle

Rows: train 237,686 / validation 35,294 / test 69,110. Selected: **gradient_boosting** (lowest validation log loss).

| Model | Test acc | Precision (pass) | Recall (pass) | F1 (pass) | F1 macro | ROC AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| baseline_majority | 0.604 | 0.604 | 1.000 | 0.753 | 0.377 | 0.500 | 0.672 | 0.239 | 0.011 |
| baseline_down_distance | 0.644 | 0.639 | 0.942 | 0.762 | 0.529 | 0.712 | 0.590 | 0.205 | 0.016 |
| baseline_qb_alignment | 0.761 | 0.796 | 0.812 | 0.804 | 0.749 | 0.745 | 0.547 | 0.181 | 0.022 |
| logistic_regression | 0.753 | 0.775 | 0.832 | 0.803 | 0.736 | 0.810 | 0.522 | 0.173 | 0.038 |
| random_forest | 0.767 | 0.803 | 0.814 | 0.809 | 0.756 | 0.838 | 0.483 | 0.160 | 0.017 |
| gradient_boosting ✔ | 0.768 | 0.805 | 0.814 | 0.809 | 0.757 | 0.842 | 0.477 | 0.158 | 0.012 |

Confusion matrix for the selected model (rows actual, columns predicted): RUN→RUN 19,145, RUN→PASS 8,222, PASS→RUN 7,782, PASS→PASS 33,961.

Permutation importance on test (increase in log loss): qb_shotgun 0.0917, down 0.0857, ydstogo 0.0723, score_differential 0.0271, defenders_in_box 0.0217, game_seconds_remaining 0.0198

## Feature set `qb_situation`

Features: qb_shotgun, qb_pistol, qb_under_center, down, ydstogo, yardline_100, goal_to_go, qtr, game_seconds_remaining, half_seconds_remaining, score_differential, posteam_timeouts_remaining, defteam_timeouts_remaining, no_huddle

Rows: train 237,917 / validation 35,378 / test 69,201. Selected: **gradient_boosting** (lowest validation log loss).

| Model | Test acc | Precision (pass) | Recall (pass) | F1 (pass) | F1 macro | ROC AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| baseline_majority | 0.604 | 0.604 | 1.000 | 0.753 | 0.377 | 0.500 | 0.671 | 0.239 | 0.011 |
| baseline_down_distance | 0.644 | 0.639 | 0.942 | 0.762 | 0.529 | 0.712 | 0.590 | 0.205 | 0.016 |
| baseline_qb_alignment | 0.761 | 0.796 | 0.812 | 0.804 | 0.749 | 0.745 | 0.547 | 0.181 | 0.022 |
| logistic_regression | 0.757 | 0.786 | 0.821 | 0.803 | 0.742 | 0.805 | 0.524 | 0.173 | 0.024 |
| random_forest | 0.767 | 0.806 | 0.810 | 0.808 | 0.756 | 0.835 | 0.487 | 0.161 | 0.016 |
| gradient_boosting ✔ | 0.768 | 0.806 | 0.810 | 0.808 | 0.757 | 0.838 | 0.482 | 0.160 | 0.010 |

Confusion matrix for the selected model (rows actual, columns predicted): RUN→RUN 19,237, RUN→PASS 8,145, PASS→RUN 7,931, PASS→PASS 33,888.

Permutation importance on test (increase in log loss): qb_shotgun 0.1108, down 0.0931, ydstogo 0.0875, score_differential 0.0352, game_seconds_remaining 0.0200, half_seconds_remaining 0.0182

## Feature set `situation`

Features: down, ydstogo, yardline_100, goal_to_go, qtr, game_seconds_remaining, half_seconds_remaining, score_differential, posteam_timeouts_remaining, defteam_timeouts_remaining, no_huddle

Rows: train 237,917 / validation 35,378 / test 69,201. Selected: **gradient_boosting** (lowest validation log loss).

| Model | Test acc | Precision (pass) | Recall (pass) | F1 (pass) | F1 macro | ROC AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| baseline_majority | 0.604 | 0.604 | 1.000 | 0.753 | 0.377 | 0.500 | 0.671 | 0.239 | 0.011 |
| baseline_down_distance | 0.644 | 0.639 | 0.942 | 0.762 | 0.529 | 0.712 | 0.590 | 0.205 | 0.016 |
| logistic_regression | 0.675 | 0.704 | 0.797 | 0.748 | 0.646 | 0.731 | 0.592 | 0.203 | 0.019 |
| random_forest | 0.710 | 0.753 | 0.774 | 0.763 | 0.694 | 0.790 | 0.529 | 0.180 | 0.012 |
| gradient_boosting ✔ | 0.715 | 0.772 | 0.749 | 0.760 | 0.704 | 0.795 | 0.524 | 0.178 | 0.009 |

Confusion matrix for the selected model (rows actual, columns predicted): RUN→RUN 18,117, RUN→PASS 9,265, PASS→RUN 10,488, PASS→PASS 31,331.

Permutation importance on test (increase in log loss): down 0.2137, ydstogo 0.1521, score_differential 0.0740, half_seconds_remaining 0.0367, game_seconds_remaining 0.0331, yardline_100 0.0211

## Feature set `full`

Features: qb_shotgun, qb_pistol, qb_under_center, defenders_in_box, down, ydstogo, yardline_100, goal_to_go, qtr, game_seconds_remaining, half_seconds_remaining, score_differential, posteam_timeouts_remaining, defteam_timeouts_remaining, no_huddle, off_rb, off_te, off_wr, off_ol, def_dl, def_lb, def_db

Rows: train 237,686 / validation 35,294 / test 69,110. Selected: **gradient_boosting** (lowest validation log loss).

| Model | Test acc | Precision (pass) | Recall (pass) | F1 (pass) | F1 macro | ROC AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| baseline_majority | 0.604 | 0.604 | 1.000 | 0.753 | 0.377 | 0.500 | 0.672 | 0.239 | 0.011 |
| baseline_down_distance | 0.644 | 0.639 | 0.942 | 0.762 | 0.529 | 0.712 | 0.590 | 0.205 | 0.016 |
| baseline_qb_alignment | 0.761 | 0.796 | 0.812 | 0.804 | 0.749 | 0.745 | 0.547 | 0.181 | 0.022 |
| logistic_regression | 0.752 | 0.772 | 0.836 | 0.803 | 0.734 | 0.811 | 0.522 | 0.173 | 0.039 |
| random_forest | 0.766 | 0.801 | 0.816 | 0.808 | 0.754 | 0.839 | 0.483 | 0.160 | 0.021 |
| gradient_boosting ✔ | 0.769 | 0.802 | 0.820 | 0.811 | 0.757 | 0.843 | 0.476 | 0.158 | 0.015 |

Confusion matrix for the selected model (rows actual, columns predicted): RUN→RUN 18,905, RUN→PASS 8,462, PASS→RUN 7,506, PASS→PASS 34,237.

Permutation importance on test (increase in log loss): qb_shotgun 0.0980, down 0.0862, ydstogo 0.0721, defenders_in_box 0.0282, score_differential 0.0250, game_seconds_remaining 0.0192

