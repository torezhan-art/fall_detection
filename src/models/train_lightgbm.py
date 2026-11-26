"""Step 6: LightGBM on the hand-made features, evaluated on the same 5 user folds.

For every test fold k (= a group of users never seen in training):
  1. train LightGBM on the other 4 folds
  2. choose the decision threshold on the training users only (see below)
  3. measure all metrics on the users of fold k
  4. read the model's gain importance (feature selection, step 2 of 2)

Two decision thresholds are reported, both chosen without touching the test users:
  * "0.5"     the default cut-off on the predicted probability
  * "sens95"  the highest cut-off that still catches >= 95 % of falls (REQ-1).
              It is found with an inner split: the 4 training folds are rotated so
              every training window gets a prediction from a model that did not see
              its user, and the cut-off is chosen on those predictions.

Metrics (same definitions as baseline_threshold.py, so the two are comparable):
  per window  precision, recall (= sensitivity), F1, specificity, accuracy, ROC AUC, PR AUC
  per trial   fall trials detected / ADL trials without any false alarm (test T1)

Run from the project folder:
    .venv\\Scripts\\python.exe src\\models\\train_lightgbm.py
    .venv\\Scripts\\python.exe src\\models\\train_lightgbm.py --features data\\processed\\top15_features.txt

The default feature list is data/processed/selected_features.txt (27 features after
the correlation filter). Every output carries the name of the feature list (<list>),
so runs on different lists do not overwrite each other:
    results/lgbm_<list>_folds.csv        metrics per fold and decision threshold
    results/lgbm_<list>_importance.csv   gain importance, mean +- std over folds
    results/lgbm_<list>_predictions.csv  predicted fall probability of every window
    figures/06_lgbm_importance_<list>.png
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT / "data" / "processed" / "features.csv"
DEFAULT_FEATURE_LIST = PROJECT / "data" / "processed" / "selected_features.txt"
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"

TARGET_RECALL = 0.95    # REQ-1
SEED = 42

# Fixed in advance and NOT tuned on test folds. Small trees, because there are
# only ~660 fall windows. (tune_lightgbm.py searches around these values.)
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,           # each tree sees 80 % of the windows ...
    "subsample_freq": 1,
    "colsample_bytree": 0.8,    # ... and 80 % of the features
    "fall_weight_power": 1.0,   # fall windows weighted by (#not-fall / #fall) ** power
    "random_state": SEED,
    "verbose": -1,
}


# ---------------------------------------------------------------- training
def train_model(table, features, params=LGBM_PARAMS):
    """Fit one LightGBM model.

    Class imbalance: fall windows get the weight (#not-fall / #fall) ** fall_weight_power,
    so power 1 = fully balanced, 0.5 = halfway (square root), 0 = no extra weight.
    """
    params = dict(params)
    power = params.pop("fall_weight_power")
    y = table.label
    imbalance = (y == 0).sum() / (y == 1).sum()
    model = lgb.LGBMClassifier(**params, scale_pos_weight=imbalance ** power)
    model.fit(table[features], y)
    return model


def fall_probability(model, table, features):
    return model.predict_proba(table[features])[:, 1]


def threshold_for_recall(probabilities, labels, target=TARGET_RECALL):
    """Highest cut-off that still flags at least `target` of the fall windows."""
    fall_scores = np.sort(probabilities[labels == 1])[::-1]       # highest first
    needed = int(np.ceil(target * len(fall_scores)))              # falls we must catch
    return fall_scores[needed - 1]


def out_of_fold_probability(table, features, params=LGBM_PARAMS):
    """Rotate the folds inside `table`: every window is predicted by a model
    that never saw its fold (= its users)."""
    probability = np.zeros(len(table))
    for fold in sorted(table.fold.unique()):
        held_out = (table.fold == fold).to_numpy()
        model = train_model(table[~held_out], features, params)
        probability[held_out] = fall_probability(model, table[held_out], features)
    return probability


def choose_sens95_threshold(train, features, params=LGBM_PARAMS):
    """Pick the sens95 cut-off using only the training users (inner split by fold)."""
    inner_probability = out_of_fold_probability(train, features, params)
    return threshold_for_recall(inner_probability, train.label.to_numpy())


# ---------------------------------------------------------------- metrics
# The first three functions are copied unchanged from baseline_threshold.py, so the
# threshold baseline and LightGBM are scored with exactly the same definitions.
def sens_spec(y, fired):
    sens = fired[y == 1].mean()
    spec = 1 - fired[y == 0].mean()
    return sens, spec


def precision_recall_f1(y, fired):
    """Precision = TP/(TP+FP), recall = TP/(TP+FN) (same as sensitivity), F1 = their harmonic mean."""
    tp = np.sum(fired & (y == 1))
    fp = np.sum(fired & (y == 0))
    fn = np.sum(~fired & (y == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def trial_metrics(meta, fired):
    """Trial-level sensitivity (falls detected) and specificity (ADL trials without false alarm)."""
    df = meta.assign(fired=fired)
    falls = df[df.fall_trial & (df.label == 1)].groupby("trial_id").fired.any()
    adls = df[~df.fall_trial].groupby("trial_id").fired.any()
    return falls.mean(), 1 - adls.mean()


def window_and_trial_metrics(test, probability, threshold):
    """All agreed metrics for one test fold at one decision threshold."""
    y = test.label.to_numpy()
    fired = probability >= threshold
    precision, recall, f1 = precision_recall_f1(y, fired)
    _, specificity = sens_spec(y, fired)
    trial_sens, trial_spec = trial_metrics(test, fired)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "specificity": specificity,
        "accuracy": (fired == (y == 1)).mean(),
        "ROC AUC": roc_auc_score(y, probability),
        "PR AUC": average_precision_score(y, probability),
        "trial sens": trial_sens,
        "trial spec": trial_spec,
    }


def gain_importance(model, features):
    """Share [%] of the total gain each feature brought to this model."""
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(100 * gain / gain.sum(), index=features)


# ---------------------------------------------------------------- the 5 folds
def cross_validate(table, features):
    fold_rows, importances = [], {}
    predictions = table[["trial_id", "user", "activity", "fold", "label"]].copy()
    predictions["probability"] = np.nan
    predictions["threshold_sens95"] = np.nan

    for k in sorted(table.fold.unique()):
        is_test = (table.fold == k).to_numpy()
        train, test = table[~is_test], table[is_test]

        model = train_model(train, features)
        probability = fall_probability(model, test, features)
        sens95 = choose_sens95_threshold(train, features)

        for rule, threshold in [("0.5", 0.5), ("sens95", sens95)]:
            fold_rows.append({"fold": k, "rule": rule,
                              **window_and_trial_metrics(test, probability, threshold)})
        importances[k] = gain_importance(model, features)
        predictions.loc[is_test, "probability"] = probability
        predictions.loc[is_test, "threshold_sens95"] = sens95
        print(f"  fold {k}: trained on {len(train)} windows, tested on {len(test)} "
              f"(sens95 cut-off {sens95:.3f})")

    return pd.DataFrame(fold_rows), pd.DataFrame(importances), predictions


# ---------------------------------------------------------------- reporting
def print_fold_table(folds, rule):
    part = folds[folds.rule == rule].drop(columns="rule").set_index("fold")
    shown = (100 * part.drop(columns="threshold")).round(1)
    shown.insert(0, "cut-off", part.threshold.round(3))
    print(shown.to_string())
    mean, std = 100 * part.mean(), 100 * part.std()
    print("mean +- std:  " + "   ".join(f"{name} {mean[name]:.1f} +- {std[name]:.1f}"
                                         for name in part.columns if name != "threshold"))


def false_alarms_by_activity(predictions):
    """Share [%] of each daily activity's windows flagged at the sens95 cut-off."""
    adl = predictions[predictions.activity.str.startswith("D")]
    fired = adl.probability >= adl.threshold_sens95
    return (100 * fired.groupby(adl.activity).mean()).round(1).sort_values(ascending=False)


def plot_importance(summary, path):
    """Horizontal bars: mean gain share over the 5 folds, with +-1 std whiskers."""
    summary = summary.sort_values("mean")                          # biggest on top
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(summary) + 1.2))
    ax.barh(summary.index, summary["mean"], xerr=summary["std"], height=0.6,
            color="#2a78d6", error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 0})
    for y, (value, spread) in enumerate(zip(summary["mean"], summary["std"])):
        ax.text(value + spread + 0.3, y, f"{value:.1f}", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Share of total gain [%]  (mean over 5 folds, line = +-1 std)", color="#52514e")
    ax.set_title("LightGBM feature importance (gain)", loc="left", fontsize=12, color="#0b0b0b")
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.tick_params(axis="x", colors="#52514e")
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_LIST,
                        help="text file with one feature name per line")
    args = parser.parse_args()

    table = pd.read_csv(FEATURES_CSV)
    features = args.features.read_text().split()
    print(f"LightGBM on {len(features)} features from {args.features.name}, "
          f"{len(table)} windows, {table.fold.nunique()} user folds")

    folds, importances, predictions = cross_validate(table, features)

    pd.set_option("display.width", 220)
    for rule in ["0.5", "sens95"]:
        print(f"\n=== decision threshold: {rule} ===  (all values in %, except the cut-off)")
        print_fold_table(folds, rule)

    print("\nFalse alarms per daily activity at the sens95 cut-off (% of that activity's windows):")
    print(false_alarms_by_activity(predictions).to_string())

    importance = pd.DataFrame({"mean": importances.mean(axis=1), "std": importances.std(axis=1)})
    importance = importance.sort_values("mean", ascending=False).round(2)
    print("\nGain importance [%], mean +- std over the 5 folds:")
    print(importance.to_string())

    name = args.features.stem                      # e.g. "selected_features"
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    folds.to_csv(RESULTS / f"lgbm_{name}_folds.csv", index=False)
    importance.to_csv(RESULTS / f"lgbm_{name}_importance.csv")
    predictions.to_csv(RESULTS / f"lgbm_{name}_predictions.csv", index=False)
    chart = FIGURES / f"06_lgbm_importance_{name}.png"
    plot_importance(importance, chart)
    print(f"\nSaved results/lgbm_{name}_*.csv and {chart.relative_to(PROJECT)}")
