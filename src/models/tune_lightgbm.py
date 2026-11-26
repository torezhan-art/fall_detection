"""Step 7: tune LightGBM's settings with a nested random search.

Why nested: if we picked the settings that score best on the test users and then
reported that score, the result would be too optimistic. So there are two loops:

  outer loop (5 user folds)  only MEASURES the result, exactly like train_lightgbm.py
  inner loop (inside the 4 training folds of each outer fold)
                             CHOOSES the settings: every candidate is trained on
                             3 training folds and checked on the 4th, rotating

Score used to compare candidates: precision at 95 % recall, i.e. "when we catch
95 % of falls (REQ-1), what share of the alarms are real falls?" Higher is better.

Candidates: 50 random combinations of settings + the fixed settings of
train_lightgbm.py as candidate 0, so the search can only match or beat them
inside the training data. The same candidates are tried in every outer fold.

Finally, the same search is run once over all users (5-fold) to pick the
settings to keep for the final model. That choice is not used for any reported score.

Run from the project folder:
    .venv\\Scripts\\python.exe src\\models\\tune_lightgbm.py
    .venv\\Scripts\\python.exe src\\models\\tune_lightgbm.py --features data\\processed\\selected_features.txt
Writes results/lgbm_tuned_<list>_folds.csv        metrics per outer fold (to compare with untuned)
       results/lgbm_tuned_<list>_search.csv       inner score of every candidate in every outer fold
       results/lgbm_tuned_<list>_predictions.csv  predicted fall probability of every window
       results/lgbm_tuned_<list>_final_params.json settings to keep for the final model
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_lightgbm import (FEATURES_CSV, LGBM_PARAMS, PROJECT, RESULTS, SEED,
                            fall_probability, false_alarms_by_activity, out_of_fold_probability,
                            print_fold_table, threshold_for_recall, train_model,
                            window_and_trial_metrics)

DEFAULT_FEATURE_LIST = PROJECT / "data" / "processed" / "top15_features.txt"
N_CANDIDATES = 50

# Search space. Every value is drawn at random once, before any data is seen.
SEARCH_SPACE = {
    "n_estimators":      lambda rng: int(rng.integers(100, 1001)),        # number of trees
    "learning_rate":     lambda rng: float(10 ** rng.uniform(-2, -1)),    # 0.01 .. 0.1, log scale
    "num_leaves":        lambda rng: int(rng.integers(7, 64)),            # tree complexity
    "min_child_samples": lambda rng: int(rng.integers(10, 101)),          # min windows per leaf
    "subsample":         lambda rng: float(rng.uniform(0.6, 1.0)),        # share of windows per tree
    "colsample_bytree":  lambda rng: float(rng.uniform(0.6, 1.0)),        # share of features per tree
    "reg_lambda":        lambda rng: float(rng.uniform(0.0, 10.0)),       # penalty on extreme leaves
    "fall_weight_power": lambda rng: float(rng.choice([0.0, 0.5, 1.0])),  # extra weight for falls
}


def make_candidates(n=N_CANDIDATES, seed=SEED):
    """Candidate 0 = the fixed settings of train_lightgbm.py, then n random ones."""
    rng = np.random.default_rng(seed)
    candidates = [dict(LGBM_PARAMS)]
    for _ in range(n):
        settings = {name: draw(rng) for name, draw in SEARCH_SPACE.items()}
        candidates.append({**settings, "subsample_freq": 1, "random_state": SEED, "verbose": -1})
    return candidates


def precision_at_recall(probability, labels):
    """Share of alarms that are real falls, at the cut-off that catches 95 % of falls."""
    cutoff = threshold_for_recall(probability, labels)
    fired = probability >= cutoff
    return (fired & (labels == 1)).sum() / fired.sum()


def search(train, features, candidates):
    """Score every candidate on `train` by rotating its folds.

    Returns a table with one row per candidate, and the out-of-fold predictions
    of every candidate (needed to set the sens95 cut-off of the winner).
    """
    labels = train.label.to_numpy()
    rows, predictions = [], []
    for number, params in enumerate(candidates):
        probability = out_of_fold_probability(train, features, params)
        rows.append({"candidate": number, "precision@95": precision_at_recall(probability, labels)})
        predictions.append(probability)
    return pd.DataFrame(rows), predictions


def nested_cross_validation(table, features, candidates):
    fold_rows, search_tables, winners = [], [], {}
    predictions = table[["trial_id", "user", "activity", "fold", "label"]].copy()
    predictions["probability"] = np.nan
    predictions["threshold_sens95"] = np.nan

    for k in sorted(table.fold.unique()):
        start = time.time()
        is_test = (table.fold == k).to_numpy()
        train, test = table[~is_test], table[is_test]

        # 1. inner loop: choose the settings on the training users only
        scores, inner_predictions = search(train, features, candidates)
        best = int(scores["precision@95"].idxmax())
        params = candidates[best]
        sens95 = threshold_for_recall(inner_predictions[best], train.label.to_numpy())

        # 2. outer loop: train the winner on all training users, measure on the test users
        model = train_model(train, features, params)
        probability = fall_probability(model, test, features)
        for rule, cutoff in [("0.5", 0.5), ("sens95", sens95)]:
            fold_rows.append({"fold": k, "rule": rule,
                              **window_and_trial_metrics(test, probability, cutoff)})

        predictions.loc[is_test, "probability"] = probability
        predictions.loc[is_test, "threshold_sens95"] = sens95
        search_tables.append(scores.assign(outer_fold=k))
        winners[int(k)] = best
        print(f"  outer fold {k}: best candidate {best} "
              f"(inner precision@95 {100 * scores['precision@95'][best]:.1f} %, "
              f"fixed settings {100 * scores['precision@95'][0]:.1f} %)  [{time.time() - start:.0f} s]")

    return pd.DataFrame(fold_rows), pd.concat(search_tables), predictions, winners


def compare_with_untuned(tuned, untuned_csv):
    """Mean of the main metrics at the sens95 cut-off: fixed vs tuned settings."""
    columns = ["recall", "precision", "specificity", "accuracy", "PR AUC", "trial sens", "trial spec"]
    rows = {"tuned (nested)": tuned}
    if untuned_csv.exists():
        rows = {"fixed settings": pd.read_csv(untuned_csv), **rows}
    table = pd.DataFrame({name: 100 * folds[folds.rule == "sens95"][columns].mean()
                          for name, folds in rows.items()}).T
    return table.round(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_LIST,
                        help="text file with one feature name per line")
    args = parser.parse_args()

    table = pd.read_csv(FEATURES_CSV)
    features = args.features.read_text().split()
    candidates = make_candidates()
    name = args.features.stem
    print(f"Nested tuning: {len(features)} features from {args.features.name}, "
          f"{len(candidates)} candidates, 5 outer x 4 inner folds")

    folds, searches, predictions, winners = nested_cross_validation(table, features, candidates)

    pd.set_option("display.width", 220)
    for rule in ["0.5", "sens95"]:
        print(f"\n=== tuned, decision threshold: {rule} ===  (all values in %, except the cut-off)")
        print_fold_table(folds, rule)

    print("\nFixed vs tuned settings, mean over the 5 outer folds at the sens95 cut-off (%):")
    print(compare_with_untuned(folds, RESULTS / f"lgbm_{name}_folds.csv").to_string())

    print("\nFalse alarms per daily activity at the sens95 cut-off (% of that activity's windows):")
    print(false_alarms_by_activity(predictions).to_string())

    # Settings to keep: the same search once more, now over all 5 folds (all users).
    print("\nChoosing the final settings on all users ...")
    final_scores, _ = search(table, features, candidates)
    final = int(final_scores["precision@95"].idxmax())
    final_params = candidates[final]
    print(f"Final choice: candidate {final} (5-fold precision@95 "
          f"{100 * final_scores['precision@95'][final]:.1f} %, fixed settings "
          f"{100 * final_scores['precision@95'][0]:.1f} %)")
    print(json.dumps(final_params, indent=2))
    print(f"Winners per outer fold: {winners}  (same number = stable choice)")

    RESULTS.mkdir(exist_ok=True)
    folds.to_csv(RESULTS / f"lgbm_tuned_{name}_folds.csv", index=False)
    searches.to_csv(RESULTS / f"lgbm_tuned_{name}_search.csv", index=False)
    predictions.to_csv(RESULTS / f"lgbm_tuned_{name}_predictions.csv", index=False)
    (RESULTS / f"lgbm_tuned_{name}_final_params.json").write_text(
        json.dumps({"candidate": final, **final_params}, indent=2))
    print(f"\nSaved results/lgbm_tuned_{name}_*")
