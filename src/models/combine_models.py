"""Step 11: average the LightGBM and 1D-CNN fall probabilities and score the result.

Both models' saved out-of-fold predictions (one row per 2 s window, same order) are
averaged window by window: p = (p_LightGBM + p_CNN) / 2. The average is then scored with
exactly the same alarm rules and cut-off choice as each single model (decision_rules.py:
every cut-off chosen on the other 4 folds only).

Also writes, for the single-window rule, the share of each daily activity's recordings
with at least one false alarm, for LightGBM, the CNN and their average (Figure A).

Run from the project folder:  .venv\\Scripts\\python.exe src\\models\\combine_models.py
Writes results/combined_lgbm_cnn_predictions.csv, results/rules_combined_lgbm_cnn_predictions.csv,
       results/false_alarms_by_activity.csv
"""
import numpy as np
import pandas as pd

from decision_rules import RESULTS, evaluate, false_alarms_by_activity, load_windows

MODELS = {
    "LightGBM": RESULTS / "lgbm_tuned_top15_features_predictions.csv",
    "1D-CNN": RESULTS / "cnn_none_predictions.csv",
}
COMBINED = RESULTS / "combined_lgbm_cnn_predictions.csv"
METRICS = ["falls caught", "ADL rec. clean", "false alarms/h", "median delay s", "over budget %"]


def average_predictions():
    lgbm, cnn = (pd.read_csv(path) for path in MODELS.values())
    keys = ["trial_id", "fold", "label"]
    assert (lgbm[keys].values == cnn[keys].values).all(), "prediction rows do not line up"
    combined = lgbm[["trial_id", "user", "activity", "fold", "label"]].copy()
    combined["probability"] = (lgbm.probability + cnn.probability) / 2
    combined["threshold_sens95"] = np.nan   # cut-offs are chosen in decision_rules.evaluate
    combined.to_csv(COMBINED, index=False)
    return COMBINED


def main():
    paths = {**MODELS, "LightGBM + CNN": average_predictions()}
    summary, activities = {}, {}
    for name, path in paths.items():
        windows = load_windows(path)
        results = evaluate(windows)
        if name == "LightGBM + CNN":
            results.to_csv(RESULTS / f"rules_{path.stem}.csv", index=False)
        summary[name] = results[results.rule == "single"][METRICS].mean()
        activities[name] = false_alarms_by_activity(windows)["single"]

    pd.set_option("display.width", 200)
    print("Single-window rule, mean over the 5 folds (cut-offs chosen on the other 4 folds):")
    print(pd.DataFrame(summary).T.round(1).to_string())

    table = pd.DataFrame(activities)
    table.index.name = "activity"
    table.to_csv(RESULTS / "false_alarms_by_activity.csv")
    print("\nShare of each daily activity's recordings with >= 1 false alarm (%):")
    print(table.to_string())


if __name__ == "__main__":
    main()
