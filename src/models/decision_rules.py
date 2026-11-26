"""Step 8: test alarm rules on top of a trained model's saved predictions.

The model gives a fall probability for every 2 s window (one new window per second).
An alarm rule decides when the watch actually raises an alert:

  single       alarm when one window is above the cut-off              (current rule)
  2-in-a-row   alarm when two consecutive windows are both above it    (fewer false alarms,
                                                                       but 1 s later)
  hybrid       alarm at once if one window is very confident, otherwise wait for 2-in-a-row

Every fall's impact lies in exactly two consecutive windows, so a fall can pass the
2-in-a-row rule, while a short jump or table hit often lights up only one window.

Cut-offs: for each test fold, every cut-off is chosen on the OTHER 4 folds' predictions
(those labels only, never the test fold's), so that 95 % of fall recordings are caught:
  single / 2-in-a-row   the highest cut-off that still catches 95 % of fall recordings
  hybrid                "very confident" = lowest cut-off where >= 90 % of single-window
                        alarms are real falls; then the 2-in-a-row cut-off is set so the
                        rule as a whole catches 95 % of fall recordings

Scored per recording, like test T1, plus:
  false alarms / hour   alarms in daily-activity recordings per hour of recording
                        (only 1.85 h of scripted, deliberately hard activities, so this
                        is a rough stress-test rate, not a real-life rate)
  delay                 seconds from the impact to the end of the window that raised
                        the alarm (the watch can only decide once that window is complete)
  over budget           share of caught falls whose delay exceeds DETECTION_BUDGET_S

Run from the project folder:
    .venv\\Scripts\\python.exe src\\models\\decision_rules.py
    .venv\\Scripts\\python.exe src\\models\\decision_rules.py --predictions results\\lgbm_top15_features_predictions.csv
Writes results/rules_<predictions name>.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
DEFAULT_PREDICTIONS = RESULTS / "lgbm_tuned_top15_features_predictions.csv"

FS = 50                     # Hz
WINDOW_S, HOP_S = 2.0, 1.0
TARGET_RECALL = 0.95        # share of fall recordings that must be caught (REQ-1)
CONFIDENT_PRECISION = 0.90  # hybrid: one window counts as "very confident" above this precision
# REQ-2 allows 3.0 s from impact to the Security desk. The gateway needs <= 1.0 s (IR-2)
# and the BLE advertising burst <= 0.5 s, which leaves 1.5 s for the watch to decide.
DETECTION_BUDGET_S = 1.5


# ---------------------------------------------------------------- preparing the data
def load_windows(predictions_csv):
    """Predictions + window start time + impact time of every fall recording."""
    windows = pd.read_csv(predictions_csv)
    meta = pd.read_csv(PROCESSED / "windows_meta.csv")
    assert (windows.trial_id.values == meta.trial_id.values).all(), "rows do not line up"
    windows["t_start"] = meta.t_start
    windows["fall_trial"] = meta.fall_trial

    # Probability of the previous window, only if it is the window right before (1 s earlier)
    previous = windows.groupby("trial_id")[["t_start", "probability"]].shift(1)
    is_neighbour = np.isclose(windows.t_start - previous.t_start, HOP_S)
    windows["previous_probability"] = previous.probability.where(is_neighbour, 0.0)

    windows["impact_time"] = impact_times(windows)
    return windows


def impact_times(windows):
    """Impact time [s] of each fall recording: the largest |a| in its two fall windows."""
    X = np.load(PROCESSED / "windows.npz")["X"]
    fall_rows = np.flatnonzero(windows.label == 1)
    acc = np.linalg.norm(X[fall_rows, :, :3], axis=2)              # (n fall windows, 100)
    peaks = pd.DataFrame({
        "trial_id": windows.trial_id.values[fall_rows],
        "peak": acc.max(axis=1),
        "time": windows.t_start.values[fall_rows] + acc.argmax(axis=1) / FS,
    })
    strongest = peaks.loc[peaks.groupby("trial_id").peak.idxmax()].set_index("trial_id").time
    return windows.trial_id.map(strongest)                          # NaN for daily activities


# ---------------------------------------------------------------- the rules
def alarms(windows, rule, cutoff, confident=None):
    """True for every window at which the rule raises an alarm."""
    now = windows.probability >= cutoff
    two_in_a_row = now & (windows.previous_probability >= cutoff)
    if rule == "single":
        return now
    if rule == "2-in-a-row":
        return two_in_a_row
    if rule == "hybrid":
        return (windows.probability >= confident) | two_in_a_row
    raise ValueError(rule)


def fall_scores(windows):
    """Per fall recording, the best score each rule sees in its two fall windows.

    A recording is caught by a rule exactly when its score is >= the cut-off, which
    turns "catch 95 % of falls" into a simple sort.
    """
    falls = windows[windows.label == 1]
    return pd.DataFrame({
        "single": falls.probability,
        "2-in-a-row": np.minimum(falls.probability, falls.previous_probability),
    }).groupby(falls.trial_id).max()


def highest_cutoff_catching(scores, target=TARGET_RECALL):
    """Highest cut-off at which at least `target` of the scores are >= cut-off."""
    ordered = np.sort(scores)[::-1]
    needed = int(np.ceil(target * len(ordered)))
    return ordered[needed - 1] if needed > 0 else np.inf


def choose_cutoffs(train):
    """All cut-offs, chosen on `train` (the other 4 folds) only."""
    scores = fall_scores(train)
    single = highest_cutoff_catching(scores["single"])
    pair = highest_cutoff_catching(scores["2-in-a-row"])

    # hybrid, part 1: "very confident" single window
    precision, _, thresholds = precision_recall_curve(train.label, train.probability)
    reaching = thresholds[precision[:-1] >= CONFIDENT_PRECISION]
    # no cut-off reaches that precision (e.g. coarse int8 outputs): hybrid = plain 2-in-a-row
    confident = reaching.min() if len(reaching) else np.inf
    # hybrid, part 2: falls not already caught by a confident window need 2-in-a-row
    caught = scores["single"] >= confident
    still_needed = int(np.ceil(TARGET_RECALL * len(scores))) - caught.sum()
    hybrid_pair = highest_cutoff_catching(scores.loc[~caught, "2-in-a-row"],
                                          target=still_needed / max((~caught).sum(), 1))

    return {"single": (single, None), "2-in-a-row": (pair, None), "hybrid": (hybrid_pair, confident)}


# ---------------------------------------------------------------- scoring
def score_rule(test, fired):
    """Per-recording metrics, false alarms per hour and delay for one test fold."""
    test = test.assign(fired=fired)
    falls = test[test.label == 1]
    adl = test[~test.fall_trial]

    caught = falls.groupby("trial_id").fired.any()
    first_alarm = falls[falls.fired].groupby("trial_id").agg(end=("t_start", "min"),
                                                              impact=("impact_time", "first"))
    delay = first_alarm.end + WINDOW_S - first_alarm.impact

    # count alarm onsets: an alarm that stays on for several windows is one alert
    previous_fired = adl.groupby("trial_id").fired.shift(1, fill_value=False)
    onsets = (adl.fired & ~previous_fired).sum()
    adl_hours = len(adl) * HOP_S / 3600

    return {
        "falls caught": 100 * caught.mean(),
        "ADL rec. clean": 100 * (1 - adl.groupby("trial_id").fired.any().mean()),
        "false alarms/h": onsets / adl_hours,
        "median delay s": delay.median(),
        "max delay s": delay.max(),
        "over budget %": 100 * (delay > DETECTION_BUDGET_S).mean(),
    }


def evaluate(windows):
    rows = []
    for k in sorted(windows.fold.unique()):
        test, train = windows[windows.fold == k], windows[windows.fold != k]
        for rule, (cutoff, confident) in choose_cutoffs(train).items():
            fired = alarms(test, rule, cutoff, confident)
            rows.append({"fold": int(k), "rule": rule, "cut-off": cutoff,
                         "confident cut-off": confident, **score_rule(test, fired)})
    return pd.DataFrame(rows)


def false_alarms_by_activity(windows):
    """Share [%] of each activity's recordings with at least one alarm, per rule."""
    table = {}
    for rule in ["single", "2-in-a-row", "hybrid"]:
        fired = pd.Series(False, index=windows.index)
        for k in sorted(windows.fold.unique()):
            is_test = windows.fold == k
            cutoff, confident = choose_cutoffs(windows[~is_test])[rule]
            fired[is_test] = alarms(windows[is_test], rule, cutoff, confident)
        adl = windows[~windows.fall_trial].assign(fired=fired)
        per_recording = adl.groupby(["activity", "trial_id"]).fired.any()
        table[rule] = (100 * per_recording.groupby("activity").mean()).round(1)
    return pd.DataFrame(table).sort_values("single", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    windows = load_windows(args.predictions)
    results = evaluate(windows)

    pd.set_option("display.width", 220, "display.max_columns", None)
    print(f"Alarm rules on {args.predictions.name} "
          f"(cut-offs chosen on the other 4 folds, target: {TARGET_RECALL:.0%} of fall recordings)\n")
    metrics = ["falls caught", "ADL rec. clean", "false alarms/h",
               "median delay s", "max delay s", "over budget %"]
    mean = results.groupby("rule", sort=False)[metrics].mean()
    std = results.groupby("rule", sort=False)[metrics].std()
    print("Mean over the 5 folds (std in brackets):")
    print(mean.round(2).astype(str) + " (" + std.round(1).astype(str) + ")")
    print(f"\n'over budget' = caught falls detected later than {DETECTION_BUDGET_S} s after "
          f"impact (3.0 s REQ-2 minus 1.0 s gateway minus 0.5 s BLE burst)")

    print("\nShare of each daily activity's recordings with at least one alarm (%):")
    print(false_alarms_by_activity(windows).to_string())

    out = RESULTS / f"rules_{args.predictions.stem}.csv"
    results.to_csv(out, index=False)
    print(f"\nPer-fold results saved to {out.relative_to(PROJECT)}")
