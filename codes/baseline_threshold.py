"""Step 3: baseline A, a peak-acceleration threshold: "fall if max |a| in the window >= T".

For every fold the threshold T is chosen on the 4 training folds and then measured
on the held-out users, exactly like a trained model. Two ways to choose T:
  * "balanced"  maximise sensitivity + specificity (Youden's J)
  * "sens95"    the highest T that still reaches 95 % sensitivity on training (REQ-1)

Metrics are reported per window and per trial (as in test T1):
  * a fall trial is detected if any of its fall windows fires
  * an ADL trial is a false alarm if any of its windows fires

Run from the project folder:  .venv\\Scripts\\python.exe src\\baseline_threshold.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
FIGS = PROJECT / "figures"

THRESHOLDS = np.round(np.arange(1.5, 8.01, 0.05), 2)   # g


def peak_acc(X):
    """Largest acceleration magnitude [g] in each window. X: (n, 100, 6), accel in columns 0-2."""
    return np.linalg.norm(X[:, :, :3], axis=2).max(axis=1)


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


def choose_threshold(peak, y, rule):
    sens = np.array([(peak[y == 1] >= t).mean() for t in THRESHOLDS])
    spec = np.array([(peak[y == 0] < t).mean() for t in THRESHOLDS])
    if rule == "balanced":
        return THRESHOLDS[np.argmax(sens + spec)]
    if rule == "sens95":
        return THRESHOLDS[sens >= 0.95].max()
    raise ValueError(rule)


def trial_metrics(meta, fired):
    """Trial-level sensitivity (falls detected) and specificity (ADL trials without false alarm)."""
    df = meta.assign(fired=fired)
    falls = df[df.fall_trial & (df.label == 1)].groupby("trial_id").fired.any()
    adls = df[~df.fall_trial].groupby("trial_id").fired.any()
    return falls.mean(), 1 - adls.mean()


def evaluate(X, y, meta, rule):
    peak = peak_acc(X)
    rows = []
    for k in sorted(meta.fold.unique()):
        train, test = (meta.fold != k).to_numpy(), (meta.fold == k).to_numpy()
        t = choose_threshold(peak[train], y[train], rule)
        fired = peak[test] >= t
        w_sens, w_spec = sens_spec(y[test], fired)
        w_prec, w_rec, w_f1 = precision_recall_f1(y[test], fired)
        t_sens, t_spec = trial_metrics(meta[test], fired)
        rows.append({"fold": k, "T [g]": t,
                     "win precision": w_prec, "win recall=sens": w_rec, "win F1": w_f1,
                     "win spec": w_spec, "win acc": (fired == (y[test] == 1)).mean(),
                     "trial sens": t_sens, "trial spec": t_spec})
    return pd.DataFrame(rows).set_index("fold")


def false_alarms_by_activity(X, y, meta, threshold):
    """Which ADLs fire at a given threshold (pooled over all users)?"""
    fired = peak_acc(X) >= threshold
    adl = meta.assign(fired=fired)[~meta.fall_trial]
    return (adl.groupby("activity").fired.mean() * 100).round(1).sort_values(ascending=False)


def plot_sweep(X, y, chosen, path):
    peak = peak_acc(X)
    sens = [(peak[y == 1] >= t).mean() * 100 for t in THRESHOLDS]
    spec = [(peak[y == 0] < t).mean() * 100 for t in THRESHOLDS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(THRESHOLDS, sens, color="#c0392b", label="sensitivity (falls caught)")
    ax.plot(THRESHOLDS, spec, color="#2c6fbb", label="specificity (ADL windows not flagged)")
    ax.axhline(95, color="grey", ls="--", lw=0.8)
    ax.text(THRESHOLDS[-1], 95.5, "REQ-1: 95 %", ha="right", va="bottom", color="grey", fontsize=9)
    t = chosen["balanced"]
    ax.axvline(t, color="black", ls=":", lw=0.8)
    ax.text(t, 3, f" chosen on training folds\n ~{t:.2f} g", fontsize=8, va="bottom")
    cross = int(np.argmin(np.abs(np.array(sens) - np.array(spec))))  # where the curves meet
    ax.plot(THRESHOLDS[cross], sens[cross], "ko", ms=4)
    ax.annotate(f"best possible balance:\n{sens[cross]:.0f} % / {spec[cross]:.0f} % at {THRESHOLDS[cross]:.2f} g",
                (THRESHOLDS[cross], sens[cross]), xytext=(25, -45), textcoords="offset points",
                fontsize=8, arrowprops={"arrowstyle": "-", "lw": 0.6})
    ax.set_xlabel("Threshold on peak |a| in a 2 s window [g]")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.set_title("Baseline A: peak-acceleration threshold (all windows pooled)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    d = np.load(PROCESSED / "windows.npz")
    X, y = d["X"], d["y"]
    meta = pd.read_csv(PROCESSED / "windows_meta.csv")
    pd.set_option("display.width", 200)

    summary = {}
    for rule in ["balanced", "sens95"]:
        res = evaluate(X, y, meta, rule)
        print(f"\n=== threshold rule: {rule}  (T chosen on training folds) ===")
        shown = (res * 100).round(1)
        shown["T [g]"] = res["T [g]"].map("{:.2f}".format)
        print(shown.to_string())
        mean, std = res.mean(), res.std()
        print("mean +- std: " + "  ".join(f"{c} {100*mean[c]:.1f}+-{100*std[c]:.1f}"
                                          for c in res.columns if c != "T [g]"))
        summary[rule] = res["T [g]"].median()

    t = summary["balanced"]
    print(f"\nADL false-alarm rate per activity at T = {t:.2f} g (% of that activity's windows):")
    print(false_alarms_by_activity(X, y, meta, t).to_string())

    FIGS.mkdir(exist_ok=True)
    plot_sweep(X, y, summary, FIGS / "04_threshold_baseline.png")
    print(f"\nFigure saved to {FIGS / '04_threshold_baseline.png'}")
