"""Report figures from the saved results (no training needed).

  A  figures/07_false_alarms_by_activity   share of each daily activity's recordings with
                                           >= 1 false alarm: LightGBM, 1D-CNN, their average
  B  figures/08_roc_pr_curves              window-level ROC and precision-recall curves
  C  figures/09_float_vs_int8              CNN fall probability, float32 vs int8 (RS-1)

All numbers come from out-of-fold predictions (5 folds split by subject): every window is
scored by the model that did not see its subject during training.

Run from the project folder, after src/models/combine_models.py:
    .venv\\Scripts\\python.exe src\\plots\\plot_results.py
Each figure is saved as .pdf (vector, for LaTeX) and .png (300 dpi).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"

# Categorical palette, fixed order (slot 1 blue, 2 orange, 3 aqua). Each series also has its
# own hatch / line style, so the figures still read in greyscale print.
SERIES = {
    "LightGBM": dict(color="#257de0", hatch="", linestyle="-"),
    "1D-CNN": dict(color="#f47230", hatch="////", linestyle="--"),
    "LightGBM + CNN": dict(color="#00b894", hatch="....", linestyle=":"),
}
INK, INK_2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"

ACTIVITY_NAMES = {
    "D01": "Walking", "D02": "Jogging", "D03": "Stairs up/down", "D04": "Sit down, get up",
    "D05": "Collapse into chair", "D06": "Crouch, tie shoes", "D07": "Stumble",
    "D08": "Gentle jump", "D09": "Hit table with hand", "D10": "Clapping hands",
    "D11": "Open/close door",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "hatch.linewidth": 0.6, "figure.dpi": 100,
    "savefig.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.pdf / .png")


# ---------------------------------------------------------------- A
def false_alarms_by_activity():
    table = pd.read_csv(RESULTS / "false_alarms_by_activity.csv", index_col="activity")
    table = table[table.max(axis=1) > 0]                       # drop never-alarming activities
    table = table.sort_values("LightGBM + CNN")                # worst at the top
    labels = [f"{ACTIVITY_NAMES[a]} ({a})" for a in table.index]

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(table))
    height = 0.26
    for i, (name, style) in enumerate(SERIES.items()):
        offset = (1 - i) * height                              # first series on top
        ax.barh(y + offset, table[name], height=height * 0.9, color=style["color"],
                hatch=style["hatch"], edgecolor="white", linewidth=0.0, label=name)
    # direct labels only on the combined detector (the proposed one)
    for yi, value in zip(y, table["LightGBM + CNN"]):
        ax.text(value + 0.8, yi - height, f"{value:.0f}%", va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Recordings of the activity with at least one false alarm [%]")
    ax.set_xlim(0, 70)
    ax.xaxis.grid(True)
    ax.spines["left"].set_visible(False)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles, names, loc="lower right", ncol=1)
    never = [ACTIVITY_NAMES[a] for a in ["D01", "D04"]]
    ax.set_title("False alarms by daily activity (single-window rule, cut-offs from other folds)",
                 loc="left", fontsize=9.5, color=INK)
    fig.text(0.0, -0.02, f"No false alarm with any model: {'; '.join(never).lower()}.",
             fontsize=7.5, color=INK_2)
    save(fig, "07_false_alarms_by_activity")


# ---------------------------------------------------------------- B
def roc_pr_curves():
    curves = {name: pd.read_csv(RESULTS / file) for name, file in [
        ("LightGBM", "lgbm_tuned_top15_features_predictions.csv"),
        ("1D-CNN", "cnn_none_predictions.csv"),
        ("LightGBM + CNN", "combined_lgbm_cnn_predictions.csv")]}

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(6.5, 2.9))
    for name, df in curves.items():
        style = SERIES[name]
        fpr, tpr, _ = roc_curve(df.label, df.probability)
        precision, recall, _ = precision_recall_curve(df.label, df.probability)
        line = dict(color=style["color"], linestyle=style["linestyle"], linewidth=1.6)
        ax_roc.plot(100 * fpr, 100 * tpr, label=f"{name} (AUC {auc(fpr, tpr):.3f})", **line)
        ax_pr.plot(100 * recall, 100 * precision, label=f"{name} (AUC {auc(recall, precision):.3f})",
                   **line)

    ax_roc.set(xlim=(0, 20), ylim=(70, 100.5), xlabel="False-positive rate [%] (0-20 % shown)",
               ylabel="True-positive rate (recall) [%]")
    ax_roc.set_title("ROC (fall windows)", loc="left", color=INK)
    ax_pr.set(xlim=(50, 100.5), ylim=(0, 102), xlabel="Recall [%] (50-100 % shown)",
              ylabel="Precision [%]")
    ax_pr.set_title("Precision-recall", loc="left", color=INK)
    ax_pr.axvline(95, color=AXIS, linewidth=0.8, linestyle="-", zorder=0)
    ax_pr.text(94.5, 4, "95 % recall", rotation=90, ha="right", va="bottom", fontsize=7.5,
               color=INK_2)
    for ax in (ax_roc, ax_pr):
        ax.grid(True)
        ax.legend(loc="lower left" if ax is ax_pr else "lower right", fontsize=7.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "08_roc_pr_curves")


# ---------------------------------------------------------------- C
def float_vs_int8():
    float_ = pd.read_csv(RESULTS / "cnn_none_predictions.csv")
    int8 = pd.read_csv(RESULTS / "cnn_none_int8_predictions.csv")
    assert (float_.trial_id.values == int8.trial_id.values).all()
    p_f, p_q = float_.probability.to_numpy(), int8.probability.to_numpy()
    same_05 = 100 * np.mean((p_f >= 0.5) == (p_q >= 0.5))
    same_cut = 100 * np.mean((p_f >= float_.threshold_sens95) == (p_q >= int8.threshold_sens95))
    mean_abs = np.mean(np.abs(p_f - p_q))

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    is_fall = float_.label.to_numpy() == 1
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=0)
    ax.scatter(p_f[~is_fall], p_q[~is_fall], s=6, marker="o", color="#9c9c9c",
               alpha=0.45, linewidths=0, label="Not-fall windows")
    ax.scatter(p_f[is_fall], p_q[is_fall], s=9, marker="^", color="#6a4ce2",
               alpha=0.7, linewidths=0, label="Fall windows")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Fall probability, float32 CNN",
           ylabel="Fall probability, int8 CNN")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper left", markerscale=2)
    ax.text(0.98, 0.04, f"Same decision: {same_05:.1f}% (cut-off 0.5)\n"
                        f"{same_cut:.1f}% (95%-recall cut-offs)\n"
                        f"mean |difference|: {mean_abs:.4f}\n"
                        f"n = {len(p_f):,} windows",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID))
    ax.set_title("Float32 vs int8 CNN output (RS-1)", loc="left", color=INK)
    save(fig, "09_float_vs_int8")
    print(f"  float vs int8: same decision {same_05:.2f}% @0.5, {same_cut:.2f}% @sens95 cut-offs")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    false_alarms_by_activity()
    roc_pr_curves()
    float_vs_int8()
