"""Step 5 (feature selection, step 1 of 2): remove near-duplicate features.

Rule (docs/features.md, section 6):
  1. Spearman correlation between every pair of features, on all windows.
     This is allowed because the step never looks at the labels.
  2. Go through the pairs with |rho| > 0.95, highest first.
  3. In each pair remove the feature that is more expensive to compute on the
     XIAO (COST below). Same cost -> keep the one listed first in
     docs/features.md, which is the simpler / easier-to-explain one.
  4. A pair is skipped once one of its features has already been removed,
     so we never remove both members of the same pair.

Step 2 of the selection (LightGBM gain importance) lives in src/models/train_lightgbm.py.

Run from the project folder:  .venv\\Scripts\\python.exe src\\features\\select_corr.py
Reads   data/processed/features.csv
Writes  data/processed/selected_features.txt   one kept feature name per line
        results/corr_removed.csv                which feature was removed and why
        figures/05_feature_correlation.png      correlation heatmap
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT / "data" / "processed" / "features.csv"
KEPT_TXT = PROJECT / "data" / "processed" / "selected_features.txt"
REMOVED_CSV = PROJECT / "results" / "corr_removed.csv"
FIGURE = PROJECT / "figures" / "05_feature_correlation.png"

THRESHOLD = 0.95

# Cost of each feature on the board, per 2 s window (docs/features.md, section 6).
# The order of this dict is the order of docs/features.md, used to break ties.
COST = {
    # A. overall level
    "a_max": 0, "a_min": 0, "a_mean": 0, "a_std": 1, "a_cv": 1,
    "w_max": 1, "w_mean": 1, "w_std": 1,
    # B. impact
    "peak_pos": 0, "jerk_max": 1, "pre_dip": 1,
    # C. rhythm
    "n_peaks": 1, "acf_max": 3, "acf_lag": 3,
    # D. after the impact
    "post_a_mean": 1, "post_a_std": 1, "post_w_mean": 1,
    # F. before the impact
    "pre_a_std": 1,
    # E. orientation change
    "tilt_change": 2,
    # S. signal shape
    "a_skew": 2, "a_kurt": 2, "w_skew": 2, "w_kurt": 2,
    # G. frequency (all share one FFT)
    "f_dom": 3, "f_dom_ratio": 3, "f_entropy": 3, "f_centroid": 3,
    "f_band_low": 3, "f_band_mid": 3, "f_band_high": 3,
}
SPEC_ORDER = list(COST)
COST_NAMES = {0: "almost free", 1: "cheap", 2: "moderate", 3: "expensive"}

# First feature of each group, to draw group separators on the heatmap.
GROUP_STARTS = {"A": "a_max", "B": "peak_pos", "C": "n_peaks", "D": "post_a_mean",
                "F": "pre_a_std", "E": "tilt_change", "S": "a_skew", "G": "f_dom"}


def correlated_pairs(rho, threshold=THRESHOLD):
    """All feature pairs with |rho| > threshold, strongest first."""
    names = rho.columns
    pairs = [(names[i], names[j], rho.iloc[i, j])
             for i in range(len(names)) for j in range(i + 1, len(names))
             if abs(rho.iloc[i, j]) > threshold]
    return sorted(pairs, key=lambda pair: -abs(pair[2]))


def pick_feature_to_remove(first, second):
    """Return (removed, kept, reason) for one correlated pair."""
    if COST[first] != COST[second]:
        removed, kept = (first, second) if COST[first] > COST[second] else (second, first)
        reason = (f"{removed} is more expensive ({COST_NAMES[COST[removed]]}) "
                  f"than {kept} ({COST_NAMES[COST[kept]]})")
    else:
        removed, kept = (second, first) if SPEC_ORDER.index(first) < SPEC_ORDER.index(second) \
            else (first, second)
        reason = f"same cost ({COST_NAMES[COST[kept]]}); {kept} is simpler (listed first in the spec)"
    return removed, kept, reason


def filter_correlated(rho, threshold=THRESHOLD):
    """Apply the rule. Returns (kept feature names, table of removals)."""
    removed = {}
    rows = []
    for first, second, value in correlated_pairs(rho, threshold):
        if first in removed or second in removed:
            continue  # rule 4: this pair is already resolved
        drop, keep, reason = pick_feature_to_remove(first, second)
        removed[drop] = keep
        rows.append({"removed": drop, "kept": keep, "rho": round(value, 3), "reason": reason})

    kept = [name for name in rho.columns if name not in removed]
    return kept, pd.DataFrame(rows, columns=["removed", "kept", "rho", "reason"])


def plot_heatmap(rho, removed, path, threshold=THRESHOLD):
    """Spearman correlation heatmap; removed features greyed, |rho| > threshold outlined."""
    # Diverging: blue = negative, neutral grey = no correlation, red = positive.
    cmap = LinearSegmentedColormap.from_list(
        "blue_grey_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a3201f"])
    names = list(rho.columns)
    n = len(names)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(rho.to_numpy(), cmap=cmap, vmin=-1, vmax=1)

    # Outline the cells above the threshold (lower triangle only, to avoid doubles).
    for i in range(n):
        for j in range(i):
            if abs(rho.iloc[i, j]) > threshold:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.5, ec="#0b0b0b"))

    # Thin white lines between feature groups.
    for name in list(GROUP_STARTS.values())[1:]:
        k = names.index(name) - 0.5
        ax.axhline(k, color="white", lw=2)
        ax.axvline(k, color="white", lw=2)

    labels = [f"{name}  (removed)" if name in removed else name for name in names]
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("#9a9893" if "(removed)" in tick.get_text() else "#0b0b0b")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Spearman correlation", color="#52514e")
    bar.outline.set_visible(False)
    ax.set_title(f"Feature correlation (all {n} features); outlined: |rho| > {threshold}",
                 loc="left", fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    table = pd.read_csv(FEATURES_CSV)
    features = [name for name in SPEC_ORDER if name in table.columns]
    assert features == SPEC_ORDER, "features.csv does not have the 30 features of docs/features.md"

    rho = table[features].corr(method="spearman")  # pairwise, ignores NaN rows per pair
    kept, removals = filter_correlated(rho)

    KEPT_TXT.write_text("\n".join(kept) + "\n")
    REMOVED_CSV.parent.mkdir(exist_ok=True)
    removals.to_csv(REMOVED_CSV, index=False)
    FIGURE.parent.mkdir(exist_ok=True)
    plot_heatmap(rho, set(removals.removed), FIGURE)

    pd.set_option("display.width", 200, "display.max_colwidth", 120)
    print(f"Pairs with |rho| > {THRESHOLD}: {len(correlated_pairs(rho))}")
    print("\nRemoved:")
    print(removals.to_string(index=False) if len(removals) else "  nothing")

    near = [(a, b, round(v, 3)) for a, b, v in correlated_pairs(rho, 0.90)
            if abs(v) <= THRESHOLD]
    print(f"\nClose but kept (0.90 < |rho| <= {THRESHOLD}), for information only:")
    for a, b, v in near:
        print(f"  {a:12s} {b:12s} {v:+.3f}")

    print(f"\nKept {len(kept)} of {len(features)} features -> {KEPT_TXT}")
    print(f"Removal table -> {REMOVED_CSV}")
    print(f"Heatmap       -> {FIGURE}")
