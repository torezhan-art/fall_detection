"""Explore WEDA-FALL: print a dataset summary and save three figures to ../figures.

Run from the project folder:  .venv\\Scripts\\python.exe src\\plot_weda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt

from weda import index_dataset, load_trial

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "raw" / "WEDA-FALL"
FIGS = PROJECT / "figures"
USER = 1  # one young participant so the comparison is the same person

FALL_COLOR, ADL_COLOR = "#c0392b", "#2c6fbb"


def pick(index, activity, user=USER, trial=1):
    row = index[(index.activity == activity) & (index.user == user) & (index.trial == trial)]
    return row.iloc[0]


def print_summary(index):
    print(f"Trials: {len(index)}  (falls {index.is_fall.sum()}, ADLs {(~index.is_fall).sum()})")
    print(f"Users: {index.user.nunique()}  (young {index[index.age_group == 'young'].user.nunique()}, "
          f"elderly {index[index.age_group == 'elderly'].user.nunique()})")
    print("\nTrials by age group and label:")
    print(index.groupby(["age_group", "is_fall"]).size().unstack(fill_value=0)
          .rename(columns={False: "ADL", True: "fall"}))
    bad = index[~index.timing_ok]
    print(f"\nTrials with broken timestamps (timing_ok=False): {len(bad)} "
          f"({bad.is_fall.sum()} falls, {(~bad.is_fall).sum()} ADLs)")
    print(f"Raw sample rate p5 / median / p95: {index.raw_rate.quantile([.05, .5, .95]).round(1).tolist()} Hz")
    print(f"Fall trials with a fall_start timestamp: {index[index.is_fall].fall_start.notna().sum()} "
          f"of {index.is_fall.sum()}")


def plot_one_fall(index):
    """Figure 1: all six axes of one fall, the magnitude, and the annotated fall interval."""
    row = pick(index, "F01")
    df = load_trial(row.path)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    for col in ["ax", "ay", "az"]:
        axes[0].plot(df.t, df[col], lw=1, label=col)
    axes[0].set_ylabel("Acceleration [g]")
    axes[0].legend(loc="upper left", ncol=3)

    for col in ["gx", "gy", "gz"]:
        axes[1].plot(df.t, df[col], lw=1, label=col)
    axes[1].set_ylabel("Angular rate [deg/s]")
    axes[1].legend(loc="upper left", ncol=3)

    axes[2].plot(df.t, df.acc_mag, color=FALL_COLOR, lw=1.2)
    axes[2].axhline(1.0, color="grey", ls="--", lw=0.8)
    peak = df.acc_mag.idxmax()
    axes[2].annotate(f"impact {df.acc_mag[peak]:.1f} g", (df.t[peak], df.acc_mag[peak]),
                     xytext=(10, -5), textcoords="offset points", color=FALL_COLOR)
    axes[2].set_ylabel("|a| [g]")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:  # annotated fall interval from fall_timestamps.csv
        ax.axvspan(row.fall_start, row.fall_end, color=FALL_COLOR, alpha=0.1)
    axes[0].text(row.fall_start, axes[0].get_ylim()[1], " annotated fall", va="top", color=FALL_COLOR, fontsize=9)

    fig.suptitle(f"WEDA-FALL {row.activity}/U{USER:02d}_R01: {row.description} (wrist, resampled to 50 Hz)")
    fig.tight_layout()
    fig.savefig(FIGS / "01_one_fall_6axis.png", dpi=150)
    plt.close(fig)


def plot_fall_vs_adl(index):
    """Figure 2: acceleration magnitude of falls next to the ADLs most likely to look like one."""
    cases = ["F01", "F03", "F06", "D05", "D07", "D09"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)

    for ax, activity in zip(axes.flat, cases):
        row = pick(index, activity)
        df = load_trial(row.path)
        color = FALL_COLOR if row.is_fall else ADL_COLOR
        ax.plot(df.t, df.acc_mag, color=color, lw=1)
        ax.axhline(1.0, color="grey", ls="--", lw=0.8)
        if row.is_fall:
            ax.axvspan(row.fall_start, row.fall_end, color=FALL_COLOR, alpha=0.1)
        ax.set_title(f"{activity} ({'FALL' if row.is_fall else 'ADL'})\n{row.description}", fontsize=10)
        ax.text(0.98, 0.95, f"peak {df.acc_mag.max():.1f} g", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("|a| [g]")

    fig.suptitle(f"Falls (red) vs fall-like daily activities (blue), wrist, user {USER}")
    fig.tight_layout()
    fig.savefig(FIGS / "02_falls_vs_hard_adls.png", dpi=150)
    plt.close(fig)


def plot_dataset_overview(index):
    """Figure 3: trials per activity, split into young and elderly participants."""
    counts = index.groupby(["activity", "age_group"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(counts.index, counts["young"], label="young (falls + ADLs)",
           color=[FALL_COLOR if a.startswith("F") else ADL_COLOR for a in counts.index])
    ax.bar(counts.index, counts["elderly"], bottom=counts["young"], label="elderly (ADLs only)",
           color="#7f8c8d")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"WEDA-FALL: {len(index)} trials from {index.user.nunique()} users "
                 f"({index.is_fall.sum()} falls in red, {(~index.is_fall).sum()} ADLs in blue / grey)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "03_dataset_overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    index = index_dataset(DATA)
    print_summary(index)
    plot_one_fall(index)
    plot_fall_vs_adl(index)
    plot_dataset_overview(index)
    print(f"\nFigures saved to {FIGS}")
