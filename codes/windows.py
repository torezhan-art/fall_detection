"""Step 1: cut WEDA-FALL trials into labelled 2 s windows (DR-1: 6 axes, 50 Hz, 1 s hop).

Labelling rule for each window [t0, t0 + 2 s):
  * fall (1)    the window contains the trial's impact, i.e. the largest |a| inside
                the annotated fall interval [fall_start, fall_end]
  * dropped     the window overlaps the annotated interval but misses the impact
                (only the slide or only the lying still: ambiguous, so not used)
  * not fall (0) every other window: all ADL windows, and the walking before /
                lying after a fall that do not touch the annotated interval

Run from the project folder:  .venv\\Scripts\\python.exe src\\windows.py
Writes data/processed/windows.npz (X, y) and data/processed/windows_meta.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from weda import FS, index_dataset, load_trial

WINDOW_S, HOP_S = 2.0, 1.0
WINDOW = int(WINDOW_S * FS)   # 100 samples
HOP = int(HOP_S * FS)         # 50 samples
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]   # g, g, g, deg/s, deg/s, deg/s

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "raw" / "WEDA-FALL"
OUT = PROJECT / "data" / "processed"


def impact_time(df, fall_start, fall_end):
    """Time [s] of the largest acceleration magnitude inside the annotated fall."""
    inside = df[(df.t >= fall_start) & (df.t <= fall_end)]
    return df.t[inside.acc_mag.idxmax()]


def windows_of_trial(row):
    """Return (X, y, meta rows) for one trial. X has shape (n, WINDOW, 6)."""
    df = load_trial(row.path)
    signals = df[CHANNELS].to_numpy(np.float32)
    t_impact = impact_time(df, row.fall_start, row.fall_end) if row.is_fall else None

    X, y, meta = [], [], []
    for start in range(0, len(df) - WINDOW + 1, HOP):
        t0, t1 = df.t[start], df.t[start] + WINDOW_S
        if row.is_fall:
            touches_fall = t0 < row.fall_end and t1 > row.fall_start
            has_impact = t0 <= t_impact < t1
            if has_impact:
                label = 1
            elif touches_fall:
                continue  # ambiguous window: dropped
            else:
                label = 0
        else:
            label = 0
        X.append(signals[start:start + WINDOW])
        y.append(label)
        meta.append({"trial_id": f"{row.activity}/U{row.user:02d}_R{row.trial:02d}",
                     "user": row.user, "activity": row.activity, "age_group": row.age_group,
                     "fall_trial": row.is_fall, "t_start": round(t0, 2), "label": label})
    return X, y, meta


def build_windows(index):
    """Window every trial with usable timestamps. Returns X, y and a meta DataFrame."""
    X, y, meta = [], [], []
    for _, row in index[index.timing_ok].iterrows():
        xs, ys, ms = windows_of_trial(row)
        X += xs; y += ys; meta += ms
    return np.stack(X), np.array(y, dtype=np.int8), pd.DataFrame(meta)


if __name__ == "__main__":
    index = index_dataset(DATA)
    X, y, meta = build_windows(index)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "windows.npz", X=X, y=y)
    meta.to_csv(OUT / "windows_meta.csv", index=False)

    used = index[index.timing_ok]
    print(f"Trials used: {len(used)} of {len(index)} (dropped {len(index) - len(used)} with broken timestamps)")
    print(f"Windows: {len(y)}  shape {X.shape}  (window {WINDOW_S:g} s = {WINDOW} samples, hop {HOP_S:g} s)")
    print(f"  fall windows: {y.sum()}   not-fall windows: {(y == 0).sum()}   "
          f"ratio 1 : {(y == 0).sum() / y.sum():.1f}")
    per_trial = meta[meta.fall_trial].groupby("trial_id").label.sum()
    print(f"Fall windows per fall trial: {per_trial.value_counts().sort_index().to_dict()}")
    print(f"Fall trials with no fall window: {(per_trial == 0).sum()}")
    print(f"Saved to {OUT}")
