"""Step 2: split windows into 5 folds by user, so nobody is in both train and test.

Windows from one person look alike, so a random window split would put the same
person on both sides and inflate accuracy. StratifiedGroupKFold keeps each user in
exactly one fold (groups = user) and balances the fall rate between folds (y).

Note: only the 14 young users performed falls; the 11 elderly users add ADLs only.

Run from the project folder:  .venv\\Scripts\\python.exe src\\splits.py
Adds a `fold` column (0-4) to data/processed/windows_meta.csv.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
SEED = 42

PROJECT = Path(__file__).resolve().parents[1]
META = PROJECT / "data" / "processed" / "windows_meta.csv"


def assign_folds(meta, n_folds=N_FOLDS, seed=SEED):
    """Return a Series: fold number (0..n_folds-1) for every window, grouped by user."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=meta.index)
    for k, (_, test_idx) in enumerate(cv.split(meta, meta.label, groups=meta.user)):
        fold.iloc[test_idx] = k
    return fold


def check_no_leak(meta):
    """Every user must sit in exactly one fold."""
    folds_per_user = meta.groupby("user").fold.nunique()
    assert (folds_per_user == 1).all(), f"users in several folds: {folds_per_user[folds_per_user > 1]}"


def fold_report(meta):
    rows = []
    for k, part in meta.groupby("fold"):
        users = sorted(part.user.unique())
        rows.append({
            "fold": k,
            "test users": ", ".join(str(u) for u in users),
            "young": sum(u < 21 for u in users),
            "elderly": sum(u >= 21 for u in users),
            "windows": len(part),
            "fall windows": int(part.label.sum()),
            "fall %": round(100 * part.label.mean(), 1),
            "fall trials": part[part.fall_trial].trial_id.nunique(),
            "ADL trials": part[~part.fall_trial].trial_id.nunique(),
        })
    return pd.DataFrame(rows).set_index("fold")


if __name__ == "__main__":
    meta = pd.read_csv(META)
    meta["fold"] = assign_folds(meta)
    check_no_leak(meta)
    meta.to_csv(META, index=False)

    pd.set_option("display.width", 200)
    print(fold_report(meta))
    print(f"\nNo user appears in more than one fold. Saved fold column to {META}")
