"""Step 4: turn every 2 s window into 30 hand-made features for LightGBM.

The full definition of every feature is in docs/features.md; this file follows it
one group at a time:

    A  overall level        a_max, a_min, a_mean, a_std, a_cv, w_max, w_mean, w_std
    B  impact               peak_pos, jerk_max, pre_dip
    C  rhythm               n_peaks, acf_max, acf_lag
    D  after the impact     post_a_mean, post_a_std, post_w_mean
    F  before the impact    pre_a_std
    E  orientation change   tilt_change
    S  signal shape         a_skew, a_kurt, w_skew, w_kurt
    G  frequency            f_dom, f_dom_ratio, f_entropy, f_centroid,
                            f_band_low, f_band_mid, f_band_high

Almost everything is computed on two signals that do not depend on how the watch
sits on the wrist:
    a = |acceleration| [g]       w = |rotation speed| [deg/s]

Features that cannot be computed (e.g. no samples after a peak at the very end of
the window) are NaN on purpose: LightGBM handles missing values itself.

Run from the project folder:  .venv\\Scripts\\python.exe src\\features\\features.py
Writes data/processed/features.csv (one row per window, same order as windows_meta.csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

PROJECT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT / "data" / "processed"

# ---------------------------------------------------------------- fixed settings
# All values come from docs/features.md and are never tuned on test results.
FS = 50                    # Hz
GAP_AROUND_PEAK = 5        # samples (0.1 s) left out on each side of the peak
MIN_SEGMENT = 10           # samples (0.2 s): shorter pre/post segments give NaN
DIP_SEARCH = 25            # samples (0.5 s) before the peak to look for a dip
PEAK_HEIGHT = 2.0          # g, a "big" peak for n_peaks
PEAK_MIN_GAP = 10          # samples (0.2 s) between two counted peaks
ACF_LAGS = range(10, 51)   # samples, 0.2 to 1.0 s
TILT_SEGMENT = 25          # samples (0.5 s) at the start and end for orientation
FFT_SIZE = 128             # window zero-padded to 128 points -> 0.39 Hz steps
FREQ_RANGE = (0.5, 20.0)   # Hz, ignore the gravity offset and the unreliable top end
BANDS = {"low": (0.5, 3.0), "mid": (3.0, 8.0), "high": (8.0, 20.0)}  # Hz


# ---------------------------------------------------------------- helpers
def magnitudes(window):
    """Return a = |acc| [g] and w = |gyro| [deg/s] for one (100, 6) window."""
    a = np.linalg.norm(window[:, 0:3], axis=1)
    w = np.linalg.norm(window[:, 3:6], axis=1)
    return a, w


def before_and_after(signal, p):
    """Split a signal around the peak index p, leaving out +-GAP_AROUND_PEAK samples.

    Returns (pre, post). A segment shorter than MIN_SEGMENT is returned as None.
    """
    pre = signal[: max(p - GAP_AROUND_PEAK, 0)]
    post = signal[p + GAP_AROUND_PEAK + 1:]
    pre = pre if len(pre) >= MIN_SEGMENT else None
    post = post if len(post) >= MIN_SEGMENT else None
    return pre, post


def mean_or_nan(x):
    return float(np.mean(x)) if x is not None else np.nan


def std_or_nan(x):
    return float(np.std(x)) if x is not None else np.nan


def shifted_correlation(x, lag):
    """Correlation between the signal and a copy of itself shifted by `lag` samples.

    1 means the signal repeats exactly every `lag` samples. NaN if either part is flat.
    """
    first, second = x[:-lag], x[lag:]
    if np.std(first) == 0 or np.std(second) == 0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


# ---------------------------------------------------------------- feature groups
def overall_level(a, w):
    """Group A: how strong and how variable the movement is in the whole window."""
    return {
        "a_max": a.max(),
        "a_min": a.min(),
        "a_mean": a.mean(),
        "a_std": a.std(),
        "a_cv": a.std() / a.mean(),
        "w_max": w.max(),
        "w_mean": w.mean(),
        "w_std": w.std(),
    }


def impact(a, p):
    """Group B: where the peak is, how sharp it is, and whether a dip came before it."""
    jerk = np.abs(np.diff(a)) * FS                 # change per second [g/s]
    before_peak = a[max(p - DIP_SEARCH, 0): p]
    return {
        "peak_pos": p / FS,
        "jerk_max": jerk.max(),
        "pre_dip": before_peak.min() if p >= GAP_AROUND_PEAK else np.nan,
    }


def rhythm(a):
    """Group C: one big impact (fall) or a repeating pattern (jogging, clapping)?"""
    peaks, _ = find_peaks(a, height=PEAK_HEIGHT, distance=PEAK_MIN_GAP)

    correlations = np.array([shifted_correlation(a, lag) for lag in ACF_LAGS])
    if np.all(np.isnan(correlations)):
        acf_max, acf_lag = np.nan, np.nan
    else:
        best = np.nanargmax(correlations)
        acf_max, acf_lag = correlations[best], ACF_LAGS[best] / FS

    return {"n_peaks": len(peaks), "acf_max": acf_max, "acf_lag": acf_lag}


def after_impact(a, w, p):
    """Group D: a fall ends in stillness, a clap or a jog keeps moving."""
    _, a_post = before_and_after(a, p)
    _, w_post = before_and_after(w, p)
    return {
        "post_a_mean": mean_or_nan(a_post),
        "post_a_std": std_or_nan(a_post),
        "post_w_mean": mean_or_nan(w_post),
    }


def before_impact(a, p):
    """Group F: calm walking before a fall vs ongoing intense movement."""
    a_pre, _ = before_and_after(a, p)
    return {"pre_a_std": std_or_nan(a_pre)}


def orientation_change(window):
    """Group E: angle [deg] between the gravity direction at the start and at the end.

    The angle between two directions does not depend on how the watch is mounted.
    """
    start = window[:TILT_SEGMENT, 0:3].mean(axis=0)
    end = window[-TILT_SEGMENT:, 0:3].mean(axis=0)
    cos_angle = np.dot(start, end) / (np.linalg.norm(start) * np.linalg.norm(end))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)      # guard against rounding just past +-1
    return {"tilt_change": np.degrees(np.arccos(cos_angle))}


def signal_shape(a, w):
    """Group S: one sharp spike (high kurtosis) vs a regular signal (low kurtosis)."""
    return {
        "a_skew": skew(a),
        "a_kurt": kurtosis(a),                     # excess kurtosis (normal = 0)
        "w_skew": skew(w),
        "w_kurt": kurtosis(w),
    }


def frequency(a):
    """Group G: how the energy of |a| is spread over frequencies (0.5-20 Hz)."""
    x = (a - a.mean()) * np.hanning(len(a))        # remove gravity, soften window edges
    power = np.abs(np.fft.rfft(x, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1 / FS)

    keep = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    power, freqs = power[keep], freqs[keep]
    total = power.sum()
    names = ["f_dom", "f_dom_ratio", "f_entropy", "f_centroid"] + [f"f_band_{b}" for b in BANDS]
    if total == 0:                                 # perfectly flat window
        return dict.fromkeys(names, np.nan)

    share = power / total                          # energy share of every frequency step
    nonzero = share[share > 0]
    features = {
        "f_dom": freqs[np.argmax(power)],
        "f_dom_ratio": share.max(),
        "f_entropy": -np.sum(nonzero * np.log(nonzero)) / np.log(len(share)),  # 0..1
        "f_centroid": np.sum(freqs * share),
    }
    for band, (low, high) in BANDS.items():
        in_band = (freqs >= low) & (freqs < high) if high < FREQ_RANGE[1] else (freqs >= low)
        features[f"f_band_{band}"] = share[in_band].sum()
    return features


# ---------------------------------------------------------------- whole table
def window_features(window):
    """All 30 features of one (100, 6) window, as a dict in the docs/features.md order."""
    a, w = magnitudes(window)
    p = int(np.argmax(a))                          # index of the peak
    return {
        **overall_level(a, w),
        **impact(a, p),
        **rhythm(a),
        **after_impact(a, w, p),
        **before_impact(a, p),
        **orientation_change(window),
        **signal_shape(a, w),
        **frequency(a),
    }


def build_feature_table(X, meta):
    """One row per window: the meta columns first, then the 30 features."""
    features = pd.DataFrame([window_features(window) for window in X])
    return pd.concat([meta.reset_index(drop=True), features], axis=1)


if __name__ == "__main__":
    X = np.load(PROCESSED / "windows.npz")["X"].astype(np.float64)
    meta = pd.read_csv(PROCESSED / "windows_meta.csv")
    assert len(X) == len(meta), "windows.npz and windows_meta.csv do not line up"

    table = build_feature_table(X, meta)
    table.to_csv(PROCESSED / "features.csv", index=False)
    feature_names = table.columns[len(meta.columns):]

    print(f"Saved {len(table)} windows x {len(feature_names)} features to {PROCESSED / 'features.csv'}")

    missing = table[feature_names].isna().sum()
    print("\nMissing values (NaN) per feature, only those with any:")
    print(missing[missing > 0].to_string() if missing.any() else "  none")

    # Spot-check against docs/features.md: medians for falls vs a few daily activities.
    groups = {
        "fall windows": table.label == 1,
        "D01 walking": table.activity == "D01",
        "D02 jogging": table.activity == "D02",
        "D10 clapping": table.activity == "D10",
        "D09 hit table": table.activity == "D09",
    }
    check = ["a_max", "n_peaks", "acf_max", "f_dom", "a_kurt", "post_a_std", "tilt_change"]
    medians = pd.DataFrame({name: table.loc[rows, check].median() for name, rows in groups.items()})
    pd.set_option("display.width", 200)
    print("\nSpot-check, median per group:")
    print(medians.round(2).to_string())
