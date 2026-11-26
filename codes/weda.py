"""Load the WEDA-FALL dataset (wrist-worn Fitbit Sense) into pandas.

github.com/joaojtmarques/WEDA-FALL. Layout of the 50 Hz data:
    dataset/50Hz/<activity>/U<user>_R<trial>_accel.csv   time [s], x, y, z [m/s^2]
    dataset/50Hz/<activity>/U<user>_R<trial>_gyro.csv    time [s], x, y, z [rad/s]
    dataset/fall_timestamps.csv                          fall start / end [s] per fall trial

What the raw files look like (checked on all 969 trials):
  * The watch delivers samples in bursts, so time steps are uneven (2 ms inside a
    burst, ~0.1 s between bursts) and the effective rate is 40-50 Hz, not exactly 50.
  * Accel and gyro files of one trial can have different row counts.
  * A few files have time jumping backwards (timestamp resets).
So we interpolate both sensors onto one uniform 50 Hz grid, and flag trials whose
timestamps are not increasing (timing_ok = False) instead of guessing their order.

Units are converted to g and deg/s, the units the LSM6DS3TR-C on the XIAO reports.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

FS = 50                  # Hz, the uniform rate we resample to
G = 9.80665              # m/s^2 per g
RAD2DEG = 180 / np.pi

ELDERLY_FROM = 21        # user ids 1-14 are young, 21-31 are elderly (README)

NAME_RE = re.compile(r"^U(?P<user>\d+)_R(?P<trial>\d+)_accel\.csv$")

ACTIVITIES = {
    "F01": "Fall forward, slip while walking", "F02": "Lateral fall, slip while walking",
    "F03": "Fall backward, slip while walking", "F04": "Fall forward, trip while walking",
    "F05": "Fall backward trying to sit down", "F06": "Fall forward while sitting (fainting)",
    "F07": "Fall backward while sitting (fainting)", "F08": "Lateral fall while sitting (fainting)",
    "D01": "Walking", "D02": "Jogging", "D03": "Walking up and downstairs",
    "D04": "Sit on a chair, wait, get up", "D05": "Try to get up, collapse into chair",
    "D06": "Crouch, tie shoes, get up", "D07": "Stumble while walking",
    "D08": "Gentle jump (reach high object)", "D09": "Hit table with hand",
    "D10": "Clapping hands", "D11": "Opening and closing door",
}


def _read_sensor(path):
    """Return (time, xyz) arrays from one accel or gyro csv."""
    df = pd.read_csv(path)
    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:4].to_numpy(float)


def _timing_ok(t):
    return len(t) > 1 and bool(np.all(np.diff(t) > 0))


def load_trial(accel_path):
    """Load one trial on a uniform 50 Hz grid.

    Columns: t [s], ax, ay, az [g], gx, gy, gz [deg/s], acc_mag [g], gyro_mag [deg/s].
    Raises ValueError if either sensor's timestamps are not strictly increasing.
    """
    accel_path = Path(accel_path)
    gyro_path = accel_path.with_name(accel_path.name.replace("_accel.csv", "_gyro.csv"))
    ta, acc = _read_sensor(accel_path)
    tg, gyr = _read_sensor(gyro_path)
    if not (_timing_ok(ta) and _timing_ok(tg)):
        raise ValueError(f"timestamps not increasing in {accel_path.parent.name}/{accel_path.name}")

    # Common time span of both sensors, sampled every 1/FS seconds
    t = np.arange(max(ta[0], tg[0]), min(ta[-1], tg[-1]), 1 / FS)
    df = pd.DataFrame({"t": t})
    for i, axis in enumerate("xyz"):
        df[f"a{axis}"] = np.interp(t, ta, acc[:, i]) / G
    for i, axis in enumerate("xyz"):
        df[f"g{axis}"] = np.interp(t, tg, gyr[:, i]) * RAD2DEG
    df["acc_mag"] = np.sqrt(df.ax**2 + df.ay**2 + df.az**2)
    df["gyro_mag"] = np.sqrt(df.gx**2 + df.gy**2 + df.gz**2)
    return df


def index_dataset(root):
    """List every trial under `root` (the WEDA-FALL folder that holds `dataset/`).

    One row per trial: path (accel file), activity, user, trial, is_fall, age_group,
    description, fall_start / fall_end [s] (falls only), raw_rate [Hz], timing_ok.
    """
    data = Path(root) / "dataset" / "50Hz"
    stamps = pd.read_csv(Path(root) / "dataset" / "fall_timestamps.csv", encoding="utf-8-sig")
    stamps = stamps.set_index("filename")

    rows = []
    for path in sorted(data.glob("*/U*_R*_accel.csv")):
        m = NAME_RE.match(path.name)
        if m is None:  # skips *_vertical_accel.csv
            continue
        activity, user = path.parent.name, int(m["user"])
        key = f"{activity}/{path.name.removesuffix('_accel.csv')}"
        ta, _ = _read_sensor(path)
        tg, _ = _read_sensor(path.with_name(path.name.replace("_accel.csv", "_gyro.csv")))
        rows.append({
            "path": path,
            "activity": activity,
            "user": user,
            "trial": int(m["trial"]),
            "is_fall": activity.startswith("F"),
            "age_group": "elderly" if user >= ELDERLY_FROM else "young",
            "description": ACTIVITIES.get(activity, "?"),
            "fall_start": stamps.start_time.get(key, np.nan),
            "fall_end": stamps.end_time.get(key, np.nan),
            "raw_rate": len(ta) / (ta[-1] - ta[0]) if len(ta) > 1 else np.nan,
            "timing_ok": _timing_ok(ta) and _timing_ok(tg),
        })
    return pd.DataFrame(rows)
