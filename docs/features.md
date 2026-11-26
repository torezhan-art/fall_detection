# Feature specification: LightGBM fall detector

Spec for the hand-made features used by the LightGBM model (and as a backup
on-device model). Decided on 2026-11-06. Inputs are the windows in
`data/processed/windows.npz` (`X`: 9175 x 100 x 6, `y`: 0/1), with per-window
information in `data/processed/windows_meta.csv`.

All constants below are fixed in advance and are **never tuned on test results**.

---

## 1. Basic definitions

- **Window:** 100 samples = 2 s at 50 Hz. Channels `ax, ay, az` [g], `gx, gy, gz` [deg/s].
- **`a[n] = sqrt(ax^2 + ay^2 + az^2)`**: total acceleration [g].
- **`w[n] = sqrt(gx^2 + gy^2 + gz^2)`**: total rotation speed [deg/s].
  Both come out the same whichever way the device is worn (Fitbit vs XIAO, left vs right wrist).
- **`p` = argmax a[n]**: index of the peak (0..99).
- **Segments around the peak** (samples within +-5 of `p`, i.e. +-0.1 s, are left out so the impact itself is not counted):
  - **pre** = samples `0 .. p-6`
  - **post** = samples `p+6 .. 99`
- **Missing values:** if pre or post has fewer than 10 samples (0.2 s), its features are left as **NaN**.
  LightGBM handles NaN natively; do not fill with 0.

## 2. Feature list (30 features, 8 groups)

| # | Name | Definition | Unit | Purpose |
|---|---|---|---|---|
| **A. Overall level** |||||
| 1 | `a_max` | max a (= peak) | g | impact strength |
| 2 | `a_min` | min a | g | dip as the wrist drops |
| 3 | `a_mean` | mean a | g | overall activity |
| 4 | `a_std` | std a | g | variability |
| 5 | `a_cv` | a_std / a_mean | - | relative variability (Kavuncuoglu 2024, F17) |
| 6 | `w_max` | max w | deg/s | fastest wrist rotation |
| 7 | `w_mean` | mean w | deg/s | rotation activity |
| 8 | `w_std` | std w | deg/s | rotation variability |
| **B. Impact** |||||
| 9 | `peak_pos` | p / 50 | s | where the impact sits (0-2 s); tells the model how much "after" data exists |
| 10 | `jerk_max` | max of abs(a[n] - a[n-1]) * 50 | g/s | sharpness of impact (table hit = very sharp) |
| 11 | `pre_dip` | min a in the 25 samples (0.5 s) before p; NaN if p < 5 | g | drop below 1 g just before impact |
| **C. Rhythm** |||||
| 12 | `n_peaks` | number of peaks in a >= 2.0 g, at least 10 samples (0.2 s) apart | count | one impact (fall) vs many (jogging, clapping) |
| 13 | `acf_max` | highest normalised autocorrelation of (a - mean a) over lags 10-50 samples (0.2-1.0 s); NaN if a_std ~ 0 | - | how rhythmic the window is (paper's best feature, F11) |
| 14 | `acf_lag` | lag where `acf_max` occurs | s | repetition period (step / clap rate) |
| **D. After the impact** (post segment) |||||
| 15 | `post_a_mean` | mean a in post | g | ~1 g when lying still |
| 16 | `post_a_std` | std a in post | g | stillness: low after a fall, high after a clap |
| 17 | `post_w_mean` | mean w in post | deg/s | wrist still rotating or not |
| **F. Before the impact** (pre segment) |||||
| 18 | `pre_a_std` | std a in pre | g | calm walking before a fall vs ongoing intense movement |
| **E. Orientation change** |||||
| 19 | `tilt_change` | angle between mean (ax,ay,az) of the first 25 samples and of the last 25 samples: arccos(u.v / (abs(u) abs(v))) | deg | wrist ends at a new angle after a fall; after a clap or jump it does not |
| **S. Signal shape** |||||
| 20 | `a_skew` | skewness of a | - | asymmetry of the signal |
| 21 | `a_kurt` | excess kurtosis of a (SciPy default) | - | one sharp spike (high) vs regular signal (low) |
| 22 | `w_skew` | skewness of w | - | |
| 23 | `w_kurt` | excess kurtosis of w | - | |
| **G. Frequency** (on a - mean a, Hann window, FFT padded to 128 points -> 0.39 Hz steps, only 0.5-20 Hz used) |||||
| 24 | `f_dom` | frequency with the most energy | Hz | rhythm rate |
| 25 | `f_dom_ratio` | that frequency's share of total energy (0.5-20 Hz) | - | how strongly rhythmic |
| 26 | `f_entropy` | spectral entropy, normalised to 0-1 | - | spread-out energy (fall) vs concentrated (jogging) |
| 27 | `f_centroid` | energy-weighted mean frequency | Hz | one-number summary of the spectrum |
| 28 | `f_band_low` | share of energy in 0.5-3 Hz | - | body movement |
| 29 | `f_band_mid` | share of energy in 3-8 Hz | - | fast hand movement |
| 30 | `f_band_high` | share of energy in 8-20 Hz | - | sharp impacts (least reliable, see notes) |

## 3. Fixed settings

| Setting | Value |
|---|---|
| Sampling rate / window | 50 Hz, 100 samples |
| Samples left out around the peak | +-5 (0.1 s) |
| Minimum pre/post length | 10 samples (0.2 s), otherwise NaN |
| Dip search before peak | 25 samples (0.5 s) |
| Peak-count level / minimum gap | 2.0 g / 10 samples (0.2 s) |
| Autocorrelation lags | 10-50 samples (0.2-1.0 s) |
| Orientation segments | first and last 25 samples |
| FFT | subtract mean, Hann window, pad to 128 points, use 0.5-20 Hz |
| Frequency bands | 0.5-3 / 3-8 / 8-20 Hz |

## 4. Left out on purpose

| Dropped | Reason |
|---|---|
| range, SMA | Repeat `a_max` - `a_min` and `a_mean` |
| Peak spacing | Covered by `acf_lag` |
| Per-axis statistics (paper's F3 etc.) | Depend on how the device is mounted (Fitbit vs XIAO); orientation is covered by `tilt_change` |
| SVD, mutual information, autoregression, cross-correlation | Expensive on the board, depend on mounting, not top performers in Kavuncuoglu 2024 |
| Frequency features on gyro | Keep the set small; add later only if error analysis asks for it |

## 5. Output table

- `data/processed/features.csv` (or `.parquet`): 9175 rows x 30 features, plus
  `trial_id, user, activity, age_group, fold, label` copied from `windows_meta.csv`
  in the **same row order**.
- Each feature uses only its own window and fixed settings, so features can be
  calculated once for all windows before splitting (no leakage).
- **Spot-check before training:**
  - clear fall window: high `a_kurt`, low `post_a_std`, large `tilt_change`
  - jogging window: high `acf_max`, `f_dom` ~ 2.5-3 Hz, `n_peaks` >= 3
  - clapping window: many peaks, high `post_a_std`

---

## 6. Feature selection

### Step 1: correlation filter (label-free)

1. Calculate **Spearman correlation** between all features on all 9175 windows
   (allowed: this step does not use labels).
2. Take all pairs with **abs(rho) > 0.95**, and work through them from the highest correlation down.
3. In each pair, **remove the more expensive feature** (cost table below). If both cost the same,
   keep the one that is easier to explain or more likely to transfer to the XIAO.
4. Once a feature is removed, skip remaining pairs that include it (never remove both members of a pair).
5. Record every removal (pair, rho, kept, reason) for the report.

**Cost of each feature on the board (per 2 s window):**

| Cost | Features | Why |
|---|---|---|
| 0 - almost free | `a_max`, `a_min`, `a_mean`, `peak_pos` | Come out of the pass that computes abs(a) |
| 1 - cheap | `a_std`, `a_cv`, `w_max`, `w_mean`, `w_std`, `jerk_max`, `pre_dip`, `post_a_mean`, `post_a_std`, `post_w_mean`, `pre_a_std`, `n_peaks` | One more simple pass over 100 samples |
| 2 - moderate | `a_skew`, `a_kurt`, `w_skew`, `w_kurt`, `tilt_change` | Higher powers or an arccos |
| 3 - expensive | `acf_max`, `acf_lag` | ~4000 multiply-adds (41 lags x 100 samples) |
| 3 - expensive | all 7 `f_*` | Need a 128-point FFT |

**Shared costs:** the 7 `f_*` features share one FFT, `acf_max`/`acf_lag` share one
autocorrelation, and the 6 `w_*` features share the rotation magnitude. Removing one member
saves almost nothing; the saving comes only when the whole group goes. Prefer removing a member
of such a group only if it helps remove the whole group; otherwise choose by explainability.

**Pairs likely to appear** (the actual numbers decide):
`a_std`/`a_cv` (keep `a_std`), `a_max`/`jerk_max` (keep `a_max`), `w_mean`/`w_std`,
`f_centroid`/`f_band_low` (keep `f_band_low`).

### Step 2: LightGBM's own importance

1. Train LightGBM on the remaining features with the usual **5 user folds**
   (same folds and metrics as every other model).
2. For each fold, read the built-in importance with **`importance_type="gain"`**
   (not the default `"split"`, which only counts how often a feature is used).
3. Convert each fold's values to percentages; report **mean +- std over the 5 folds**.
   A feature important in only one fold is not reliable.
4. Drop features with near-zero gain, **retrain once**, and confirm PR AUC, recall and
   specificity do not get worse.

**Limitation:** gain is measured on training data. Warning signs: a feature with high gain but
unstable across folds, or `f_band_high` near the top (interpolation and sensor-filter issues).
If either happens, run permutation importance on the test folds as a one-off check.

### Results to save for the report

1. Table of features removed by the correlation filter (pair, rho, kept, reason)
2. Bar chart of gain importance (mean +- std over 5 folds)
3. Metrics before vs after dropping weak features

---

## 7. Notes

- `f_band_high` (8-20 Hz) is the least trustworthy feature: the Fitbit delivered bursty
  40-50 Hz data that we interpolated (which smooths out fast changes), and its internal filters
  differ from the XIAO's LSM6DS3TR-C. A high rank is a reason for suspicion, not celebration.
- `peak_pos` near the end of the window means `post` features are short or NaN; the next
  window (1 s later) contains the same impact with more post-impact data.
- Reference: Kavuncuoglu et al., "Comprehensive analysis of feature-algorithm interactions for
  fall detection", Computational Intelligence, 2024 (`papers/`). Its ~98 % results come from
  peak-centred 0.5 s waist segments and do not transfer directly to streaming wrist windows.
