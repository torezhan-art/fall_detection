# Fall Detection ML — Progress Summary

*Last updated: 2026-11-17*

ENG 400 capstone: wrist-worn fall detector on the Seeed XIAO nRF52840 Sense.
My part covers the **dataset and the TinyML model**.

Targets:
- Offline accuracy and sensitivity of at least 95% (M1).
- int8 model on the chip with a working-memory (tensor arena) footprint under 15 KB, goal under 8 KB (M2 / RS-1).
- No more than 1 point of accuracy lost to quantization (RS-1).
- Impact-to-alert latency of at most 3.0 s end-to-end (REQ-2). Minus the gateway forwarding (≤ 1.0 s, IR-2) and the BLE advertising burst (≤ 0.5 s), this leaves a **1.5 s detection budget on the watch** (`DETECTION_BUDGET_S` in `decision_rules.py`).

---

## 1. Data pipeline

| Step | Details | Code |
|---|---|---|
| Dataset | **WEDA-FALL**: wrist-worn Fitbit Sense, 50 Hz, 969 trials, 25 users. It replaced SisFall, which is recorded at the waist. | — |
| Loading | Timestamps arrive in bursts, so the signals are resampled to a steady 50 Hz. 57 trials with broken timestamps are flagged. | `codes/weda.py` |
| Windowing | 6 axes (accelerometer + gyroscope), 2 s windows (100 samples), 1 s step between windows. The 75 windows with a flat gyroscope signal are kept. | `codes/windows.py` → `data/processed/windows.npz`, `windows_meta.csv` |
| Splits | 5 folds by user: each fold holds out different people (no user appears in both training and test). | `codes/splits.py` |

## 2. Baseline: acceleration threshold

`codes/baseline_threshold.py`: a fixed threshold on the size of the acceleration.
Precision is **34.4%**. Only **57.5%** of daily-activity (ADL) recordings have no false alarm.
The ML models below are compared against this.

## 3. LightGBM

### Features
- 30 features that don't depend on how the watch is rotated, in 8 groups (spec: `docs/features.md`, code: `src/features/features.py`).
- Correlation filter: when two features have Spearman |ρ| > 0.95, the more expensive one to compute is dropped. This removed `a_cv`, `w_std` and `a_std`, leaving 27 features (`src/features/select_corr.py` → `selected_features.txt`).
- The 15 most important features by LightGBM gain over the 5 folds were kept (`top15_features.txt`).

### Training and tuning
- `src/models/train_lightgbm.py` trains with fixed settings.
- `src/models/tune_lightgbm.py` runs a nested random search over 51 candidate settings. Each candidate is scored by precision when recall is 95%.
- The winning settings use bigger trees and stronger regularization: 54 leaves, 427 trees, learning rate 0.079, reg_lambda 6.7, min_child_samples 59 (saved in `results/lgbm_tuned_top15_features_final_params.json`).

### Results (5-fold mean, window level, cut-off set for about 95% recall)

| Metric | All 27 features | Top 15, fixed settings | **Top 15, tuned** |
|---|---|---|---|
| Recall | 96.2 | 96.7 | 94.9 |
| Precision | 62.9 | 64.8 | **71.1** |
| Specificity | 95.6 | 95.9 | **97.0** |
| Accuracy | 95.7 | 96.0 | **96.9** |
| ROC AUC | 99.2 | 99.2 | 99.2 |
| PR AUC | 92.2 | 92.4 | 92.3 |
| ADL recordings with no false alarm | 66.4 | 68.4 | **76.2** |

Most remaining false alarms come from **D08 jump**, **D09 hitting a table** and **D07 stumble**.

## 4. 1D-CNN

`src/models/train_cnn.py` trains a small Conv1D network:
- **6,897 weights.**
- 8 input channels: the 6 axes plus the size of the acceleration vector (|a|) and of the rotation-rate vector (|ω|).
- Input scaling that doesn't depend on how the watch is rotated.
- Training stops early based on validation fold k+1.
- Two variants: no augmentation (`none`) and random-rotation augmentation (`rotate`). Models are saved in `models/cnn_<augment>/`.

### Results (5-fold mean, window level)

| Metric | CNN, no augmentation | CNN, rotation augmentation |
|---|---|---|
| Accuracy @0.5 | 96.1 | 94.3 |
| Precision @0.5 | 67.2 | 56.7 |
| Recall @0.5 | 91.5 | 95.1 |
| Accuracy @ ~95% recall | 94.2 | 94.8 |
| ROC AUC | 98.2 | 98.7 |
| PR AUC | 89.5 | 87.4 |

Notes:
- The cut-off for 95% recall is picked on a single validation fold, so it is unreliable for the CNN. In fold 2 the cut-off came out at 0.017 and precision fell to 31%. The recording-level comparison in section 5 is more trustworthy.
- Full rotation augmentation **hurts**: precision drops and false alarms rise.

## 5. Fair comparison at the recording level

`src/models/decision_rules.py` judges whole recordings, using alarm cut-offs chosen on the other folds. Numbers are 5-fold means with the single-window alarm rule.

| Model | Falls caught | ADL recordings clean | False alarms per hour | Median delay |
|---|---|---|---|---|
| LightGBM (tuned, top 15) | 94.6 | 81.4 | 68.8 | 0.6 s |
| CNN, no augmentation | 95.0 | 80.7 | 67.9 | 0.6 s |
| CNN, rotation augmentation | 95.6 | 70.0 | 110.1 | 0.6 s |
| CNN, no augmentation, int8 | 94.7 | 81.4 | 65.6 | 0.6 s |
| **Average of LightGBM + CNN (no aug.)** | **95.3** | **89.9** | **about half** | — |

- At the recording level, LightGBM and the CNN without augmentation are **tied**.
- **Averaging their outputs** gives the best result: the two models make mistakes on different activities, so false alarms roughly halve. This result is not yet saved as a file in `results/`.
- **"2 windows in a row" rule:** small gain (ADL recordings clean 81 → 85%). Jumps and table hits are unchanged, because their impacts also span 2 windows. 52% of falls would take longer than the watch's 1.5 s detection budget, which puts REQ-2 (3.0 s end-to-end) at risk.
- **Hybrid rule** (alarm on one very confident window, otherwise require 2 in a row): the same small gain, with 7% of falls still over the 1.5 s detection budget.
- **Decision: keep the single-window rule.**

## 6. int8 quantization (CNN, no augmentation)

`src/models/quantize_cnn.py` converts the model to int8 (TFLite reference kernels).

| | Float | int8 | Change |
|---|---|---|---|
| Accuracy @0.5 | 96.08 | 96.03 | −0.05 pt |
| Accuracy @ ~95% recall | 94.18 | 93.81 | −0.37 pt |
| ROC AUC | 98.18 | 98.03 | −0.15 |
| PR AUC | 89.5 | 82.9 | −6.6 (see note below) |

- The float and int8 models make the same decision on **99.8–99.9%** of windows. **RS-1 is met on the PC.**
- The PR AUC drop isn't a real loss. The int8 output has only 256 levels, so 58% of windows score exactly 0.
- `.tflite` file size: **17.4 KB**. Estimated working memory: **about 5.1 KB** (3.1 KB activations + 2 KB margin), under the 8 KB goal.
- Operations used: CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, LOGISTIC, MAX_POOL_2D, MEAN, RESHAPE.
- `fold0_int8.h` and `fold0_golden.npz` are written. These are **demo files only**: the final model must be trained on all users.

## 7. Status against milestones

| Milestone | Status |
|---|---|
| M1: offline accuracy ≥ 95% | Met at window level (LightGBM 96.9, CNN 96.1). At recording level, sensitivity is 94.6–95.0 for single models and 95.3 for the average. |
| M2 / RS-1: int8, ≤ 1 pt drop, working memory < 15 KB | Met on the PC with the real TensorFlow Lite Micro runtime (−0.37 pt, **6.09 KB measured arena**, section 6b). Not yet tested on the chip: the board hasn't arrived. |

## 6b. TensorFlow Lite Micro on the PC (2026-11-18)

`tflm_host/` builds the real TFLM runtime (reference kernels) for Windows and runs the int8 model from `fold0_int8.h`. Details are in `tflm_host/README.md`. The build needs no admin rights: pip provides `ziglang` (clang), `cmake` and `ninja`.

Each window is checked two ways:
- **A:** Python's scaled input → TFLM.
- **B:** raw IMU window → C pre-processing (`tflm_host/preprocess.h`, firmware-ready) → TFLM.

| Check | Result |
|---|---|
| Header constants vs model | Match |
| C pre-processing vs Python | Identical input bytes (0 of 1.45 M differ) |
| Golden windows | 20 / 20 bit-exact |
| All 1,807 fold-0 test windows | 1,805 bit-exact, **0 alarm decisions changed** |
| **Tensor arena** (32-bit build, plain `MicroInterpreter`) | **6,088 B = 5.95 KB**, measured (the earlier estimate was ~5.1 KB). Under both the 15 KB requirement and the 8 KB goal. |

**The 2 differing windows:**
- A layer-by-layer comparison (`layer_dump` + `layer_diff.py`) shows every layer matches up to MEAN. The difference first appears as a ±1 step in a FULLY_CONNECTED output.
- Python's TFLite (TF 2.21) requantizes with single rounding; TFLM uses double rounding. This was confirmed by recomputing the output layer of window 979 by hand.
- The sigmoid stretched one case to 5 output steps (91 vs 96, both far above the cut-off).
- So **the board's golden outputs are now TFLM's** (`tflm_host/vectors/*_tflm_expected.i8`), not Python's.

## 8. Next steps

- [ ] Save the LightGBM + CNN average as a script and results file (currently not reproducible from `results/`).
- [ ] Decide whether to put both models on the chip (CNN plus LightGBM feature extraction and trees) or deploy the CNN alone.
- [ ] Train the final model on all users and regenerate the int8 header file.
- [x] Run TFLite Micro on the PC: golden vectors, firmware pre-processing, real arena size (section 6b).
- [ ] Replay whole recordings through the C streaming pipeline (ring buffer → pre-processing → TFLM → alarm) and compare with the recording-level Python results.
- [ ] Test on the chip (T4) once the XIAO board arrives: real memory use, speed, and comparison against the TFLM golden outputs.
- [ ] Record our own wrist data after IREC approval.

## Results file index

| File | Contents |
|---|---|
| `results/lgbm_*_folds.csv` | LightGBM per-fold window metrics |
| `results/lgbm_tuned_top15_features_search.csv` | Tuning search log |
| `results/cnn_{none,rotate}_folds.csv` | CNN per-fold window metrics |
| `results/cnn_none_int8_folds.csv` | Float vs int8 comparison |
| `results/rules_*_predictions.csv` | Recording-level results for the single, 2-in-a-row and hybrid rules |
