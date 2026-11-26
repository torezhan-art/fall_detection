# Fall Detection ML — Progress Update

*Updated: 17 November 2026*

Status of the dataset and TinyML model for the wrist-worn fall detector (XIAO nRF52840 Sense).

---

## Targets

- Accuracy and sensitivity of at least **95%** offline (M1).
- **int8** model running on the nRF52840 under TensorFlow Lite Micro, with a tensor arena under **15 KB** (goal: under 8 KB) and **≤ 1 percentage point** of accuracy lost to quantization (M2 / RS-1).
- Impact-to-alert latency of at most **3.0 s** end-to-end (REQ-2). After the gateway forwarding (≤ 1.0 s, IR-2) and the BLE advertising burst (≤ 0.5 s), this leaves **1.5 s for the watch to detect the fall**.

## Data

- **Dataset:** WEDA-FALL, a public dataset recorded with a wrist-worn smartwatch (Fitbit Sense): 25 subjects, 969 trials, accelerometer + gyroscope. It replaced SisFall, which is recorded at the waist and doesn't match our wrist placement.
- **Preprocessing:**
  - The watch's timestamps arrive in bursts, so the data is resampled to a steady **50 Hz**.
  - 57 trials have broken timestamps and are flagged.
- **Windowing:** **2 s windows** (100 samples × 6 axes), with a **1 s step** between windows. The model makes one decision per second.
- **Evaluation:** **5-fold cross-validation with folds split by subject**. Test subjects never appear in training, which gives an honest estimate for new users.
- Our own recordings with the LSM6DS3 will be added after IREC approval.

## Baseline: acceleration threshold

A fixed threshold on the size of the acceleration (the classic impact detector).
- Precision only **34%**, i.e. about 2 of every 3 alarms are false.
- Only **58%** of daily-activity (ADL) recordings had no false alarm.

This showed that an ML model is needed.

## Model A: LightGBM (gradient-boosted trees)

- **Features:**
  - 30 hand-crafted features that don't depend on how the watch is rotated on the wrist: impact peak, free-fall phase, change in orientation, stillness after impact, signal energy, and so on.
  - Redundancy removed with a Spearman correlation filter (|ρ| > 0.95), giving 27 features.
  - Then the **15 most important** were kept, by LightGBM gain importance.
- **Tuning:** nested random search over 51 hyperparameter sets. Each set is scored by **precision when recall is 95%**, so recall stays high while false alarms are minimized.
- **Results** (window level, cut-off set for about 95% recall, 5-fold mean):
  - Recall **94.9%**, precision **71.1%**, specificity **97.0%**, accuracy **96.9%**
  - ROC AUC **99.2%**, PR AUC **92.3%**

## Model B: 1D-CNN

- A small Conv1D network with **6,897 parameters**, working directly on the raw signal.
- **8 input channels:** the 6 IMU axes plus the size of the acceleration vector (|a|) and of the rotation-rate vector (|ω|).
- Input scaling that doesn't depend on how the watch is rotated.
- Early stopping on a separate validation fold.
- **Results** (window level, 5-fold mean, cut-off 0.5): accuracy **96.1%**, recall **91.5%**, precision **67.2%**, ROC AUC **98.2%**.
- **Rotation augmentation tried and rejected:** randomly rotating the training windows increased false alarms by about 60%.

## Recording-level comparison

Window scores don't directly tell us what the user experiences, so both models were also judged on **whole recordings**. Alarm cut-offs were chosen on other folds.

| Model | Falls detected | ADL recordings with no false alarm | Median latency |
|---|---|---|---|
| Baseline threshold | — | 58% | — |
| LightGBM | 94.6% | 81.4% | 0.6 s |
| 1D-CNN | 95.0% | 80.7% | 0.6 s |
| 1D-CNN, int8 | 94.7% | 81.4% | 0.6 s |
| **LightGBM + 1D-CNN (averaged output)** | **95.3%** | **89.9%** | — |

- The two models are **tied** individually.
- Their errors fall on **different activities**, so averaging their outputs roughly **halves the false-alarm rate**.
- The activities that still cause most false alarms are **jumping, hitting the hand on a table and stumbling**. All of them produce fall-like impact spikes.
- **Rules that need several windows** gave only a small reduction in false alarms, and they make alarms slower:
  - "2 windows in a row" pushed about half (52%) of fall detections past the watch's 1.5 s detection budget, which would put the 3.0 s end-to-end requirement (REQ-2) at risk.
  - A hybrid rule (also alarm on one very confident window) still left about 7% over the limit.
  - So we keep the **single-window decision**.

## Quantization (int8) for the microcontroller

The CNN was converted to a **fully int8 TFLite model** with post-training quantization. A sample of training data was used to measure the range of each layer's values (calibration). Inputs, weights, activations and outputs are all int8.

| | Float32 | int8 |
|---|---|---|
| Accuracy (cut-off 0.5) | 96.08% | 96.03% (−0.05 pt) |
| Accuracy (cut-off for 95% recall) | 94.18% | 93.81% (−0.37 pt) ✅ |
| Same decisions as float | — | 99.8–99.9% |
| Model size | — | **17.4 KB** |
| Tensor arena (estimated) | — | **~5.1 KB** ✅ |

- The operations used are all supported by TFLite Micro: Conv2D, FullyConnected, MaxPool, Mean, Logistic, Reshape, ExpandDims.
- The model has been exported as a C array for the firmware, together with **golden test vectors**: reference inputs and outputs to check that the chip gives exactly the same results.
- This export is a **demo trained on one fold only**. The final model will be trained on all subjects.
- Quantization-aware training isn't needed, since post-training quantization is already well within the 1-point limit.

## Milestone status

| Milestone | Status |
|---|---|
| M1: offline accuracy ≥ 95% | ✅ Window-level accuracy 96–97%. Recording-level sensitivity is about 95% (95.3% for the averaged models). |
| M2 / RS-1: int8 with ≤ 1 pt drop, arena < 15 KB | ✅ Met in simulation on the PC (−0.37 pt, ~5.1 KB) |
| On-chip deployment and test | ⏳ Waiting for the XIAO board |

## Interfaces to agree on with the team

- **Firmware:**
  - IMU settings: 50 Hz output rate, accelerometer and gyroscope ranges, units (g and dps) and axis orientation. These must match what the model expects.
  - A buffer that holds the last 100 samples and triggers inference every second.
  - The alarm message format over BLE.
- **Hardware / power:** the inference duty cycle (one inference per second) for the battery-life estimate. The on-chip timing still needs to be measured.
- **Everyone:** the CNN-only vs. combined-models decision.
