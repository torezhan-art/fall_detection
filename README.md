# Wrist Fall Detection — ML and TinyML

Machine-learning part of the ENG 400 capstone *"Wearable Edge-AI Fall Detection System for Workers and Elderly Campus Residents"* (Nazarbayev University, NU-CaLLs), supervised by Prof. Mohammad Hashmi.

The wearable is a Seeed XIAO nRF52840 Sense. It samples its 6-axis IMU at about 50 Hz and decides "fall / no fall" once per second, using the last 2 s of motion and an int8 1D-CNN under TensorFlow Lite Micro.

**Full progress report:** [`docs/progress.md`](docs/progress.md) · team summary: [`docs/progress_for_team.md`](docs/progress_for_team.md)

## Results so far (5-fold cross-validation, test subjects never seen in training)

| | LightGBM (15 features, tuned) | 1D-CNN (6,897 parameters) |
|---|---|---|
| Window accuracy | 96.9% | 96.1% |
| Falls caught (whole recordings) | 94.6% | 95.0% |
| Daily-activity recordings with no false alarm | 81.4% | 80.7% |

- **int8 quantization of the CNN:** −0.37 pt accuracy, 17.4 KB model.
- **TensorFlow Lite Micro on a PC:** identical alarm decisions on all 1,807 test windows; **measured tensor arena 5.95 KB**. See [`tflm_host/README.md`](tflm_host/README.md).

## Layout

| Folder | Contents |
|---|---|
| `codes/` | Data loading and windowing (`weda.py`, `windows.py`, `splits.py`), threshold baseline, plots |
| `src/features/` | Feature extraction and correlation filter for LightGBM |
| `src/models/` | LightGBM training and tuning, CNN training, int8 quantization, decision rules |
| `results/` | Per-fold metrics and predictions for every experiment (the evidence for `docs/progress.md`) |
| `models/` | Trained CNNs (`.keras`), int8 models (`.tflite`) and the firmware header `fold0_int8.h` |
| `tflm_host/` | TensorFlow Lite Micro PC test harness and firmware-ready pre-processing (`preprocess.h`) |
| `figures/`, `presentation/` | Plots and the progress deck |
| `docs/` | Feature spec and progress reports |

## Data (not included)

The models are trained on **WEDA-FALL** ([github.com/joaojtmarques/WEDA-FALL](https://github.com/joaojtmarques/WEDA-FALL)): wrist-worn Fitbit Sense, 50 Hz, 25 subjects. The dataset has no licence, so neither it nor any file containing its sensor values is published here.

To reproduce:
1. Clone WEDA-FALL into `data/raw/WEDA-FALL`.
2. Run `codes/windows.py` to create the windows and `src/features/features.py` to compute the features.
3. Run the scripts in `src/models/`.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
