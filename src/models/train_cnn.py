"""Step 9: a small 1D-CNN on the raw 2 s windows, evaluated on the same 5 user folds.

Instead of hand-made features, the network learns its own patterns from the raw signal.

Input (per window, 100 time steps x 8 channels):
    ax, ay, az [g], gx, gy, gz [deg/s]   the raw sensor axes
    |a|, |w|                             magnitudes: the same however the watch is worn

Network (~6.7k weights, small enough for the XIAO nRF52840):
    Conv1D 16 x5 -> MaxPool /2 -> Conv1D 32 x5 -> MaxPool /2 -> Conv1D 32 x3
    -> GlobalAveragePooling -> Dropout -> Dense 16 -> Dense 1 (sigmoid = fall probability)

For every test fold k (users never seen in training):
  * validation fold = the next fold (k+1); training folds = the other 3
  * scaling constants are computed on the training users only
  * training stops when PR AUC on the validation users stops improving (early stopping)
  * the sens95 cut-off (catch >= 95 % of falls) is chosen on the validation users
  * the test users are used once, at the end, for all metrics

--augment rotate  randomly rotates every training window in 3D each epoch (accelerometer and
                  gyroscope together), so the network cannot rely on how the Fitbit was worn.
                  This protects against the XIAO's axes pointing differently.

Run from the project folder:
    .venv\\Scripts\\python.exe src\\models\\train_cnn.py --augment none
    .venv\\Scripts\\python.exe src\\models\\train_cnn.py --augment rotate
Writes results/cnn_<augment>_folds.csv, results/cnn_<augment>_predictions.csv
       models/cnn_<augment>/fold<k>.keras + fold<k>_scaling.json   (used by the int8 step)
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's start-up messages

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.transform import Rotation
from sklearn.metrics import average_precision_score

from train_lightgbm import (PROJECT, RESULTS, SEED, false_alarms_by_activity, print_fold_table,
                            threshold_for_recall, window_and_trial_metrics)

PROCESSED = PROJECT / "data" / "processed"
MODELS = PROJECT / "models"

# Training settings, fixed in advance (not tuned on test folds).
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8               # stop after 8 epochs without a better validation PR AUC


# ---------------------------------------------------------------- inputs
def add_magnitudes(X):
    """(n, 100, 6) raw windows -> (n, 100, 8) with |a| and |w| appended."""
    acc_magnitude = np.linalg.norm(X[..., 0:3], axis=-1, keepdims=True)
    gyro_magnitude = np.linalg.norm(X[..., 3:6], axis=-1, keepdims=True)
    return np.concatenate([X, acc_magnitude, gyro_magnitude], axis=-1).astype(np.float32)


def fit_scaling(X8):
    """Scaling constants from the TRAINING windows only.

    The 3 accelerometer axes share one scale (their RMS), and so do the 3 gyroscope axes.
    A shared RMS does not change when the watch is rotated, so the scaling does not depend
    on how the device is worn. The two magnitudes are standardised (mean 0, std 1).
    """
    return {
        "acc_rms": float(np.sqrt(np.mean(X8[..., 0:3] ** 2))),
        "gyro_rms": float(np.sqrt(np.mean(X8[..., 3:6] ** 2))),
        "acc_mag_mean": float(X8[..., 6].mean()), "acc_mag_std": float(X8[..., 6].std()),
        "gyro_mag_mean": float(X8[..., 7].mean()), "gyro_mag_std": float(X8[..., 7].std()),
    }


def apply_scaling(X8, s):
    """Apply the constants from fit_scaling. The watch firmware must do exactly the same."""
    out = X8.copy()
    out[..., 0:3] /= s["acc_rms"]
    out[..., 3:6] /= s["gyro_rms"]
    out[..., 6] = (out[..., 6] - s["acc_mag_mean"]) / s["acc_mag_std"]
    out[..., 7] = (out[..., 7] - s["gyro_mag_mean"]) / s["gyro_mag_std"]
    return out


def model_input(X_raw, scaling):
    return apply_scaling(add_magnitudes(X_raw), scaling)


def randomly_rotate(X, rng):
    """Give every window its own random 3D rotation (same rotation for acc and gyro)."""
    R = Rotation.random(len(X), rng=rng).as_matrix().astype(np.float32)   # (n, 3, 3)
    acc = np.einsum("nij,ntj->nti", R, X[..., 0:3])
    gyro = np.einsum("nij,ntj->nti", R, X[..., 3:6])
    return np.concatenate([acc, gyro], axis=-1)


# ---------------------------------------------------------------- the network
def build_model(n_channels=8):
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(100, n_channels)),
        layers.Conv1D(16, kernel_size=5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),                                   # 100 -> 50 time steps
        layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),                                   # 50 -> 25 time steps
        layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),                          # 25 x 32 -> 32 numbers
        layers.Dropout(0.2),                                      # only active in training
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),                    # fall probability
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="binary_crossentropy")
    return model


def predict(model, x):
    return model.predict(x, batch_size=1024, verbose=0).ravel()


def train_with_early_stopping(X_train, y_train, X_val, y_val, augment, seed):
    """Train one network. Returns (model, scaling, best epoch, best validation PR AUC)."""
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    scaling = fit_scaling(add_magnitudes(X_train))
    x_val = model_input(X_val, scaling)
    # fall windows are rare (1 in 13), so each one counts (#not-fall / #fall) times more
    class_weight = {0: 1.0, 1: float((y_train == 0).sum() / (y_train == 1).sum())}

    model = build_model()
    best_score, best_epoch, best_weights = -1.0, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        X_epoch = randomly_rotate(X_train, rng) if augment == "rotate" else X_train
        model.fit(model_input(X_epoch, scaling), y_train, batch_size=BATCH_SIZE, epochs=1,
                  class_weight=class_weight, shuffle=True, verbose=0)

        score = average_precision_score(y_val, predict(model, x_val))   # validation PR AUC
        if score > best_score:
            best_score, best_epoch, best_weights = score, epoch, model.get_weights()
        elif epoch - best_epoch >= PATIENCE:
            break

    model.set_weights(best_weights)
    return model, scaling, best_epoch, best_score


# ---------------------------------------------------------------- the 5 folds
def cross_validate(X, meta, augment):
    folds = sorted(meta.fold.unique())
    y = meta.label.to_numpy()
    out_dir = MODELS / f"cnn_{augment}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    predictions = meta[["trial_id", "user", "activity", "fold", "label"]].copy()
    predictions["probability"] = np.nan
    predictions["threshold_sens95"] = np.nan

    for k in folds:
        start = time.time()
        val_fold = folds[(folds.index(k) + 1) % len(folds)]
        is_test = (meta.fold == k).to_numpy()
        is_val = (meta.fold == val_fold).to_numpy()
        is_train = ~is_test & ~is_val

        model, scaling, best_epoch, val_pr_auc = train_with_early_stopping(
            X[is_train], y[is_train], X[is_val], y[is_val], augment, seed=SEED + int(k))

        sens95 = threshold_for_recall(predict(model, model_input(X[is_val], scaling)), y[is_val])
        probability = predict(model, model_input(X[is_test], scaling))

        test_meta = meta[is_test]
        for rule, cutoff in [("0.5", 0.5), ("sens95", sens95)]:
            fold_rows.append({"fold": int(k), "rule": rule,
                              **window_and_trial_metrics(test_meta, probability, cutoff)})
        predictions.loc[is_test, "probability"] = probability
        predictions.loc[is_test, "threshold_sens95"] = sens95

        model.save(out_dir / f"fold{k}.keras")
        (out_dir / f"fold{k}_scaling.json").write_text(json.dumps(
            {**scaling, "sens95_cutoff": float(sens95), "validation_fold": int(val_fold),
             "best_epoch": best_epoch}, indent=2))
        print(f"  fold {k}: train {is_train.sum()}, val {is_val.sum()} (fold {val_fold}), "
              f"test {is_test.sum()} | best epoch {best_epoch}, val PR AUC {100 * val_pr_auc:.1f} %, "
              f"sens95 cut-off {sens95:.3f}  [{time.time() - start:.0f} s]")

    return pd.DataFrame(fold_rows), predictions


def compare_with_lightgbm(cnn_folds):
    """Mean at the sens95 cut-off: tuned LightGBM (if its results exist) vs this CNN."""
    columns = ["recall", "precision", "specificity", "accuracy", "ROC AUC", "PR AUC",
               "trial sens", "trial spec"]
    rows = {}
    lgbm_csv = RESULTS / "lgbm_tuned_top15_features_folds.csv"
    if lgbm_csv.exists():
        rows["LightGBM (tuned, top 15)"] = pd.read_csv(lgbm_csv)
    rows["1D-CNN"] = cnn_folds
    table = pd.DataFrame({name: 100 * folds[folds.rule.astype(str) == "sens95"][columns].mean()
                          for name, folds in rows.items()}).T
    return table.round(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--augment", choices=["none", "rotate"], default="rotate")
    args = parser.parse_args()

    X = np.load(PROCESSED / "windows.npz")["X"].astype(np.float32)
    meta = pd.read_csv(PROCESSED / "windows_meta.csv")
    print(f"1D-CNN, augment = {args.augment}: {len(X)} windows, input {X.shape[1]} x 8 channels, "
          f"{build_model().count_params()} weights")

    folds, predictions = cross_validate(X, meta, args.augment)

    pd.set_option("display.width", 220)
    for rule in ["0.5", "sens95"]:
        print(f"\n=== decision threshold: {rule} ===  (all values in %, except the cut-off)")
        print_fold_table(folds, rule)

    print("\nComparison at the sens95 cut-off, mean over the 5 folds (%):")
    print(compare_with_lightgbm(folds).to_string())

    print("\nFalse alarms per daily activity at the sens95 cut-off (% of that activity's windows):")
    print(false_alarms_by_activity(predictions).to_string())

    RESULTS.mkdir(exist_ok=True)
    folds.to_csv(RESULTS / f"cnn_{args.augment}_folds.csv", index=False)
    predictions.to_csv(RESULTS / f"cnn_{args.augment}_predictions.csv", index=False)
    print(f"\nSaved results/cnn_{args.augment}_*.csv and models/cnn_{args.augment}/")
