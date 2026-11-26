"""Step 10: convert the 1D-CNN to int8 TensorFlow Lite and check what it costs (RS-1).

For every fold's model (models/cnn_<augment>/fold<k>.keras, from train_cnn.py):
  1. convert to a fully int8 .tflite model. The converter needs a "representative dataset"
     to learn the value range of every layer: 500 windows from that fold's TRAINING users.
  2. run the int8 model with TensorFlow Lite's reference kernels (the plain implementations,
     closest to what TensorFlow Lite Micro does on the nRF52840) on the same test users
  3. compare float vs int8 with the same metrics. The sens95 cut-off of each model is chosen
     on its own validation-fold predictions, exactly as in train_cnn.py, because the watch
     will compare the int8 model's output with the cut-off.

RS-1 asks for an accuracy drop of at most 1 point (float -> int8).

Also reports, per fold: .tflite size (flash) and an estimate of the TensorFlow Lite Micro
arena (RAM): the largest set of layer outputs that must exist at the same time, plus a
margin for TensorFlow Lite Micro's own bookkeeping. The exact arena is read on the board.

For the firmware it writes, for fold 0 only (a demonstration until the final model is
trained on all users):
  models/cnn_<augment>/fold0_int8.h       the model as a C array + scaling + quantisation
  models/cnn_<augment>/fold0_golden.npz   test windows with the expected int8 output, to
                                          check the C++ pre-processing on the board

Run from the project folder:  .venv\\Scripts\\python.exe src\\models\\quantize_cnn.py
Writes results/cnn_<augment>_int8_folds.csv, results/cnn_<augment>_int8_predictions.csv
"""
import argparse
import json
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
warnings.filterwarnings("ignore", message=".*Statistics for quantized inputs.*")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")   # hide harmless "retracing" notices from model.predict

from train_cnn import MODELS, PROCESSED, model_input, predict
from train_lightgbm import PROJECT, RESULTS, SEED, threshold_for_recall, window_and_trial_metrics

N_REPRESENTATIVE = 500
TFLM_OVERHEAD_BYTES = 2048      # rough allowance for TensorFlow Lite Micro's own structures
N_GOLDEN = 20                   # test windows saved as golden vectors


# ---------------------------------------------------------------- conversion
def convert_to_int8(model, representative_windows):
    """Fully int8 model: int8 weights, int8 activations, int8 input and output."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([w[None]] for w in representative_windows)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def interpreter_for(tflite_bytes, batch=None):
    """Interpreter with the reference kernels and no speed-up delegate (closest to TFLM)."""
    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    if batch is not None:
        interpreter.resize_tensor_input(interpreter.get_input_details()[0]["index"],
                                        [batch, 100, 8])
    interpreter.allocate_tensors()
    return interpreter


def int8_probability(tflite_bytes, x):
    """Run the int8 model on scaled float inputs x (n, 100, 8). Returns (probability, raw int8)."""
    interpreter = interpreter_for(tflite_bytes, batch=len(x))
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    x_int8 = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(inp["index"], x_int8)
    interpreter.invoke()
    y_int8 = interpreter.get_tensor(out["index"]).ravel()
    return (y_int8.astype(np.float32) - out_zero) * out_scale, y_int8


# ---------------------------------------------------------------- memory estimate
def arena_estimate(tflite_bytes):
    """Peak bytes of layer outputs alive at the same time (int8 = 1 byte per value).

    A tensor is alive from the operation that creates it until the last operation that
    reads it. Weights are not counted: they stay in flash.
    """
    interpreter = interpreter_for(tflite_bytes)
    ops = interpreter._get_ops_details()
    size = {t["index"]: int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
            for t in interpreter.get_tensor_details()}
    model_input_index = interpreter.get_input_details()[0]["index"]

    created = {model_input_index: -1}
    last_used = {}
    for step, op in enumerate(ops):
        for t in op["outputs"]:
            created[t] = step
        for t in op["inputs"]:
            if t in created:                       # only activations, not weights
                last_used[t] = step

    peak = 0
    for step in range(len(ops)):
        alive = [t for t, start in created.items() if start <= step <= last_used.get(t, start)]
        peak = max(peak, sum(size[t] for t in alive))
    op_names = sorted({op["op_name"] for op in ops})
    return peak, op_names


# ---------------------------------------------------------------- firmware files
def write_c_header(path, tflite_bytes, scaling, cutoff, quantization):
    """The model as a C byte array, plus everything the firmware needs around it."""
    rows = [", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 12])
            for i in range(0, len(tflite_bytes), 12)]
    (in_scale, in_zero), (out_scale, out_zero) = quantization
    path.write_text(f"""// Fall-detection 1D-CNN, int8 TensorFlow Lite model. Generated by quantize_cnn.py.
// Input: 100 samples x 8 channels at 50 Hz:
//   ax, ay, az [g] / ACC_RMS, gx, gy, gz [deg/s] / GYRO_RMS,
//   (|a| - ACC_MAG_MEAN) / ACC_MAG_STD, (|w| - GYRO_MAG_MEAN) / GYRO_MAG_STD
// then quantised: q = round(x / INPUT_SCALE) + INPUT_ZERO_POINT, clipped to [-128, 127].
// Output: probability = (q - OUTPUT_ZERO_POINT) * OUTPUT_SCALE; alarm if >= FALL_CUTOFF.
#pragma once
#include <stdint.h>

const float ACC_RMS = {scaling['acc_rms']:.8g}f;
const float GYRO_RMS = {scaling['gyro_rms']:.8g}f;
const float ACC_MAG_MEAN = {scaling['acc_mag_mean']:.8g}f;
const float ACC_MAG_STD = {scaling['acc_mag_std']:.8g}f;
const float GYRO_MAG_MEAN = {scaling['gyro_mag_mean']:.8g}f;
const float GYRO_MAG_STD = {scaling['gyro_mag_std']:.8g}f;
const float INPUT_SCALE = {in_scale:.8g}f;
const int INPUT_ZERO_POINT = {in_zero};
const float OUTPUT_SCALE = {out_scale:.8g}f;
const int OUTPUT_ZERO_POINT = {out_zero};
const float FALL_CUTOFF = {cutoff:.8g}f;

alignas(16) const unsigned char fall_model[] = {{
  {(","+chr(10)+"  ").join(rows)}
}};
const unsigned int fall_model_len = {len(tflite_bytes)};
""")


def save_golden_vectors(path, X_raw, x_scaled, tflite_bytes, rows):
    """Raw test windows + what the board must produce for them."""
    probability, y_int8 = int8_probability(tflite_bytes, x_scaled)
    np.savez(path, raw_window=X_raw, scaled_input=x_scaled,
             expected_int8_output=y_int8, expected_probability=probability, row=rows)


# ---------------------------------------------------------------- main loop
def quantize_all_folds(X, meta, augment):
    model_dir = MODELS / f"cnn_{augment}"
    y = meta.label.to_numpy()
    rng = np.random.default_rng(SEED)
    fold_rows, size_rows = [], []
    predictions = meta[["trial_id", "user", "activity", "fold", "label"]].copy()
    predictions["probability"] = np.nan
    predictions["threshold_sens95"] = np.nan

    for k in sorted(meta.fold.unique()):
        model = tf.keras.models.load_model(model_dir / f"fold{k}.keras")
        info = json.loads((model_dir / f"fold{k}_scaling.json").read_text())
        is_test = (meta.fold == k).to_numpy()
        is_val = (meta.fold == info["validation_fold"]).to_numpy()
        is_train = ~is_test & ~is_val

        # 1. convert, calibrating on training users only
        train_rows = rng.choice(np.flatnonzero(is_train), N_REPRESENTATIVE, replace=False)
        tflite_bytes = convert_to_int8(model, model_input(X[train_rows], info))
        (model_dir / f"fold{k}_int8.tflite").write_bytes(tflite_bytes)

        # 2. float and int8 predictions; each model gets its own sens95 cut-off from validation
        x_val, x_test = model_input(X[is_val], info), model_input(X[is_test], info)
        for version, probability_of in [
                ("float", lambda x: predict(model, x)),
                ("int8", lambda x: int8_probability(tflite_bytes, x)[0])]:
            sens95 = threshold_for_recall(probability_of(x_val), y[is_val])
            test_probability = probability_of(x_test)
            for rule, cutoff in [("0.5", 0.5), ("sens95", sens95)]:
                fold_rows.append({"fold": int(k), "version": version, "rule": rule,
                                  **window_and_trial_metrics(meta[is_test], test_probability, cutoff)})
            if version == "int8":
                predictions.loc[is_test, "probability"] = test_probability
                predictions.loc[is_test, "threshold_sens95"] = sens95
                int8_cutoff = sens95
            else:
                float_test_probability = test_probability
        int8_test_probability = predictions.loc[is_test, "probability"].to_numpy()

        # 3. size and memory
        arena, op_names = arena_estimate(tflite_bytes)
        size_rows.append({
            "fold": int(k), "tflite KB": len(tflite_bytes) / 1024,
            "activations KB": arena / 1024,
            "arena estimate KB": (arena + TFLM_OVERHEAD_BYTES) / 1024,
            "mean |float - int8|": np.abs(float_test_probability - int8_test_probability).mean(),
            "same decision %": 100 * np.mean((float_test_probability >= 0.5) ==
                                             (int8_test_probability >= 0.5)),
        })

        if k == sorted(meta.fold.unique())[0]:
            interpreter = interpreter_for(tflite_bytes)
            quantization = (interpreter.get_input_details()[0]["quantization"],
                            interpreter.get_output_details()[0]["quantization"])
            write_c_header(model_dir / f"fold{k}_int8.h", tflite_bytes, info, int8_cutoff,
                           quantization)
            test_rows = np.flatnonzero(is_test)
            golden = np.concatenate([rng.choice(test_rows[y[is_test] == 1], N_GOLDEN // 2, replace=False),
                                     rng.choice(test_rows[y[is_test] == 0], N_GOLDEN // 2, replace=False)])
            save_golden_vectors(model_dir / f"fold{k}_golden.npz", X[golden],
                                model_input(X[golden], info), tflite_bytes, golden)
            print(f"  operations used: {', '.join(op_names)}")
        print(f"  fold {k}: {len(tflite_bytes) / 1024:.1f} KB model, "
              f"arena ~{(arena + TFLM_OVERHEAD_BYTES) / 1024:.1f} KB")

    return pd.DataFrame(fold_rows), pd.DataFrame(size_rows).set_index("fold"), predictions


def float_vs_int8(folds):
    """Mean over folds per version and rule, and the int8 - float difference (points)."""
    columns = ["accuracy", "recall", "precision", "specificity", "F1", "ROC AUC", "PR AUC",
               "trial sens", "trial spec"]
    mean = 100 * folds.groupby(["rule", "version"])[columns].mean()
    rows = []
    for rule in ["0.5", "sens95"]:
        float_row, int8_row = mean.loc[(rule, "float")], mean.loc[(rule, "int8")]
        rows += [(rule, "float", *float_row), (rule, "int8", *int8_row),
                 (rule, "change (int8 - float)", *(int8_row - float_row))]
    return pd.DataFrame(rows, columns=["cut-off", "model", *columns]).set_index(["cut-off", "model"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--augment", choices=["none", "rotate"], default="none")
    args = parser.parse_args()

    X = np.load(PROCESSED / "windows.npz")["X"].astype(np.float32)
    meta = pd.read_csv(PROCESSED / "windows_meta.csv")
    print(f"int8 conversion of models/cnn_{args.augment}/ (5 folds)")

    folds, sizes, predictions = quantize_all_folds(X, meta, args.augment)

    pd.set_option("display.width", 220)
    print("\nFloat vs int8, mean over the 5 test folds (%; last row of each block = change in points):")
    print(float_vs_int8(folds).round(2).to_string())

    print("\nSize and memory per fold (RS-1 / REQ-4: arena < 15 KB, objective < 8 KB):")
    print(sizes.round(3).to_string())

    folds.to_csv(RESULTS / f"cnn_{args.augment}_int8_folds.csv", index=False)
    predictions.to_csv(RESULTS / f"cnn_{args.augment}_int8_predictions.csv", index=False)
    print(f"\nSaved results/cnn_{args.augment}_int8_*.csv, models/cnn_{args.augment}/fold*_int8.tflite, "
          f"fold0_int8.h and fold0_golden.npz")
