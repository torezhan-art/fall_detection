"""Export test vectors for the TensorFlow Lite Micro PC test (tflm_host).

Writes raw binary files that main.cc reads (little-endian, row-major):
  vectors/<set>_raw.f32       (n, 100, 6) float32  raw IMU windows [g, deg/s]
  vectors/<set>_scaled.f32    (n, 100, 8) float32  model input after scaling (Python)
  vectors/<set>_expected.i8   (n,)        int8     int8 model output from TFLite (Python, reference kernels)
  vectors/<set>_n.txt         number of windows

Sets:
  golden  the 20 golden windows saved by quantize_cnn.py (fold0_golden.npz)
  fold0   every fold-0 test window, run through fold0_int8.tflite here

Run from the project folder:  .venv\\Scripts\\python.exe tflm_host\\export_vectors.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src" / "models"))

from quantize_cnn import int8_probability        # noqa: E402  (same reference interpreter)
from train_cnn import MODELS, PROCESSED, model_input  # noqa: E402

OUT = Path(__file__).resolve().parent / "vectors"


def write_set(name, raw, scaled, expected):
    OUT.mkdir(exist_ok=True)
    raw.astype("<f4").tofile(OUT / f"{name}_raw.f32")
    scaled.astype("<f4").tofile(OUT / f"{name}_scaled.f32")
    expected.astype(np.int8).tofile(OUT / f"{name}_expected.i8")
    (OUT / f"{name}_n.txt").write_text(str(len(raw)))
    print(f"  {name}: {len(raw)} windows")


def main():
    model_dir = MODELS / "cnn_none"
    tflite_bytes = (model_dir / "fold0_int8.tflite").read_bytes()
    scaling = json.loads((model_dir / "fold0_scaling.json").read_text())

    # 1. golden windows, exactly as saved for the board
    golden = np.load(model_dir / "fold0_golden.npz")
    _, recomputed = int8_probability(tflite_bytes, golden["scaled_input"])
    assert np.array_equal(recomputed, golden["expected_int8_output"]), "golden file is stale"
    write_set("golden", golden["raw_window"], golden["scaled_input"],
              golden["expected_int8_output"])

    # 2. every fold-0 test window
    X = np.load(PROCESSED / "windows.npz")["X"].astype(np.float32)
    meta = pd.read_csv(PROCESSED / "windows_meta.csv")
    is_test = (meta.fold == 0).to_numpy()
    raw = X[is_test]
    scaled = model_input(raw, scaling)
    _, expected = int8_probability(tflite_bytes, scaled)
    write_set("fold0", raw, scaled, expected)


if __name__ == "__main__":
    main()
