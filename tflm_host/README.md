# tflm_host — TensorFlow Lite Micro on the PC

Runs the int8 fall-detection CNN (`models/cnn_none/fold0_int8.h`) with the real
TensorFlow Lite Micro (TFLM) runtime on Windows, so the model, the firmware pre-processing
and the arena size can be checked before the XIAO nRF52840 board arrives.

## Run it

From Git Bash in the project folder:

```bash
.venv/Scripts/python.exe tflm_host/export_vectors.py   # once, or after re-quantizing
bash tflm_host/build.sh        # 32-bit build (pointer size like Cortex-M), builds + runs
bash tflm_host/build.sh 64     # optional 64-bit build
```

The first build compiles TFLM (~155 files, a few minutes); later builds only recompile what changed.

## What is in here

| File | Purpose |
|---|---|
| `preprocess.h` | **Firmware-ready** pre-processing: raw window (100 × 6, g and deg/s) → \|a\|, \|ω\| → scaling → int8 input. Copy into the firmware as is. |
| `main.cc` | The check (`tflm_check`): runs every window two ways and compares with Python, measures the arena, writes golden outputs for the board. |
| `layer_dump.cc` + `layer_diff.py` | Debug tools: dump every layer's output for one window from TFLM and compare it with TFLite (Python), layer by layer. |
| `export_vectors.py` | Writes the test windows (`vectors/`) from the golden file and all fold-0 test windows. |
| `CMakeLists.txt`, `build.sh` | Build: TFLM core + reference kernels as a static library (kernel list read from TFLM's own `sources.inc`). |
| `.venv-tools/` | Build tools from pip, no admin rights needed: `ziglang` (clang C/C++ compiler), `cmake`, `ninja`. |
| `third_party/` | TFLM source (GitHub, commit 9611ed2) and its dependencies (flatbuffers, kissfft, printf), fetched with TFLM's own `download_dependencies.py`. |

## The two checks per window

- **A:** Python's scaled input, quantized in C, then TFLM. Checks the runtime and model.
- **B:** raw IMU window, then `preprocess.h`, then TFLM. Also checks the firmware pre-processing.

## Results (2026-11-18, fold-0 model)

| | Result |
|---|---|
| Header constants (scales, zero points, shape) match the model | yes |
| C pre-processing vs Python, input bytes | **identical** (0 of 1,445,600 differ) |
| Golden windows (20) | 20 / 20 bit-exact |
| All fold-0 test windows (1,807) | 1,805 / 1,807 bit-exact; **0 alarm decisions changed** |
| Tensor arena, 32-bit (plain `MicroInterpreter`) | **6,088 B (5.95 KB)**: under the 15 KB requirement and the 8 KB goal |
| Tensor arena, 64-bit | 7,440 B (larger only because pointers are 8 bytes) |
| Ops to register (`MicroMutableOpResolver<7>`) | CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, LOGISTIC, MAX_POOL_2D, MEAN, RESHAPE |

### Why 2 windows are not bit-exact

`layer_dump` / `layer_diff.py` show that every layer matches up to MEAN. The difference first appears in a **FULLY_CONNECTED** output, off by 1 step:

- Python's TFLite (TF 2.21) requantizes with **single rounding**.
- TFLM uses **double rounding** (the gemmlowp-style method).

Recomputing the output layer of window 979 by hand gives 12 with single rounding (= TFLite) and 13 with double rounding (= TFLM).

When an accumulator lands near a rounding boundary the two disagree by 1. The sigmoid can stretch that to a few output steps (window 979: 91 vs 96, both far above the alarm cut-off).

This is a difference between the two runtimes, not a bug in our code. The board runs TFLM, so:

- **Board golden outputs = `vectors/<set>_tflm_expected.i8`** (written by `tflm_check`), not the Python ones.
- If the firmware uses the CMSIS-NN optimized kernels instead of the reference ones, re-check against these files. CMSIS-NN is designed to match, but should be verified.

## Notes for the firmware

- Use `lrintf` (round half to even) for quantization, as in `preprocess.h`. `roundf` would occasionally differ from Python.
- Allocate an arena of ~6.1 KB plus a small margin (e.g. 7 KB). On the board, check `interpreter.arena_used_bytes()` once: ARM alignment and CMSIS-NN scratch buffers can change it slightly.
- The PC Invoke() time (~0.2–0.35 ms) says nothing about the nRF52840; it has to be measured on the board.
- `fold0_int8.h` is a demo model trained without fold 0. Re-run `export_vectors.py` and `build.sh` after the final all-user model is quantized.
