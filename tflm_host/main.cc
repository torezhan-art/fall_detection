// TensorFlow Lite Micro on the PC: checks the int8 fall-detection CNN before the board exists.
//
// For each vector set (see export_vectors.py) it runs every window twice:
//   A. Python's scaled input -> quantized here -> TFLM      (checks the TFLM runtime + model)
//   B. raw IMU window -> preprocess.h -> TFLM               (checks the firmware pre-processing too)
// and compares the int8 output byte with the Python TFLite reference output.
// It also measures the tensor arena the model really needs.
//
// Usage: tflm_check <vectors_dir>
#include <io.h>   // _dup/_dup2 (Windows PC build only; not needed on the board)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "fold0_int8.h"   // model bytes + scaling constants (from quantize_cnn.py)
#include "preprocess.h"   // firmware pre-processing

namespace {

constexpr size_t kArenaSize = 32 * 1024;   // generous for measuring; the real need is printed
alignas(16) uint8_t tensor_arena[kArenaSize];

// Exactly the 7 operations in the model: the firmware should register the same list.
using FallOpResolver = tflite::MicroMutableOpResolver<7>;
bool register_ops(FallOpResolver& r) {
  return r.AddConv2D() == kTfLiteOk && r.AddExpandDims() == kTfLiteOk &&
         r.AddFullyConnected() == kTfLiteOk && r.AddLogistic() == kTfLiteOk &&
         r.AddMaxPool2D() == kTfLiteOk && r.AddMean() == kTfLiteOk &&
         r.AddReshape() == kTfLiteOk;
}

// AllocateTensors with TFLM's error messages silenced (the arena search fails on purpose).
bool quiet_allocate(tflite::MicroInterpreter& interpreter) {
  fflush(stderr);
  int saved = _dup(_fileno(stderr));
  FILE* null_file = fopen("NUL", "w");
  _dup2(_fileno(null_file), _fileno(stderr));
  bool ok = interpreter.AllocateTensors() == kTfLiteOk;
  fflush(stderr);
  _dup2(saved, _fileno(stderr));
  _close(saved);
  fclose(null_file);
  return ok;
}

template <typename T>
std::vector<T> read_file(const std::string& path, size_t count) {
  std::vector<T> data(count);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f || fread(data.data(), sizeof(T), count, f) != count) {
    fprintf(stderr, "cannot read %zu values from %s\n", count, path.c_str());
    exit(1);
  }
  fclose(f);
  return data;
}

size_t read_count(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  size_t n = 0;
  if (!f || fscanf(f, "%zu", &n) != 1) {
    fprintf(stderr, "cannot read %s\n", path.c_str());
    exit(1);
  }
  fclose(f);
  return n;
}

struct Summary {
  size_t n = 0, a_match = 0, a_max_out_diff = 0, decision_diff_a = 0;
  size_t b_match = 0, b_input_bytes_diff = 0, b_max_out_diff = 0;
  size_t alarms_expected = 0, alarms_a = 0, alarms_b = 0, decision_diff_b = 0;
  double invoke_us = 0;
};

Summary run_set(tflite::MicroInterpreter& interpreter, const std::string& dir,
                const std::string& name) {
  const size_t n = read_count(dir + "/" + name + "_n.txt");
  const size_t raw_len = WINDOW_SAMPLES * RAW_CHANNELS, in_len = WINDOW_SAMPLES * MODEL_CHANNELS;
  auto raw = read_file<float>(dir + "/" + name + "_raw.f32", n * raw_len);
  auto scaled = read_file<float>(dir + "/" + name + "_scaled.f32", n * in_len);
  auto expected = read_file<int8_t>(dir + "/" + name + "_expected.i8", n);

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  Summary s;
  s.n = n;
  std::vector<int8_t> input_a(in_len), tflm_outputs(n);

  for (size_t i = 0; i < n; ++i) {
    // A: Python-scaled input, quantized in C
    for (size_t j = 0; j < in_len; ++j) input_a[j] = quantize_input(scaled[i * in_len + j]);
    memcpy(input->data.int8, input_a.data(), in_len);
    auto t0 = std::chrono::steady_clock::now();
    if (interpreter.Invoke() != kTfLiteOk) { fprintf(stderr, "Invoke failed\n"); exit(1); }
    s.invoke_us += std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
    int8_t out_a = output->data.int8[0];

    // B: raw IMU window through the firmware pre-processing
    const auto* window = reinterpret_cast<const float(*)[RAW_CHANNELS]>(&raw[i * raw_len]);
    preprocess_window(window, input->data.int8);
    for (size_t j = 0; j < in_len; ++j) s.b_input_bytes_diff += input->data.int8[j] != input_a[j];
    if (interpreter.Invoke() != kTfLiteOk) { fprintf(stderr, "Invoke failed\n"); exit(1); }
    int8_t out_b = output->data.int8[0];

    if (out_a != expected[i])
      printf("  [%s] window %zu: Python %d, TFLM %d (diff %d)\n", name.c_str(), i, expected[i],
             out_a, out_a - expected[i]);
    s.a_match += out_a == expected[i];
    s.a_max_out_diff = std::max(s.a_max_out_diff, (size_t)abs(out_a - expected[i]));
    s.decision_diff_a += is_fall(out_a) != is_fall(expected[i]);
    tflm_outputs[i] = out_b;
    s.b_match += out_b == expected[i];
    size_t diff = (size_t)abs(out_b - expected[i]);
    if (diff > s.b_max_out_diff) s.b_max_out_diff = diff;
    s.alarms_expected += is_fall(expected[i]);
    s.alarms_a += is_fall(out_a);
    s.alarms_b += is_fall(out_b);
    s.decision_diff_b += is_fall(out_b) != is_fall(expected[i]);
  }
  s.invoke_us /= n;

  // Golden outputs for the board: what TFLM itself produces from the RAW window (path B).
  // The board runs the same kernels, so it must reproduce these bytes exactly.
  FILE* f = fopen((dir + "/" + name + "_tflm_expected.i8").c_str(), "wb");
  fwrite(tflm_outputs.data(), 1, n, f);
  fclose(f);
  return s;
}

void print_summary(const char* name, const Summary& s) {
  printf("\n[%s] %zu windows\n", name, s.n);
  printf("  A  Python-scaled input -> TFLM : %zu / %zu outputs bit-exact"
         " (max output diff %zu step, decisions differing: %zu)\n", s.a_match, s.n,
         s.a_max_out_diff, s.decision_diff_a);
  printf("  B  raw window -> C preprocess -> TFLM : %zu / %zu outputs bit-exact"
         " (max output diff %zu step)\n", s.b_match, s.n, s.b_max_out_diff);
  printf("     input bytes differing from A: %zu of %zu\n", s.b_input_bytes_diff,
         s.n * WINDOW_SAMPLES * MODEL_CHANNELS);
  printf("  alarms at cut-off %.4f: Python %zu, TFLM-A %zu, TFLM-B %zu"
         " (decisions differing in B: %zu)\n", FALL_CUTOFF, s.alarms_expected, s.alarms_a,
         s.alarms_b, s.decision_diff_b);
  printf("  mean Invoke() time on this PC: %.1f us (not representative of the nRF52840)\n",
         s.invoke_us);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "vectors";
  setvbuf(stdout, nullptr, _IONBF, 0);
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(fall_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    fprintf(stderr, "schema version %lu != %d\n", (unsigned long)model->version(),
            TFLITE_SCHEMA_VERSION);
    return 1;
  }

  FallOpResolver resolver;
  if (!register_ops(resolver)) { fprintf(stderr, "op registration failed\n"); return 1; }

  tflite::RecordingMicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) { fprintf(stderr, "AllocateTensors failed\n"); return 1; }

  const size_t used = interpreter.arena_used_bytes();
  printf("=== TensorFlow Lite Micro PC check (%zu-bit build) ===\n", sizeof(void*) * 8);
  printf("model: %u bytes, %zu ops in subgraph\n", fall_model_len,
         (size_t)model->subgraphs()->Get(0)->operators()->size());
  printf("tensor arena used: %zu bytes (%.2f KB); minimum arena to allocate: %zu bytes\n",
         used, used / 1024.0, used + 16);

  // Input/output tensors must match what the header (and so the firmware) assumes.
  TfLiteTensor* in = interpreter.input(0);
  TfLiteTensor* out = interpreter.output(0);
  printf("input  %s [%d,%d,%d] scale %.8g zp %d (header %.8g / %d)\n",
         TfLiteTypeGetName(in->type), in->dims->data[0], in->dims->data[1], in->dims->data[2],
         in->params.scale, in->params.zero_point, INPUT_SCALE, INPUT_ZERO_POINT);
  printf("output %s scale %.8g zp %d (header %.8g / %d)\n", TfLiteTypeGetName(out->type),
         out->params.scale, out->params.zero_point, OUTPUT_SCALE, OUTPUT_ZERO_POINT);
  bool header_ok = in->type == kTfLiteInt8 && out->type == kTfLiteInt8 &&
                   in->params.scale == INPUT_SCALE && in->params.zero_point == INPUT_ZERO_POINT &&
                   out->params.scale == OUTPUT_SCALE &&
                   out->params.zero_point == OUTPUT_ZERO_POINT &&
                   in->dims->data[1] == WINDOW_SAMPLES && in->dims->data[2] == MODEL_CHANNELS;
  printf("header constants match the model: %s\n", header_ok ? "yes" : "NO");

  printf("\narena breakdown:\n");
  interpreter.GetMicroAllocator().PrintAllocations();

  Summary golden = run_set(interpreter, dir, "golden");
  Summary fold0 = run_set(interpreter, dir, "fold0");
  print_summary("golden", golden);
  print_summary("fold0 test", fold0);

  // The firmware uses a plain MicroInterpreter, which needs a little less than the recording
  // one. Find the smallest arena it accepts (one interpreter at a time, each in its own scope).
  size_t lo = 256, hi = used + 16;   // hi is known to be enough
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    bool ok;
    {
      alignas(16) static uint8_t trial_arena[kArenaSize];
      tflite::MicroInterpreter trial(model, resolver, trial_arena, mid);
      ok = quiet_allocate(trial);
    }
    if (ok) hi = mid; else lo = mid + 1;
  }
  printf("\nsmallest arena for a plain MicroInterpreter: %zu bytes (%.2f KB)\n", lo, lo / 1024.0);

  // Pass criteria. Python's TFLite (TF 2.21) requantizes with single rounding, TFLM with
  // double rounding, so a rare +-1 step difference in a FULLY_CONNECTED output is expected
  // (verified layer by layer with layer_dump + layer_diff.py). What must hold:
  //   - the model matches the header constants the firmware uses
  //   - the C pre-processing reproduces Python's input bytes exactly
  //   - no alarm decision changes, and >= 99.5% of outputs are bit-exact
  auto set_ok = [](const Summary& s) {
    return s.decision_diff_a == 0 && s.decision_diff_b == 0 && s.b_input_bytes_diff == 0 &&
           s.a_match >= 0.995 * s.n;
  };
  bool pass = header_ok && set_ok(golden) && set_ok(fold0);
  printf("\nTFLM golden outputs for the board written to %s/<set>_tflm_expected.i8\n", dir.c_str());
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
