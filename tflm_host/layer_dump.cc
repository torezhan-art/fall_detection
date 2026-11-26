// Debug tool: run ONE window through TensorFlow Lite Micro and save every operator's output,
// so layer_diff.py can compare them layer by layer with TensorFlow Lite (Python).
//
// Usage: layer_dump <vectors_dir> <set> <window_index> <out_dir>
//   writes <out_dir>/t<tensor_index>.bin for each operator output, and <out_dir>/ops.txt
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "fold0_int8.h"
#include "preprocess.h"

namespace {
alignas(16) uint8_t tensor_arena[64 * 1024];   // preserve_all_tensors needs more room
}

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: layer_dump <vectors_dir> <set> <window_index> <out_dir>\n");
    return 1;
  }
  const std::string dir = argv[1], set = argv[2], out_dir = argv[4];
  const long index = atol(argv[3]);
  tflite::InitializeTarget();

  // the window's Python-scaled input, quantized exactly as in main.cc (check A)
  const size_t in_len = WINDOW_SAMPLES * MODEL_CHANNELS;
  std::vector<float> scaled(in_len);
  FILE* f = fopen((dir + "/" + set + "_scaled.f32").c_str(), "rb");
  if (!f || fseek(f, (long)(index * in_len * sizeof(float)), SEEK_SET) != 0 ||
      fread(scaled.data(), sizeof(float), in_len, f) != in_len) {
    fprintf(stderr, "cannot read window %ld\n", index);
    return 1;
  }
  fclose(f);

  const tflite::Model* model = tflite::GetModel(fall_model);
  tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D(); resolver.AddExpandDims(); resolver.AddFullyConnected();
  resolver.AddLogistic(); resolver.AddMaxPool2D(); resolver.AddMean(); resolver.AddReshape();

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena),
                                       nullptr, nullptr, /*preserve_all_tensors=*/true);
  if (interpreter.AllocateTensors() != kTfLiteOk) return 1;
  for (size_t j = 0; j < in_len; ++j) interpreter.input(0)->data.int8[j] = quantize_input(scaled[j]);
  if (interpreter.Invoke() != kTfLiteOk) return 1;

  const auto* subgraph = model->subgraphs()->Get(0);
  const auto* codes = model->operator_codes();
  FILE* ops = fopen((out_dir + "/ops.txt").c_str(), "w");
  for (size_t op = 0; op < subgraph->operators()->size(); ++op) {
    const auto* op_def = subgraph->operators()->Get(op);
    auto code = tflite::micro::GetBuiltinCode(codes->Get(op_def->opcode_index()));
    for (int32_t t : *op_def->outputs()) {
      TfLiteEvalTensor* tensor = interpreter.GetTensor(t);
      size_t bytes = 0;
      tflite::TfLiteEvalTensorByteLength(tensor, &bytes);
      FILE* out = fopen((out_dir + "/t" + std::to_string(t) + ".bin").c_str(), "wb");
      fwrite(tensor->data.raw, 1, bytes, out);
      fclose(out);
      fprintf(ops, "%zu %s %d %zu\n", op, tflite::EnumNameBuiltinOperator(code), t, bytes);
    }
  }
  fclose(ops);
  printf("dumped %zu operators of %s window %ld to %s\n", (size_t)subgraph->operators()->size(),
         set.c_str(), index, out_dir.c_str());
  return 0;
}
