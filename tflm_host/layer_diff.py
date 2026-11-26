"""Compare TensorFlow Lite Micro (layer_dump output) with TensorFlow Lite, layer by layer.

Run from the project folder, after layer_dump:
  .venv\\Scripts\\python.exe tflm_host\\layer_diff.py <set> <window_index> <dump_dir>
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "models" / "cnn_none" / "fold0_int8.tflite"


def main(set_name, index, dump_dir):
    index, dump_dir = int(index), Path(dump_dir)
    scaled = np.fromfile(HERE / "vectors" / f"{set_name}_scaled.f32", dtype="<f4")
    window = scaled.reshape(-1, 100, 8)[index]

    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL), experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    scale, zero = inp["quantization"]
    interpreter.set_tensor(inp["index"], np.clip(np.round(window / scale + zero), -128, 127)
                           .astype(np.int8)[None])
    interpreter.invoke()

    print(f"{'op':>3} {'operator':<16} {'tensor':>6} {'shape':<14} {'differing':>10} {'max |diff|':>10}")
    for line in (dump_dir / "ops.txt").read_text().split("\n"):
        if not line:
            continue
        op, name, tensor, _ = line.split()
        tflite_value = interpreter.get_tensor(int(tensor))
        tflm_value = np.fromfile(dump_dir / f"t{tensor}.bin", dtype=tflite_value.dtype)
        tflm_value = tflm_value.reshape(tflite_value.shape)
        diff = np.abs(tflm_value.astype(np.int32) - tflite_value.astype(np.int32))
        print(f"{op:>3} {name:<16} {tensor:>6} {str(tflite_value.shape):<14} "
              f"{int((diff > 0).sum()):>10} {int(diff.max()):>10}")


if __name__ == "__main__":
    main(*sys.argv[1:])
