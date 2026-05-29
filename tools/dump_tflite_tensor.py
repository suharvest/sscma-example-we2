#!/usr/bin/env python3
"""Run a TFLite model and dump one tensor as raw bytes."""

import argparse
from pathlib import Path

import numpy as np

from compare_facedbg_tensor import fixed_input, load_tflite_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixed-test-seed", type=int, default=7)
    args = parser.parse_args()

    interpreter = load_tflite_model(args.model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    shape = tuple(int(x) for x in input_detail["shape"])
    q = input_detail.get("quantization_parameters", {})
    zp = int(np.asarray(q.get("zero_points", [0])).flat[0])
    tensor_type = 9 if input_detail["dtype"] == np.int8 else 3
    data = fixed_input(args.fixed_test_seed, int(np.prod(shape)), tensor_type).reshape(shape)
    if input_detail["dtype"] == np.uint8:
        data = (data.astype(np.int16) + 128).astype(np.uint8)
    elif zp != -1 and input_detail["dtype"] == np.int8:
        data = data.astype(np.int8)
    interpreter.set_tensor(input_detail["index"], data)
    interpreter.invoke()
    tensor = interpreter.get_tensor(args.tensor)
    Path(args.out).write_bytes(np.asarray(tensor).reshape(-1).tobytes())
    print(f"dumped tensor={args.tensor} shape={tensor.shape} bytes={tensor.size} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
