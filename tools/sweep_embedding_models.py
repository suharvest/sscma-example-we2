#!/usr/bin/env python3
"""Run candidate embedding TFLite models on one int8 input and rank output match."""

import argparse
from pathlib import Path

import numpy as np


def load_interpreter(path: str):
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=path)
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path)


def cosine_int8(a: np.ndarray, b: np.ndarray, zp: int, scale: float) -> float:
    n = min(a.size, b.size)
    af = (a[:n].astype(np.float32) - zp) * scale
    bf = (b[:n].astype(np.float32) - zp) * scale
    an = float(np.linalg.norm(af))
    bn = float(np.linalg.norm(bf))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.dot(af / an, bf / bn))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bin", required=True)
    parser.add_argument("--device-output-bin", required=True)
    parser.add_argument("--root", default="sscma-example-we2/model_zoo/tflm_face_embedding")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-vela", action="store_true")
    parser.add_argument("--name-filter", default="mobilefacenet,ghostfacenet")
    args = parser.parse_args()

    input_data = np.frombuffer(Path(args.input_bin).read_bytes(), dtype=np.int8)
    device_output = np.frombuffer(Path(args.device_output_bin).read_bytes(), dtype=np.int8)
    results = []
    errors = 0

    for model_path in sorted(Path(args.root).rglob("*.tflite")):
        if not args.include_vela and "vela" in model_path.stem.lower():
            continue
        filters = [x.strip().lower() for x in args.name_filter.split(",") if x.strip()]
        if filters and not any(token in str(model_path).lower() for token in filters):
            continue
        try:
            interpreter = load_interpreter(str(model_path))
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            input_shape = tuple(int(x) for x in input_detail["shape"])
            if int(np.prod(input_shape)) != input_data.size:
                continue
            interpreter.set_tensor(input_detail["index"], input_data.reshape(input_shape))
            interpreter.invoke()
            output = interpreter.get_tensor(output_detail["index"]).reshape(-1).astype(np.int8)
            if output.size != device_output.size:
                continue
            q = output_detail.get("quantization_parameters", {})
            scales = np.asarray(q.get("scales", []), dtype=np.float32)
            zps = np.asarray(q.get("zero_points", []), dtype=np.int32)
            scale = float(scales.flat[0]) if scales.size else float(output_detail.get("quantization", (1.0, 0))[0])
            zp = int(zps.flat[0]) if zps.size else int(output_detail.get("quantization", (1.0, 0))[1])
            diff = output.astype(np.int16) - device_output.astype(np.int16)
            results.append((
                cosine_int8(output, device_output, zp, scale),
                float(np.mean(np.abs(diff))),
                int(np.max(np.abs(diff))),
                str(model_path),
                zp,
                scale,
            ))
        except Exception:
            errors += 1

    results.sort(reverse=True, key=lambda x: x[0])
    print(f"checked={len(results)} skipped_or_failed={errors}")
    for cos, mean_abs, max_abs, path, zp, scale in results[: args.limit]:
        print(f"{cos:.6f} mean_abs={mean_abs:.4f} max_abs={max_abs:3d} zp={zp:4d} scale={scale:.10f} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
