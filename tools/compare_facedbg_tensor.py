#!/usr/bin/env python3
"""Compare device MobileFaceNet tensors with local TFLite int8 inference."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def load_tflite_model(path: str, resolver: str = "auto"):
    try:
        from tflite_runtime.interpreter import Interpreter
        if resolver != "auto":
            raise RuntimeError("tflite_runtime does not expose TensorFlow OpResolverType")
        return Interpreter(model_path=path)
    except ImportError:
        import tensorflow as tf
        resolver_type = tf.lite.experimental.OpResolverType.AUTO
        if resolver == "builtin_ref":
            resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_REF
        elif resolver != "auto":
            raise ValueError(f"unknown resolver: {resolver}")
        return tf.lite.Interpreter(model_path=path, experimental_op_resolver_type=resolver_type)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        return x
    return x / norm


def decode_tensor(section: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(section["data_b64"]), dtype=np.int8)


def fixed_input(seed: int, size: int, tensor_type: int) -> np.ndarray:
    vals = (np.arange(size, dtype=np.uint32) * 73 + np.uint32(seed) * 29 + (np.arange(size, dtype=np.uint32) >> 3)) & 0xFF
    if tensor_type == 9:
        return (vals.astype(np.int16) - 128).astype(np.int8)
    return vals.astype(np.uint8).view(np.int8)


def dims_from(section: dict) -> tuple[int, ...]:
    dims = section.get("dims") or []
    return tuple(int(x) for x in dims if int(x) > 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="/tmp/himax_facedbg.json")
    parser.add_argument(
        "--model",
        default="model_zoo/tflm_face_embedding/foamliu_mobilefacenet_128d/foamliu_mobilefacenet_128d_qat_int8.tflite",
    )
    parser.add_argument("--dump-prefix", default="/tmp/himax")
    parser.add_argument("--fail-on-mismatch", action="store_true",
                        help="Exit non-zero unless device and local output are effectively identical")
    parser.add_argument("--cos-min", type=float, default=0.999,
                        help="Minimum normalized cosine when --fail-on-mismatch is set")
    parser.add_argument("--mean-diff-max", type=float, default=0.5,
                        help="Maximum mean raw int8 diff when --fail-on-mismatch is set")
    parser.add_argument("--resolver", choices=("auto", "builtin_ref"), default="auto",
                        help="TFLite op resolver for the local pre-Vela model")
    parser.add_argument("--prefer-dumped-input", action="store_true",
                        help="Use FACEDBG emb_input.data_b64 even when fixed_test_seed is available")
    parser.add_argument("--input-bin", default=None,
                        help="Use this raw int8 input buffer for local inference")
    args = parser.parse_args()

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    facedbg = payload["facedbg"]
    data = facedbg["data"]
    if not data.get("valid"):
        raise RuntimeError("FACEDBG payload is not valid; run AT+INVOKE with a visible face first")

    dev_in_info = data["emb_input"]
    dev_out_info = data["emb_output"]
    if args.input_bin:
        dev_input = np.frombuffer(Path(args.input_bin).read_bytes(), dtype=np.int8)
    elif payload.get("fixed_test_seed") is not None and not args.prefer_dumped_input:
        dev_input = fixed_input(int(payload["fixed_test_seed"]), int(dev_in_info["bytes"]), int(dev_in_info.get("type", 9)))
    elif dev_in_info.get("data_b64"):
        dev_input = decode_tensor(dev_in_info)
    else:
        raise RuntimeError("FACEDBG payload has no input data_b64 and no fixed_test_seed to reconstruct it")
    dev_output = decode_tensor(dev_out_info)

    Path(f"{args.dump_prefix}_emb_input_int8.bin").write_bytes(dev_input.tobytes())
    Path(f"{args.dump_prefix}_emb_output_int8.bin").write_bytes(dev_output.tobytes())

    interpreter = load_tflite_model(args.model, args.resolver)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    input_shape = tuple(int(x) for x in input_detail["shape"])
    device_shape = dims_from(dev_in_info)
    shape = input_shape if int(np.prod(input_shape)) == dev_input.size else device_shape
    if int(np.prod(shape)) != dev_input.size:
        raise RuntimeError(f"cannot reshape input bytes={dev_input.size} to model={input_shape} or device={device_shape}")

    interpreter.set_tensor(input_detail["index"], dev_input.reshape(shape))
    interpreter.invoke()
    pc_output = interpreter.get_tensor(output_detail["index"]).reshape(-1).astype(np.int8)
    Path(f"{args.dump_prefix}_local_output_int8.bin").write_bytes(pc_output.tobytes())

    n = min(dev_output.size, pc_output.size)
    raw_equal = bool(np.array_equal(dev_output[:n], pc_output[:n]))
    diff = pc_output[:n].astype(np.int16) - dev_output[:n].astype(np.int16)

    out_q = output_detail.get("quantization_parameters", {})
    model_scale = float(np.asarray(out_q.get("scales", [dev_out_info.get("scale_nano", 0) / 1e9])).flat[0])
    model_zp = int(np.asarray(out_q.get("zero_points", [dev_out_info.get("zp", 0)])).flat[0])
    dev_scale = float(dev_out_info.get("scale_nano", 0)) / 1e9
    dev_zp = int(dev_out_info.get("zp", model_zp))

    pc_deq = (pc_output[:n].astype(np.float32) - model_zp) * model_scale
    dev_deq = (dev_output[:n].astype(np.float32) - dev_zp) * dev_scale
    pc_emb = l2_normalize(pc_deq)
    dev_emb = l2_normalize(dev_deq)
    cos = float(np.dot(pc_emb, dev_emb))

    invoke = payload.get("invoke") or {}
    faces = ((invoke.get("data") or {}).get("faces") or [])
    json_emb_cos = None
    if faces and "embedding" in faces[0]:
        json_emb = l2_normalize(np.asarray(faces[0]["embedding"], dtype=np.float32))
        json_emb_cos = float(np.dot(dev_emb[: json_emb.size], json_emb))

    print("Device tensor:")
    print(f"  input bytes={dev_input.size} dims={device_shape} type={dev_in_info.get('type')} zp={dev_in_info.get('zp')} scale={dev_in_info.get('scale_nano')}/1e9")
    print(f"  output bytes={dev_output.size} dims={dims_from(dev_out_info)} type={dev_out_info.get('type')} zp={dev_zp} scale={dev_scale:.10f}")
    print("Local TFLite:")
    print(f"  model={args.model}")
    print(f"  resolver={args.resolver}")
    print(f"  input_shape={input_shape} output_bytes={pc_output.size} zp={model_zp} scale={model_scale:.10f}")
    print("Comparison:")
    print(f"  raw_equal={raw_equal}")
    print(f"  compared={n} bytes")
    print(f"  diff_abs_max={int(np.max(np.abs(diff))) if n else 0}")
    print(f"  diff_abs_mean={float(np.mean(np.abs(diff))) if n else 0:.4f}")
    print(f"  diff_nonzero={int(np.count_nonzero(diff))}/{n}")
    print(f"  normalized_cosine={cos:.6f}")
    if json_emb_cos is not None:
        print(f"  device_json_vs_dump_cosine={json_emb_cos:.6f}")
    print(f"  dumped_prefix={args.dump_prefix}_emb_*.bin")
    print(f"  dumped_local_output={args.dump_prefix}_local_output_int8.bin")
    mean_abs = float(np.mean(np.abs(diff))) if n else 0.0
    status_ok = raw_equal or (cos >= args.cos_min and mean_abs <= args.mean_diff_max)
    print(f"  status={'PASS' if status_ok else 'FAIL'}")
    if args.fail_on_mismatch and not status_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
