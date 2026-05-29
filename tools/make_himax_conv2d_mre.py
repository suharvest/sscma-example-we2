#!/usr/bin/env python3
"""Build a tiny Conv2D TFLite JSON that reproduces the WE2 first-layer mismatch.

The script consumes a flatc JSON dump of the v2_lfw6 MobileFaceNet model and
emits a standalone one-operator model plus the deterministic input used by the
firmware AT+FACEEMBTEST path.
"""

import argparse
import json
import struct
from pathlib import Path


def fixed_input(seed: int, size: int) -> bytes:
    data = bytearray(size)
    for i in range(size):
        v = (i * 73 + seed * 29 + (i >> 3)) & 0xFF
        data[i] = (v - 128) & 0xFF
    return bytes(data)


def tensor_quant(tensor: dict) -> dict:
    q = tensor.get("quantization") or {}
    return {
        "scale": q.get("scale") or [1.0],
        "zero_point": q.get("zero_point") or [0],
        "details_type": "NONE",
        "quantized_dimension": q.get("quantized_dimension", 0),
    }


def int32_first(values: list[int], index: int) -> list[int]:
    start = index * 4
    return values[start : start + 4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True, help="flatc JSON dump of mfn_w1_pairft_128d.int8.tflite")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--variant",
        choices=("channel0", "single_tap"),
        default="channel0",
        help="channel0 keeps the first real 3x3x3 output-channel weights; single_tap keeps only the center tap.",
    )
    parser.add_argument(
        "--with-projection",
        action="store_true",
        help="Append a 1x1 identity Conv2D projection, matching the failing prefix probes.",
    )
    args = parser.parse_args()

    source = json.loads(Path(args.source_json).read_text(encoding="utf-8"))
    src_sub = source["subgraphs"][0]
    src_tensors = src_sub["tensors"]
    src_op0 = src_sub["operators"][0]
    if src_op0.get("builtin_options_type") != "Conv2DOptions":
        raise ValueError("source op0 is not Conv2D")

    input_tensor = src_tensors[src_op0["inputs"][0]]
    weight_tensor = src_tensors[src_op0["inputs"][1]]
    bias_tensor = src_tensors[src_op0["inputs"][2]]
    output_tensor = src_tensors[src_op0["outputs"][0]]
    src_buffers = source["buffers"]

    input_shape = [1, 112, 112, 3]
    output_shape = [1, 56, 56, 1]
    input_size = 1
    for dim in input_shape:
        input_size *= dim

    original_weights = list(src_buffers[weight_tensor["buffer"]].get("data", []))
    if len(original_weights) < 27:
        raise ValueError("source first Conv2D weight buffer is too small")
    weights = [0] * 27
    if args.variant == "channel0":
        weights[:] = original_weights[:27]
    else:
        # OHWI layout: output channel 0, kernel center (1, 1), input channel 0.
        weights[(1 * 3 * 3) + (1 * 3) + 0] = 64

    original_bias = list(src_buffers[bias_tensor["buffer"]].get("data", []))
    bias = int32_first(original_bias, 0)
    if args.variant == "single_tap":
        bias = list(struct.pack("<i", 0))

    weight_q = tensor_quant(weight_tensor)
    bias_q = tensor_quant(bias_tensor)
    weight_q["scale"] = [weight_q["scale"][0]]
    weight_q["zero_point"] = [0]
    weight_q["quantized_dimension"] = 0
    bias_q["scale"] = [bias_q["scale"][0]]
    bias_q["zero_point"] = [0]
    bias_q["quantized_dimension"] = 0

    buffers = [{}, {}, {"data": weights}, {"data": bias}, {}]
    tensors = [
        {
            "shape": input_shape,
            "type": "INT8",
            "buffer": 1,
            "name": "mre_input",
            "quantization": tensor_quant(input_tensor),
            "is_variable": False,
            "has_rank": True,
        },
        {
            "shape": [1, 3, 3, 3],
            "type": "INT8",
            "buffer": 2,
            "name": "mre_conv_weights",
            "quantization": weight_q,
            "is_variable": False,
            "has_rank": True,
        },
        {
            "shape": [1],
            "type": "INT32",
            "buffer": 3,
            "name": "mre_conv_bias",
            "quantization": bias_q,
            "is_variable": False,
            "has_rank": True,
        },
        {
            "shape": output_shape,
            "type": "INT8",
            "buffer": 4,
            "name": "mre_conv_output",
            "quantization": {
                "scale": [tensor_quant(output_tensor)["scale"][0]],
                "zero_point": [tensor_quant(output_tensor)["zero_point"][0]],
                "details_type": "NONE",
                "quantized_dimension": 0,
            },
            "is_variable": False,
            "has_rank": True,
        },
    ]
    operators = [
        {
            "opcode_index": 0,
            "inputs": [0, 1, 2],
            "outputs": [3],
            "builtin_options_type": "Conv2DOptions",
            "builtin_options": {
                "padding": "SAME",
                "stride_w": 2,
                "stride_h": 2,
                "fused_activation_function": "RELU6",
                "dilation_w_factor": 1,
                "dilation_h_factor": 1,
            },
            "custom_options_format": "FLEXBUFFERS",
            "large_custom_options_offset": 0,
            "large_custom_options_size": 0,
            "builtin_options_2_type": "NONE",
        }
    ]
    outputs = [3]
    if args.with_projection:
        proj_weight_buffer = len(buffers)
        buffers.append({"data": [1]})
        proj_bias_buffer = len(buffers)
        buffers.append({"data": list(struct.pack("<i", 0))})
        proj_output_buffer = len(buffers)
        buffers.append({})
        conv_output_quant = tensors[3]["quantization"]
        proj_weight_index = len(tensors)
        tensors.append(
            {
                "shape": [1, 1, 1, 1],
                "type": "INT8",
                "buffer": proj_weight_buffer,
                "name": "mre_project_weights",
                "quantization": {
                    "scale": [1.0],
                    "zero_point": [0],
                    "details_type": "NONE",
                    "quantized_dimension": 0,
                },
                "is_variable": False,
                "has_rank": True,
            }
        )
        proj_bias_index = len(tensors)
        tensors.append(
            {
                "shape": [1],
                "type": "INT32",
                "buffer": proj_bias_buffer,
                "name": "mre_project_bias",
                "quantization": {
                    "scale": [conv_output_quant["scale"][0]],
                    "zero_point": [0],
                    "details_type": "NONE",
                    "quantized_dimension": 0,
                },
                "is_variable": False,
                "has_rank": True,
            }
        )
        proj_output_index = len(tensors)
        tensors.append(
            {
                "shape": output_shape,
                "type": "INT8",
                "buffer": proj_output_buffer,
                "name": "mre_project_output",
                "quantization": conv_output_quant,
                "is_variable": False,
                "has_rank": True,
            }
        )
        operators.append(
            {
                "opcode_index": 0,
                "inputs": [3, proj_weight_index, proj_bias_index],
                "outputs": [proj_output_index],
                "builtin_options_type": "Conv2DOptions",
                "builtin_options": {
                    "padding": "VALID",
                    "stride_w": 1,
                    "stride_h": 1,
                    "fused_activation_function": "NONE",
                    "dilation_w_factor": 1,
                    "dilation_h_factor": 1,
                },
                "custom_options_format": "FLEXBUFFERS",
                "large_custom_options_offset": 0,
                "large_custom_options_size": 0,
                "builtin_options_2_type": "NONE",
            }
        )
        outputs = [proj_output_index]

    model = {
        "version": 3,
        "operator_codes": [
            {
                "deprecated_builtin_code": 3,
                "version": 3,
                "builtin_code": "CONV_2D",
            }
        ],
        "subgraphs": [
            {
                "tensors": tensors,
                "inputs": [0],
                "outputs": outputs,
                "operators": operators,
                "name": "main",
            }
        ],
        "description": f"WE2 Conv2D MRE from v2_lfw6 op0, variant={args.variant}, projection={args.with_projection}",
        "buffers": buffers,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.variant}_proj" if args.with_projection else args.variant
    json_path = out_dir / f"we2_conv2d_mre_{suffix}.json"
    input_path = out_dir / f"input_seed{args.seed}_uint8_pattern.bin"
    manifest_path = out_dir / f"we2_conv2d_mre_{suffix}.manifest.json"
    json_path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    input_path.write_bytes(fixed_input(args.seed, input_size))
    manifest_path.write_text(
        json.dumps(
            {
                "source_json": str(Path(args.source_json).resolve()),
                "variant": args.variant,
                "with_projection": args.with_projection,
                "seed": args.seed,
                "model_json": str(json_path),
                "input_bin": str(input_path),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "op": "CONV_2D SAME stride=2 dilation=1 fused_activation=RELU6",
                "note": "Input bytes are the firmware FACEEMBTEST uint8 pattern; interpreted as int8 tensor bytes.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"model_json={json_path}")
    print(f"input_bin={input_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
