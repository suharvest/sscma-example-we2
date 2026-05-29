#!/usr/bin/env python3
"""Create a TFLite op slice and append a small 1x1 int8 projection output."""

import argparse
import json
import struct
from pathlib import Path


def quant(tensor: dict) -> tuple[int, float]:
    q = tensor.get("quantization") or {}
    zp = int((q.get("zero_point") or [0])[0])
    scale = float((q.get("scale") or [1.0])[0])
    return zp, scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--start-op", type=int, required=True)
    parser.add_argument("--end-op", type=int, required=True)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    model = json.loads(Path(args.json).read_text(encoding="utf-8"))
    sub = model["subgraphs"][0]
    tensors = sub["tensors"]
    operators = sub["operators"]

    if args.start_op < 0 or args.end_op < args.start_op or args.end_op >= len(operators):
        raise ValueError(f"invalid op range {args.start_op}..{args.end_op}")

    input_tensor_index = operators[args.start_op]["inputs"][0]
    src_tensor_index = operators[args.end_op]["outputs"][0]
    sub["inputs"] = [input_tensor_index]
    operators[:] = operators[args.start_op : args.end_op + 1]

    src_tensor = tensors[src_tensor_index]
    src_shape = list(src_tensor["shape"])
    if len(src_shape) != 4:
        raise ValueError(f"source tensor must be NHWC rank-4, got {src_shape}")
    src_zp, src_scale = quant(src_tensor)
    in_channels = int(src_shape[3])
    out_channels = min(args.channels, in_channels)

    buffers = model["buffers"]
    weight_buf_index = len(buffers)
    weight_data = bytearray(out_channels * in_channels)
    for ch in range(out_channels):
        weight_data[ch * in_channels + ch] = 1
    buffers.append({"data": list(weight_data)})

    bias_buf_index = len(buffers)
    buffers.append({"data": list(struct.pack("<" + "i" * out_channels, *([0] * out_channels)))})

    output_buf_index = len(buffers)
    buffers.append({})

    weight_tensor_index = len(tensors)
    tensors.append(
        {
            "shape": [out_channels, 1, 1, in_channels],
            "type": "INT8",
            "buffer": weight_buf_index,
            "name": f"debug_project_t{src_tensor_index}_weights",
            "quantization": {
                "scale": [1.0] * out_channels,
                "zero_point": [0] * out_channels,
                "details_type": "NONE",
                "quantized_dimension": 0,
            },
            "is_variable": False,
            "has_rank": True,
        }
    )

    bias_tensor_index = len(tensors)
    tensors.append(
        {
            "shape": [out_channels],
            "type": "INT32",
            "buffer": bias_buf_index,
            "name": f"debug_project_t{src_tensor_index}_bias",
            "quantization": {
                "scale": [src_scale] * out_channels,
                "zero_point": [0] * out_channels,
                "details_type": "NONE",
                "quantized_dimension": 0,
            },
            "is_variable": False,
            "has_rank": True,
        }
    )

    output_tensor_index = len(tensors)
    out_shape = [int(src_shape[0]), int(src_shape[1]), int(src_shape[2]), out_channels]
    tensors.append(
        {
            "shape": out_shape,
            "type": "INT8",
            "buffer": output_buf_index,
            "name": f"debug_project_t{src_tensor_index}",
            "quantization": {
                "scale": [src_scale],
                "zero_point": [src_zp],
                "details_type": "NONE",
                "quantized_dimension": 0,
            },
            "is_variable": False,
            "has_rank": True,
        }
    )

    operators.append(
        {
            "opcode_index": 0,
            "inputs": [src_tensor_index, weight_tensor_index, bias_tensor_index],
            "outputs": [output_tensor_index],
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
    sub["outputs"] = [output_tensor_index]
    model.pop("signature_defs", None)

    Path(args.out_json).write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(
        f"input_tensor={input_tensor_index} input_shape={tensors[input_tensor_index]['shape']} "
        f"source_tensor={src_tensor_index} output_shape={out_shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
