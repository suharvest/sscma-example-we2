#!/usr/bin/env python3
"""Create a clean linear TFLite slice from consecutive operators."""

import argparse
import copy
import json
import struct
from pathlib import Path


def quant(tensor: dict) -> tuple[int, float]:
    q = tensor.get("quantization") or {}
    return int((q.get("zero_point") or [0])[0]), float((q.get("scale") or [1.0])[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--start-op", type=int, required=True)
    parser.add_argument("--end-op", type=int, required=True)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    src = json.loads(Path(args.json).read_text(encoding="utf-8"))
    src_sub = src["subgraphs"][0]
    src_ops = src_sub["operators"]
    src_tensors = src_sub["tensors"]

    if args.start_op < 0 or args.end_op < args.start_op or args.end_op >= len(src_ops):
        raise ValueError(f"invalid op range {args.start_op}..{args.end_op}")

    slice_ops = src_ops[args.start_op : args.end_op + 1]
    input_src = slice_ops[0]["inputs"][0]
    output_src = slice_ops[-1]["outputs"][0]

    tensor_src_order: list[int] = [input_src]
    for op in slice_ops:
        for idx in op.get("inputs", []) + op.get("outputs", []):
            if idx >= 0 and idx not in tensor_src_order:
                tensor_src_order.append(idx)

    buffer_map = {0: 0}
    buffers = [{}]

    def map_buffer(old: int) -> int:
        if old in buffer_map:
            return buffer_map[old]
        buffer_map[old] = len(buffers)
        buffers.append(copy.deepcopy(src["buffers"][old]))
        return buffer_map[old]

    tensor_map = {}
    tensors = []
    for old in tensor_src_order:
        tensor_map[old] = len(tensors)
        t = copy.deepcopy(src_tensors[old])
        t["buffer"] = map_buffer(int(t.get("buffer", 0)))
        tensors.append(t)

    opcode_map = {}
    operator_codes = []

    def map_opcode(old: int) -> int:
        if old in opcode_map:
            return opcode_map[old]
        opcode_map[old] = len(operator_codes)
        operator_codes.append(copy.deepcopy(src["operator_codes"][old]))
        return opcode_map[old]

    operators = []
    for op in slice_ops:
        new_op = copy.deepcopy(op)
        new_op["opcode_index"] = map_opcode(int(op["opcode_index"]))
        new_op["inputs"] = [tensor_map[i] if i >= 0 else i for i in op.get("inputs", [])]
        new_op["outputs"] = [tensor_map[i] if i >= 0 else i for i in op.get("outputs", [])]
        operators.append(new_op)

    src_tensor = src_tensors[output_src]
    src_shape = list(src_tensor["shape"])
    if len(src_shape) != 4:
        raise ValueError(f"source tensor must be NHWC rank-4, got {src_shape}")
    src_zp, src_scale = quant(src_tensor)
    in_channels = int(src_shape[3])
    out_channels = min(args.channels, in_channels)

    conv_opcode_index = None
    for i, code in enumerate(operator_codes):
        if code.get("builtin_code") == "CONV_2D":
            conv_opcode_index = i
            break
    if conv_opcode_index is None:
        conv_opcode_index = len(operator_codes)
        operator_codes.append({"deprecated_builtin_code": 3, "version": 6, "builtin_code": "CONV_2D"})

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
            "name": f"debug_project_t{output_src}_weights",
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
            "name": f"debug_project_t{output_src}_bias",
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
            "name": f"debug_project_t{output_src}",
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
            "opcode_index": conv_opcode_index,
            "inputs": [tensor_map[output_src], weight_tensor_index, bias_tensor_index],
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

    model = {
        "version": src.get("version", 3),
        "operator_codes": operator_codes,
        "subgraphs": [
            {
                "tensors": tensors,
                "inputs": [tensor_map[input_src]],
                "outputs": [output_tensor_index],
                "operators": operators,
                "name": f"clean_slice_{args.start_op}_{args.end_op}",
            }
        ],
        "description": f"clean slice ops {args.start_op}..{args.end_op}",
        "buffers": buffers,
        "metadata": [],
    }
    Path(args.out_json).write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(
        f"input_src={input_src} input_shape={src_tensors[input_src]['shape']} "
        f"output_src={output_src} output_shape={out_shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
