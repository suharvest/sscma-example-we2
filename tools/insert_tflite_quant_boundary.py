#!/usr/bin/env python3
"""Insert an explicit TFLite QUANTIZE boundary after an operator output."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def quantize_opcode_index(model: dict) -> int:
    for i, opcode in enumerate(model["operator_codes"]):
        if opcode.get("builtin_code") == "QUANTIZE":
            return i
    model["operator_codes"].append(
        {
            "deprecated_builtin_code": 114,
            "version": 2,
            "builtin_code": "QUANTIZE",
        }
    )
    return len(model["operator_codes"]) - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--after-op", type=int, required=True)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--zero-point", type=int)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    model = json.loads(Path(args.json).read_text(encoding="utf-8"))
    sub = model["subgraphs"][0]
    tensors = sub["tensors"]
    operators = sub["operators"]
    buffers = model["buffers"]

    src_tensor_index = operators[args.after_op]["outputs"][0]
    src_tensor = tensors[src_tensor_index]
    new_tensor = copy.deepcopy(src_tensor)
    new_tensor["name"] = f"{src_tensor.get('name', f't{src_tensor_index}')}_explicit_quant"
    new_tensor["buffer"] = len(buffers)
    q = copy.deepcopy(new_tensor.get("quantization") or {})
    if args.scale is not None:
        q["scale"] = [args.scale]
    if args.zero_point is not None:
        q["zero_point"] = [args.zero_point]
    new_tensor["quantization"] = q
    buffers.append({})
    new_tensor_index = len(tensors)
    tensors.append(new_tensor)

    q_opcode = quantize_opcode_index(model)
    quant_op = {
        "opcode_index": q_opcode,
        "inputs": [src_tensor_index],
        "outputs": [new_tensor_index],
        "builtin_options_type": "NONE",
        "custom_options_format": "FLEXBUFFERS",
        "large_custom_options_offset": 0,
        "large_custom_options_size": 0,
        "builtin_options_2_type": "NONE",
        "debug_metadata_index": -1,
    }
    operators.insert(args.after_op + 1, quant_op)

    for op_index, op in enumerate(operators):
        if op_index <= args.after_op + 1:
            continue
        op["inputs"] = [new_tensor_index if x == src_tensor_index else x for x in op.get("inputs", [])]
    sub["outputs"] = [new_tensor_index if x == src_tensor_index else x for x in sub.get("outputs", [])]

    model.pop("signature_defs", None)
    Path(args.out_json).write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(
        f"inserted QUANTIZE after op {args.after_op}: "
        f"t{src_tensor_index} -> t{new_tensor_index}, scale={q.get('scale')} zp={q.get('zero_point')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
