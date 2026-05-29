#!/usr/bin/env python3
"""Create a prefix TFLite model and append a small 1x1 int8 projection output."""

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
    parser.add_argument("--op", type=int, required=True, help="Last original op index to keep, or -1 to project model input")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--channel-offset", type=int, default=0)
    parser.add_argument(
        "--op0-variant",
        choices=("none", "no_relu6", "valid_padding", "stride1", "per_tensor_weight"),
        default="none",
    )
    parser.add_argument(
        "--op0-weight-variant",
        choices=(
            "none",
            "zero",
            "single_tap",
            "channel0_only",
            "channel0_input0_only",
            "channel0_input1_only",
            "channel0_input2_only",
            "channel0_center_only",
        ),
        default="none",
    )
    parser.add_argument("--input-zero-point", type=int)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    model = json.loads(Path(args.json).read_text(encoding="utf-8"))
    sub = model["subgraphs"][0]
    tensors = sub["tensors"]
    operators = sub["operators"]
    if args.input_zero_point is not None:
        input_tensor = tensors[sub["inputs"][0]]
        input_tensor.setdefault("quantization", {})["zero_point"] = [args.input_zero_point]
    if args.op0_variant != "none":
        op0 = operators[0]
        op0_opts = op0["builtin_options"]
        out_tensor = tensors[op0["outputs"][0]]
        if args.op0_variant == "no_relu6":
            op0_opts["fused_activation_function"] = "NONE"
        elif args.op0_variant == "valid_padding":
            op0_opts["padding"] = "VALID"
            out_tensor["shape"] = [1, 55, 55, 64]
            if "shape_signature" in out_tensor:
                out_tensor["shape_signature"] = [-1, 55, 55, 64]
        elif args.op0_variant == "stride1":
            op0_opts["stride_w"] = 1
            op0_opts["stride_h"] = 1
            out_tensor["shape"] = [1, 112, 112, 64]
            if "shape_signature" in out_tensor:
                out_tensor["shape_signature"] = [-1, 112, 112, 64]
        elif args.op0_variant == "per_tensor_weight":
            weight = tensors[op0["inputs"][1]]
            bias = tensors[op0["inputs"][2]]
            wq = weight.get("quantization", {})
            bq = bias.get("quantization", {})
            w_scales = wq.get("scale") or [1.0]
            b_scales = bq.get("scale") or [1.0]
            wq["scale"] = [float(sum(w_scales) / len(w_scales))]
            wq["zero_point"] = [0]
            wq["quantized_dimension"] = 0
            bq["scale"] = [float(sum(b_scales) / len(b_scales))]
            bq["zero_point"] = [0]
            bq["quantized_dimension"] = 0
    if args.op0_weight_variant != "none":
        op0 = operators[0]
        weight = tensors[op0["inputs"][1]]
        bias = tensors[op0["inputs"][2]]
        weight_buf = model["buffers"][weight["buffer"]]
        bias_buf = model["buffers"][bias["buffer"]]
        original_weight_data = list(weight_buf.get("data", []))
        weight_data = [0] * int(weight_buf.get("size", len(original_weight_data)))
        if not weight_data:
            weight_data = [0] * len(original_weight_data)
        if args.op0_weight_variant == "channel0_only":
            weight_data[:27] = original_weight_data[:27]
        elif args.op0_weight_variant.startswith("channel0_input"):
            input_ch = int(args.op0_weight_variant.removeprefix("channel0_input").removesuffix("_only"))
            for kh in range(3):
                for kw in range(3):
                    idx = (0 * 3 * 3 * 3) + (kh * 3 * 3) + (kw * 3) + input_ch
                    weight_data[idx] = original_weight_data[idx]
        elif args.op0_weight_variant == "channel0_center_only":
            for input_ch in range(3):
                idx = (0 * 3 * 3 * 3) + (1 * 3 * 3) + (1 * 3) + input_ch
                weight_data[idx] = original_weight_data[idx]
        elif args.op0_weight_variant == "single_tap":
            # OHWI layout: output channel 0, kernel center (1, 1), input channel 0.
            weight_data[(0 * 3 * 3 * 3) + (1 * 3 * 3) + (1 * 3) + 0] = 64
        weight_buf["data"] = weight_data
        bias_len = len(bias_buf.get("data", []))
        if args.op0_weight_variant.startswith("channel0_"):
            bias_data = [0] * bias_len
            bias_data[:4] = list(bias_buf.get("data", []))[:4]
            bias_buf["data"] = bias_data
        else:
            bias_buf["data"] = [0] * bias_len
    if args.op >= 0:
        src_tensor_index = operators[args.op]["outputs"][0]
        operators[:] = operators[: args.op + 1]
    else:
        src_tensor_index = sub["inputs"][0]
        operators[:] = []
    src_tensor = tensors[src_tensor_index]
    src_shape = list(src_tensor["shape"])
    if len(src_shape) != 4:
        raise ValueError(f"source tensor must be NHWC rank-4, got {src_shape}")
    src_zp, src_scale = quant(src_tensor)
    in_channels = int(src_shape[3])
    if args.channel_offset < 0 or args.channel_offset >= in_channels:
        raise ValueError(f"channel offset must be in [0, {in_channels}), got {args.channel_offset}")
    out_channels = min(args.channels, in_channels - args.channel_offset)

    buffers = model["buffers"]
    weight_buf_index = len(buffers)
    weight_data = bytearray(out_channels * in_channels)
    for ch in range(out_channels):
        weight_data[ch * in_channels + args.channel_offset + ch] = 1
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
    print(f"source_tensor={src_tensor_index} source_shape={src_shape} output_shape={out_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
