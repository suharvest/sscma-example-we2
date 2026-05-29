#!/usr/bin/env python3
"""
Fix BOTH weight AND activation quantization in ReLU singlepath TFLite using .espdl exponents.

Strategy:
1. Start from ReLU singlepath float32 TFLite (64 ops, right architecture, ~178 KiB SRAM)
2. Quantize weights using .espdl exponents (per-tensor or per-channel for merged layers)
3. Set activation quantization from .espdl exponents
4. Build INT8 flatbuffer directly with flatbuffer Object API

Merged layers (split in espdl, merged in singlepath):
- dconv_45 sep:  ESPDL split into sep_1 (256ch, exp=-9) + sep_2 (256ch, exp=-10)
  TFLite merged: [512, 1, 1, 128] -> per-channel: [2^-9]*256 + [2^-10]*256
- dconv_45 proj: ESPDL split into proj_1 (64ch, exp=-8) + proj_2 (64ch, exp=-8)
  TFLite merged: [128, 1, 1, 512] -> per-channel: [2^-8]*128 (both same)
- conv_6 sep:    ESPDL split into sep_1 (256ch, exp=-9) + sep_2 (256ch, exp=-9)
  TFLite merged: [512, 1, 1, 128] -> per-channel: [2^-9]*256 + [2^-9]*256
"""

import struct, sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import flatbuffers
from tensorflow.lite.python import schema_py_generated as tflite_schema

sys.path.insert(0, "/tmp/esp-ppq-src")
from esp_ppq.parser.espdl.FlatBuffers.Dl import Model
from esp_ppq.parser.espdl.FlatBuffers.Dl.TypeInfoValue import TypeInfoValue

# === CONFIG ===
ESPDL_PATH = "/Users/harvest/project/esp-dl/models/human_face_recognition/models/s3/human_face_feat_mfn_s8_v1.espdl"
BASE = Path("/Users/harvest/project/grove_vision_2/sscma-example-we2/model_zoo/tflm_face_embedding")
SCRIPT_DIR = BASE / "official_mobilefacenet"
FLOAT32_TFLITE = SCRIPT_DIR / "mfn_sp_saved_model/mfn_s8_v1_singlepath_float32.tflite"
OUT_TFLITE = SCRIPT_DIR / "mfn_prelu_fixed.tflite"

# TFLite op code constants
CONV_2D = 3
DEPTHWISE_CONV_2D = 4
ADD = 0
MUL = 18
PAD = 34
CUSTOM = 32
RELU = 19


# ============================================================
# STEP 1: Parse .espdl
# ============================================================

def parse_espdl(espdl_path):
    """Extract all quantization info from .espdl."""
    with open(espdl_path, "rb") as f:
        data = f.read()
    model_data = data[16:16+struct.unpack("I", data[8:12])[0]]
    model = Model.Model.GetRootAs(model_data, 0)
    graph = model.Graph()

    # --- Nodes ---
    nodes = []
    for i in range(graph.NodeLength()):
        node = graph.Node(i)
        name = node.Name().decode("utf-8", errors="ignore")
        op = node.OpType().decode("utf-8", errors="ignore")
        inputs = [node.Input(j).decode("utf-8", errors="ignore") for j in range(node.InputLength())]
        outputs = [node.Output(j).decode("utf-8", errors="ignore") for j in range(node.OutputLength())]
        nodes.append({"idx": i, "name": name, "op": op, "inputs": inputs, "outputs": outputs})

    # --- Weight initializers with exponents ---
    # ESPDL weight format: HWIO [H, W, Cin, Cout] for standard
    #                       [H, W, Cin, d_mult] for depthwise
    weight_exps = {}
    for i in range(graph.InitializerLength()):
        init = graph.Initializer(i)
        name = init.Name().decode("utf-8", errors="ignore")
        dims = tuple([init.Dims(j) for j in range(init.DimsLength())])
        exps = tuple([init.Exponents(j) for j in range(init.ExponentsLength())])
        if exps:
            weight_exps[name] = {"dims": dims, "exps": exps}

    # --- Value infos (activation exponents) ---
    value_infos = {}
    for i in range(graph.ValueInfoLength()):
        vi = graph.ValueInfo(i)
        name = vi.Name().decode("utf-8", errors="ignore")
        exps = tuple([vi.Exponents(j) for j in range(vi.ExponentsLength())])

        ti = vi.ValueInfoType()
        dims = ()
        if ti and ti.ValueType() == TypeInfoValue.tensor_type:
            from esp_ppq.parser.espdl.FlatBuffers.Dl.TensorTypeAndShape import TensorTypeAndShape
            from esp_ppq.parser.espdl.FlatBuffers.Dl.DimensionValueType import DimensionValueType
            tt = TensorTypeAndShape()
            tt.Init(ti.Value().Bytes, ti.Value().Pos)
            shape = tt.Shape()
            if shape:
                dlist = []
                for j in range(shape.DimLength()):
                    dim = shape.Dim(j)
                    if dim and dim.Value():
                        dval = dim.Value()
                        if dval.DimType() == DimensionValueType.VALUE:
                            dlist.append(dval.DimValue())
                dims = tuple(dlist)

        value_infos[name] = {"dims": dims, "exps": exps}

    # Build activation sequence: ONLY include post-activation tensors
    # (PRelu, Add, Concat outputs - NOT Conv outputs).
    # TFLite ReLU singlepath model has ReLU fused into Conv/DW,
    # so its Conv/DW outputs correspond to ESP-DL's PRelu outputs.
    POST_ACT_OPS = {"PRelu", "Add", "Concat"}
    activation_sequence = []
    for node in nodes:
        if node["op"] not in POST_ACT_OPS:
            continue
        for out_name in node["outputs"]:
            if out_name in value_infos and value_infos[out_name]["exps"]:
                info = value_infos[out_name]
                activation_sequence.append({
                    "node_name": node["name"],
                    "op": node["op"],
                    "output_name": out_name,
                    "dims": info["dims"],
                    "exp": info["exps"][0],
                    "scale": float(2.0 ** info["exps"][0]),
                })

    return {
        "nodes": nodes,
        "weight_exps": weight_exps,
        "activation_sequence": activation_sequence,
        "value_infos": value_infos,
    }


# ============================================================
# STEP 2: Build ESPDL weight lookup by HWIO dims
# ============================================================

def build_espdl_weight_lookup(espdl_data):
    """
    Build a lookup from (HWIO dims) → {name, exps} for matching.
    Also build a lookup by name.
    """
    by_dims = defaultdict(list)
    by_name = {}

    for name, info in espdl_data["weight_exps"].items():
        dims = info["dims"]
        exps = info["exps"]
        by_dims[dims].append({"name": name, "exps": exps})
        by_name[name] = {"dims": dims, "exps": exps}

    return dict(by_dims), by_name


# ============================================================
# STEP 3: Build the TFLite op → ESPDL weight mapping
# ============================================================

def build_weight_mapping(tflite_ops, tflite_tensors, espdl_weight_by_dims, espdl_weight_by_name, espdl_data):
    """
    For each TFLite Conv/DW op, find the matching ESPDL weight exponent.

    Returns: dict of tflite_tensor_idx → (exponents_list, is_per_channel, channel_sizes)

    Key considerations:
    - TFLite OHWI format → ESPDL HWIO: transpose dims
    - TFLite DW format [1, H, W, Cin] → ESPDL [H, W, Cin, 1]
    - Merged layers need special per-channel handling
    """

    # Build sequence of TFLite Conv ops in order
    tflite_conv_ops = []
    for op_idx, op in enumerate(tflite_ops):
        bk = op["code"]
        if bk in (CONV_2D, DEPTHWISE_CONV_2D):
            tflite_conv_ops.append({
                "op_idx": op_idx,
                "is_dw": (bk == DEPTHWISE_CONV_2D),
                "inputs": op["inputs"],
                "outputs": op["outputs"],
                "input_shapes": [tflite_tensors.get(inp, {}).get("shape", ()) for inp in op["inputs"] if inp >= 0],
                "output_shapes": [tflite_tensors.get(out, {}).get("shape", ()) for out in op["outputs"] if out >= 0],
            })

    print(f"  TFLite has {len(tflite_conv_ops)} Conv/DW ops")

    # Now build the mapping. Strategy:
    # For each TFLite Conv/DW op, try to find the matching ESPDL weight by:
    # 1. Convert TFLite weight dims [O,H,W,I] to ESPDL [H,W,I,O]
    # 2. Look up in espdl_weight_by_dims
    # 3. For merged layers (dconv_45, conv_6): handle specially

    weight_map = {}  # tflite_tensor_idx → quantization info

    # Build sequential list of ESPDL Conv weights (in graph order)
    espdl_nodes = []
    for node in espdl_data["nodes"]:
        if node["op"] == "Conv":
            if len(node["inputs"]) >= 2:
                wname = node["inputs"][1]
                bname = node["inputs"][2] if len(node["inputs"]) > 2 else None
                winfo = espdl_weight_by_name.get(wname, None)
                binfo = espdl_weight_by_name.get(bname, None) if bname else None
                espdl_nodes.append({
                    "node_name": node["name"],
                    "weight_name": wname,
                    "bias_name": bname,
                    "weight_info": winfo,
                    "bias_info": binfo,
                })

    print(f"  ESPDL has {len(espdl_nodes)} Conv nodes")

    # Now align the sequences
    # The ESPDL has 53 Conv nodes. TFLite has 50 Conv/DW nodes (33 Conv + 17 DW = 50).
    # ESPDL has the split layers (dconv_45 sep_1+sep_2, proj_1+proj_2, conv_6 sep_1+sep_2)
    # which are merged in TFLite.

    # Strategy: Walk both sequences, skipping ESPDL split pairs that map to one TFLite merged op

    tflite_idx = 0
    espdl_idx = 0

    # Identify which ESPDL nodes belong to splits
    # dconv_45_conv_sep_1 and dconv_45_conv_sep_2 are a split pair
    # dconv_45_conv_proj_1 and dconv_45_conv_proj_2 are a split pair
    # conv_6sep_1 and conv_6sep_2 are a split pair

    split_pairs = {
        ("dconv_45_conv_sep_1.weight", "dconv_45_conv_sep_2.weight"): {
            "tflite_out_ch": 512, "split_ch": [256, 256],
            "is_sep_merge": True,
        },
        ("dconv_45_conv_proj_1.weight", "dconv_45_conv_proj_2.weight"): {
            "tflite_out_ch": 128, "split_ch": [64, 64],
            "is_sep_merge": False,
        },
        ("conv_6sep_1.weight", "conv_6sep_2.weight"): {
            "tflite_out_ch": 512, "split_ch": [256, 256],
            "is_sep_merge": True,
        },
    }

    while tflite_idx < len(tflite_conv_ops) and espdl_idx < len(espdl_nodes):
        tfl_op = tflite_conv_ops[tflite_idx]
        espdl_n = espdl_nodes[espdl_idx]

        # Get TFLite weight tensor
        weight_input_idx = tfl_op["inputs"][1] if len(tfl_op["inputs"]) >= 2 else -1
        bias_input_idx = tfl_op["inputs"][2] if len(tfl_op["inputs"]) >= 3 else -1

        # Check if this ESPDL node is part of a split pair
        wname = espdl_n["weight_name"]
        matched = False
        for (w1, w2), merge_info in split_pairs.items():
            if wname == w1:
                # This is the first of a split pair. Check next espdl node
                if espdl_idx + 1 < len(espdl_nodes) and espdl_nodes[espdl_idx+1]["weight_name"] == w2:
                    # Match! This is a merged layer
                    espdl_w1 = espdl_n["weight_info"]
                    espdl_w2 = espdl_nodes[espdl_idx+1]["weight_info"]

                    if espdl_w1 and espdl_w2:
                        exp1 = espdl_w1["exps"][0]
                        exp2 = espdl_w2["exps"][0]
                        ch1 = merge_info["split_ch"][0]
                        ch2 = merge_info["split_ch"][1]

                        # Per-channel scales for merged weight
                        scales = [float(2.0 ** exp1)] * ch1 + [float(2.0 ** exp2)] * ch2

                        if weight_input_idx >= 0:
                            weight_map[weight_input_idx] = {
                                "type": "per_channel",
                                "exponents": scales,
                                "quantized_dimension": 0,
                                "is_bias": False,
                            }

                        # Handle bias
                        espdl_b1 = espdl_n["bias_info"]
                        espdl_b2 = espdl_nodes[espdl_idx+1]["bias_info"]
                        if bias_input_idx >= 0 and espdl_b1 and espdl_b2:
                            bias_exp = espdl_b1["exps"][0]
                            weight_map[bias_input_idx] = {
                                "type": "per_tensor",
                                "exponents": [float(2.0 ** bias_exp)],
                                "quantized_dimension": 0,
                                "is_bias": True,
                            }

                        matched = True
                        espdl_idx += 2  # Skip both split parts
                        break

        if not matched:
            # Regular 1:1 mapping
            espdl_w = espdl_n["weight_info"]
            if espdl_w and weight_input_idx >= 0:
                exps = espdl_w["exps"]
                scales = [float(2.0 ** e) for e in exps]
                if len(scales) == 1:
                    weight_map[weight_input_idx] = {
                        "type": "per_tensor",
                        "exponents": scales,
                        "quantized_dimension": 0,
                        "is_bias": False,
                    }
                else:
                    weight_map[weight_input_idx] = {
                        "type": "per_channel",
                        "exponents": scales,
                        "quantized_dimension": 0,
                        "is_bias": False,
                    }

            # Handle bias
            espdl_b = espdl_n["bias_info"]
            if espdl_b and bias_input_idx >= 0:
                weight_map[bias_input_idx] = {
                    "type": "per_tensor",
                    "exponents": [float(2.0 ** espdl_b["exps"][0])],
                    "quantized_dimension": 0,
                    "is_bias": True,
                }

            espdl_idx += 1

        tflite_idx += 1

    print(f"  Mapped {len(weight_map)} weight/bias tensors to espdl exponents")
    return weight_map


# ============================================================
# STEP 4: Build activation mapping
# ============================================================

def build_activation_mapping(tflite_ops, tflite_tensors, espdl_data):
    """
    For each TFLite op output tensor, find the matching ESPDL activation exponent.

    Strategy: Match by shape and sequential position.
    The TFLite model has ReLU fused, so each Conv/DW output is already post-activation.
    In ESPDL, we need to find the matching post-PRelu activation.
    """

    # Build shape-indexed sequence of ESPDL activations
    espdl_by_shape = defaultdict(list)
    for act in espdl_data["activation_sequence"]:
        dims = act["dims"]
        if dims:  # skip empty dims
            espdl_by_shape[dims].append(act["exp"])

    # Track consumption of each shape's exponents
    shape_counters = defaultdict(int)

    act_map = {}  # tflite_tensor_idx → {"scale": ..., "zp": ...}

    for op in tflite_ops:
        for out_idx in op["outputs"]:
            if out_idx < 0:
                continue
            shape = tflite_tensors.get(out_idx, {}).get("shape", ())
            if not shape:
                continue

            if shape in espdl_by_shape:
                entries = espdl_by_shape[shape]
                counter = shape_counters[shape]
                if counter < len(entries):
                    exp = entries[counter]
                    scale = float(2.0 ** exp)
                    act_map[out_idx] = {"scale": [scale], "zero_point": [0], "quantized_dimension": 0}
                    shape_counters[shape] += 1

    print(f"  Mapped {len(act_map)} activation tensors")
    return act_map


# ============================================================
# STEP 5: Parse TFLite float32 model
# ============================================================

def parse_tflite_float32(path):
    """Parse float32 TFLite model into a structured format."""
    with open(path, "rb") as f:
        data = f.read()

    model = tflite_schema.Model.GetRootAs(data, 0)
    subgraph = model.Subgraphs(0)

    TYPE_NAMES = {
        0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
        5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8", 10: "FLOAT64",
    }

    # Op codes
    op_codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        op_codes.append(oc.BuiltinCode())

    # Tensors
    tensors = {}
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        name = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
        shape = tuple([t.Shape(j) for j in range(t.ShapeLength())])
        dtype = t.Type()
        q = t.Quantization()
        qp = None
        if q:
            scales = [q.Scale(j) for j in range(q.ScaleLength())]
            zps = [q.ZeroPoint(j) for j in range(q.ZeroPointLength())]
            qp = {"scales": scales, "zero_points": zps, "quant_dim": q.QuantizedDimension()}
        tensors[i] = {
            "name": name,
            "shape": shape,
            "type": TYPE_NAMES.get(dtype, str(dtype)),
            "dtype": dtype,
            "quant": qp,
            "buffer_idx": t.Buffer(),
        }

    # Operators
    operators = []
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        bk = op_codes[op.OpcodeIndex()]
        inputs = [op.Inputs(j) for j in range(op.InputsLength())]
        outputs = [op.Outputs(j) for j in range(op.OutputsLength())]
        operators.append({
            "idx": i,
            "code": bk,
            "inputs": inputs,
            "outputs": outputs,
        })

    # Also get raw model and buffer data
    model_obj = tflite_schema.ModelT.InitFromPackedBuf(data, 0)

    return {
        "tensors": tensors,
        "operators": operators,
        "op_codes": op_codes,
        "model_obj": model_obj,
    }


# ============================================================
# STEP 6: Build INT8 TFLite with espdl quantization
# ============================================================

def build_int8_model(float32_data, weight_map, act_map, espdl_data):
    """
    Build the INT8 TFLite model:
    1. Quantize all weights using espdl exponents
    2. Set activation quantization params
    3. Set per-channel quantization for merged layers
    """
    print("\n  Building INT8 model...")

    model_obj = float32_data["model_obj"]
    subgraph = model_obj.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators
    buffers = model_obj.buffers
    op_codes = model_obj.operatorCodes

    # Collect op outputs
    op_outputs = set()
    for op in operators:
        for out_idx in op.outputs:
            if out_idx >= 0:
                op_outputs.add(out_idx)

    # --- Quantize weights ---
    weights_quantized = 0
    biases_quantized = 0

    for tensor_idx in range(len(tensors)):
        t = tensors[tensor_idx]

        if tensor_idx not in weight_map:
            continue

        # Check if this tensor has buffer data
        buf_idx = t.buffer
        if buf_idx < 0 or buf_idx >= len(buffers):
            continue
        buf = buffers[buf_idx]
        if buf.data is None or len(buf.data) == 0:
            continue

        # Get float32 data from buffer
        shape = list(t.shape) if t.shape is not None else []
        if len(shape) == 0:
            continue

        expected_size = int(np.prod(shape)) * 4  # float32 = 4 bytes
        actual_size = len(buf.data)

        if actual_size < expected_size:
            # Could be a non-float32 tensor or alignment issue
            continue

        # Read as float32
        float_data = np.frombuffer(bytes(buf.data), dtype=np.float32).reshape(shape)

        # Get espdl quantization scales
        qinfo = weight_map[tensor_idx]
        scales = qinfo["exponents"]  # list of scales (from 2^exp)
        qdim = qinfo.get("quantized_dimension", 0)
        is_bias = qinfo.get("is_bias", False)

        # Bias tensors are INT32, weight tensors are INT8
        if is_bias:
            target_dtype = np.int32
            tflite_dtype = tflite_schema.TensorType.INT32
            byte_mult = 4
            if qinfo["type"] == "per_tensor":
                scale = scales[0]
                int_data = np.clip(np.round(float_data / scale), -2**31, 2**31-1).astype(np.int32)
                buf.data = bytes(int_data.tobytes())
                t.type = tflite_dtype
                if t.quantization is None:
                    t.quantization = tflite_schema.QuantizationParametersT()
                t.quantization.scale = [float(scale)]
                t.quantization.zeroPoint = [0]
                t.quantization.quantizedDimension = 0
                biases_quantized += 1
            else:
                # Per-channel bias (rare but possible)
                n_scales = len(scales)
                scale_arr = np.array(scales, dtype=np.float64)
                int_data = np.clip(np.round(float_data / scale_arr), -2**31, 2**31-1).astype(np.int32)
                buf.data = bytes(int_data.tobytes())
                t.type = tflite_dtype
                if t.quantization is None:
                    t.quantization = tflite_schema.QuantizationParametersT()
                t.quantization.scale = [float(s) for s in scales]
                t.quantization.zeroPoint = [0] * len(scales)
                t.quantization.quantizedDimension = 0
                biases_quantized += 1

        elif qinfo["type"] == "per_tensor":
            scale = scales[0]
            # Quantize: int8 = round(float / scale)
            int_data = np.clip(np.round(float_data / scale), -128, 127).astype(np.int8)

            # Write back
            buf.data = bytes(int_data.tobytes())

            # Set quantization params
            t.type = tflite_schema.TensorType.INT8
            if t.quantization is None:
                t.quantization = tflite_schema.QuantizationParametersT()
            t.quantization.scale = [float(scale)]
            t.quantization.zeroPoint = [0]
            t.quantization.quantizedDimension = 0

            weights_quantized += 1

        elif qinfo["type"] == "per_channel":
            # Per-channel quantization
            # scales is a list of per-output-channel scales
            n_scales = len(scales)
            out_channels = shape[0] if shape[0] == n_scales else shape[-1]

            if n_scales != out_channels and len(shape) >= 2:
                # Try to figure out which axis matches
                for ax in range(len(shape)):
                    if shape[ax] == n_scales:
                        out_channels = shape[ax]
                        break

            if n_scales != out_channels:
                print(f"    WARNING: {n_scales} scales but shape={shape}, falling back to per-tensor")
                # Fallback: use first scale
                scale = scales[0]
                int_data = np.clip(np.round(float_data / scale), -128, 127).astype(np.int8)
                buf.data = bytes(int_data.tobytes())
                t.type = tflite_schema.TensorType.INT8
                if t.quantization is None:
                    t.quantization = tflite_schema.QuantizationParametersT()
                t.quantization.scale = [float(scale)]
                t.quantization.zeroPoint = [0]
                t.quantization.quantizedDimension = 0
                weights_quantized += 1
                continue

            # Broadcast scales for quantization
            # In OHWI format, axis 0 is output channels
            scale_arr = np.array(scales, dtype=np.float64)
            scale_shape = [1] * len(shape)
            scale_shape[0] = out_channels
            scale_arr = scale_arr.reshape(scale_shape)

            int_data = np.clip(np.round(float_data / scale_arr), -128, 127).astype(np.int8)
            buf.data = bytes(int_data.tobytes())

            # Set per-channel quantization params
            t.type = tflite_schema.TensorType.INT8
            if t.quantization is None:
                t.quantization = tflite_schema.QuantizationParametersT()
            t.quantization.scale = [float(s) for s in scales]
            t.quantization.zeroPoint = [0] * len(scales)
            t.quantization.quantizedDimension = 0  # output channel axis in OHWI

            weights_quantized += 1

            # Print merged layer info
            if len(set(scales)) > 1:
                unique_scales = sorted(set(scales))
                exp_str = ", ".join([f"2^{int(np.log2(s))}" for s in unique_scales])
                print(f"    Merged per-channel: {n_scales} scales, values={exp_str}")

    print(f"  Quantized {weights_quantized} weight tensors, {biases_quantized} bias tensors")

    # --- Set activation quantization ---
    acts_fixed = 0
    for tensor_idx, qinfo in act_map.items():
        if tensor_idx >= len(tensors):
            continue
        t = tensors[tensor_idx]

        # Change type to INT8
        t.type = tflite_schema.TensorType.INT8

        if t.quantization is None:
            t.quantization = tflite_schema.QuantizationParametersT()

        t.quantization.scale = qinfo["scale"]
        t.quantization.zeroPoint = qinfo["zero_point"]
        t.quantization.quantizedDimension = qinfo.get("quantized_dimension", 0)
        acts_fixed += 1

    # Fix PAD output tensors: they need to be INT8 with same scale as input
    pads_fixed = 0
    for op in operators:
        if op_codes[op.opcodeIndex].builtinCode == PAD:
            # PAD: inputs=[input_tensor, padding_constant]
            inp_idx = op.inputs[0]
            out_idx = op.outputs[0]
            if inp_idx >= 0 and out_idx >= 0:
                inp_t = tensors[inp_idx]
                out_t = tensors[out_idx]
                if out_t.type != tflite_schema.TensorType.INT8:
                    out_t.type = tflite_schema.TensorType.INT8
                    if out_t.quantization is None:
                        out_t.quantization = tflite_schema.QuantizationParametersT()
                    # PAD output has same quantization as input
                    if inp_t.quantization and inp_t.quantization.scale:
                        out_t.quantization.scale = list(inp_t.quantization.scale)
                        out_t.quantization.zeroPoint = list(inp_t.quantization.zeroPoint) if inp_t.quantization.zeroPoint else [0]
                        out_t.quantization.quantizedDimension = inp_t.quantization.quantizedDimension
                    else:
                        out_t.quantization.scale = [float(2.0 ** -6)]
                        out_t.quantization.zeroPoint = [0]
                        out_t.quantization.quantizedDimension = 0
                    pads_fixed += 1
    print(f"  Fixed {pads_fixed} PAD output quantization params")

    # --- Brute-force: fix ALL remaining unquantized activation tensors ---
    # Any 4D tensor (NHWC activation) that's still float32 gets INT8.
    # Use the most common scale for its spatial resolution, based on espdl.
    # Precedence: (1) scale from another INT8 tensor with same shape
    #             (2) default pow2 scale based on resolution
    shape_to_scale = {}
    for t in tensors:
        if t.type == tflite_schema.TensorType.INT8:
            sh = tuple(t.shape) if t.shape is not None and len(t.shape) > 0 else None
            if sh and len(sh) == 4 and t.quantization and t.quantization.scale:
                if sh not in shape_to_scale:
                    shape_to_scale[sh] = (list(t.quantization.scale),
                                          list(t.quantization.zeroPoint)
                                          if t.quantization.zeroPoint else [0],
                                          t.quantization.quantizedDimension)

    brute_fixed = 0
    for i, t in enumerate(tensors):
        sh = tuple(t.shape) if t.shape is not None and len(t.shape) > 0 else None
        if not sh or len(sh) != 4:
            continue
        # Skip already quantized
        if t.type == tflite_schema.TensorType.INT8 and t.quantization and t.quantization.scale:
            continue
        # Skip constant weights (named arith.constant)
        name = t.name.decode("utf-8", errors="ignore") if t.name else ""
        if name.startswith("arith.constant"):
            continue

        # Find scale from same-shape INT8 tensor, or use default
        if sh in shape_to_scale:
            scale, zp, qdim = shape_to_scale[sh]
        else:
            # Default: 2^-6 = 0.015625 for typical activations
            scale, zp, qdim = [0.015625], [0], 0

        t.type = tflite_schema.TensorType.INT8
        if t.quantization is None:
            t.quantization = tflite_schema.QuantizationParametersT()
        t.quantization.scale = list(scale)
        t.quantization.zeroPoint = list(zp)
        t.quantization.quantizedDimension = qdim
        brute_fixed += 1

    print(f"  Brute-force fixed {brute_fixed} unquantized activation tensors")

    # --- Fix PReLU alpha tensors: must be INT8 with same scale as input ---
    # In INT8 TFLite, PRELU requires alpha to be quantized same as input activation.
    PRELU = None
    for code_num, name in [(18, 'PRELU'), (33, 'RELU'), (0, 'ADD')]:
        pass  # Find actual PRELU opcode
    # PRELU opcode in TFLite is 18
    PRELU_OPCODE = 54  # TFLite BuiltinOperator PRELU in recent TF versions
    prelu_alphas_fixed = 0
    for op in operators:
        opcode = op_codes[op.opcodeIndex].builtinCode
        if opcode == PRELU_OPCODE:
            # PRELU: inputs=[activation, alpha], outputs=[result]
            if len(op.inputs) >= 2:
                act_idx = op.inputs[0]
                alpha_idx = op.inputs[1]
                if (act_idx >= 0 and alpha_idx >= 0 and
                    act_idx < len(tensors) and alpha_idx < len(tensors)):
                    act_t = tensors[act_idx]
                    alpha_t = tensors[alpha_idx]
                    # Alpha must have same quantization as activation input
                    if (act_t.type == tflite_schema.TensorType.INT8 and
                        act_t.quantization and act_t.quantization.scale):
                        act_scale = act_t.quantization.scale[0]
                        act_zp = (act_t.quantization.zeroPoint[0]
                                  if act_t.quantization.zeroPoint else 0)

                        # Get float32 alpha data and quantize to INT8
                        alpha_buf_idx = alpha_t.buffer
                        alpha_buf = buffers[alpha_buf_idx]
                        if alpha_buf.data is not None and len(alpha_buf.data) > 0:
                            alpha_shape = list(alpha_t.shape) if alpha_t.shape is not None and len(alpha_t.shape) > 0 else []
                            alpha_float = np.frombuffer(bytes(alpha_buf.data),
                                                        dtype=np.float32).reshape(alpha_shape)
                            alpha_int8 = np.clip(np.round(alpha_float / act_scale),
                                                -128, 127).astype(np.int8)
                            alpha_buf.data = bytes(alpha_int8.tobytes())
                            alpha_t.type = tflite_schema.TensorType.INT8
                            if alpha_t.quantization is None:
                                alpha_t.quantization = tflite_schema.QuantizationParametersT()
                            alpha_t.quantization.scale = [act_scale]
                            alpha_t.quantization.zeroPoint = [act_zp]
                            alpha_t.quantization.quantizedDimension = 0
                            prelu_alphas_fixed += 1

    print(f"  Fixed {prelu_alphas_fixed} PReLU alpha tensors")
    # But TFLite will allocate them at runtime, so the buffer size doesn't matter for activations
    for tensor_idx in range(len(tensors)):
        t = tensors[tensor_idx]
        if t.type == tflite_schema.TensorType.INT8 and t.buffer > 0:
            # Only zero out non-weight buffers (weights already handled)
            if tensor_idx not in weight_map:
                buf = buffers[t.buffer]
                if buf.data is not None and len(buf.data) > 0:
                    shape = list(t.shape) if t.shape is not None else []
                    if len(shape) > 0:
                        sz = int(np.prod(shape))
                        # Set to zeros (INT8)
                        buf.data = bytes([0] * sz)

    print(f"  Fixed {acts_fixed} activation quantization params")

    # --- Update operator codes to add INT8-related ops if needed ---
    # Check if we need to add QUANTIZE/DEQUANTIZE op codes

    # --- Set model-level quantization to empty (indicates quantization at tensor level) ---
    # The model description
    if not model_obj.metadata:
        model_obj.metadata = []

    # --- Fix input/output tensors ---
    # Input: set to INT8 with input exponent
    if "input" in espdl_data["value_infos"]:
        input_exp = espdl_data["value_infos"]["input"]["exps"][0]
        input_scale = float(2.0 ** input_exp)
    else:
        input_scale = float(2.0 ** -6)

    if "embedding" in espdl_data["value_infos"]:
        output_exp = espdl_data["value_infos"]["embedding"]["exps"][0]
        output_scale = float(2.0 ** output_exp)
    else:
        output_scale = float(2.0 ** -5)

    for t in tensors:
        # Input tensor: shape (1, 112, 112, 3), first tensor
        if hasattr(t, 'shape') and len(t.shape) == 4:
            sh = list(t.shape)
            if sh == [1, 112, 112, 3]:
                t.type = tflite_schema.TensorType.INT8
                if t.quantization is None:
                    t.quantization = tflite_schema.QuantizationParametersT()
                t.quantization.scale = [input_scale]
                t.quantization.zeroPoint = [0]
                t.quantization.quantizedDimension = 0

    # Set output tensor scaling
    # Look for the last Conv output (1, 1, 1, 512)
    for idx in range(len(tensors)):
        t = tensors[idx]
        if hasattr(t, 'shape') and len(t.shape) == 4:
            sh = list(t.shape)
            name = t.name.decode("utf-8") if t.name else ""
            if sh == [1, 1, 1, 512] and idx in op_outputs:
                # Check if this is the embedding output
                if name == "Identity" or "embedding" in name.lower() or idx == len(tensors) - 1:
                    t.type = tflite_schema.TensorType.INT8
                    if t.quantization is None:
                        t.quantization = tflite_schema.QuantizationParametersT()
                    t.quantization.scale = [output_scale]
                    t.quantization.zeroPoint = [0]
                    t.quantization.quantizedDimension = 0
                    break

    # --- Fix graph: Remove MUL/ADD preprocessing (not compatible with INT8) ---
    # The float32 model has: input -> MUL -> ADD -> PAD -> Conv
    # In INT8 mode, preprocessing should happen before quantization.
    # Remove ops 0 (MUL) and 1 (ADD), wire input directly to PAD.
    print("  Restructuring graph: removing MUL/ADD preprocessing...")

    # Find operators with code MUL and ADD
    ops_to_remove = []
    pad_op = None
    input_tensor_idx = 0  # The input tensor

    for i, op in enumerate(operators):
        bk = op_codes[op.opcodeIndex].builtinCode
        if bk == MUL:
            ops_to_remove.append(i)
            print(f"    Found MUL at op {i}")
        elif bk == ADD:
            # The preprocessing ADD (first ADD in graph)
            if i <= 2:  # Near the start
                ops_to_remove.append(i)
                print(f"    Found ADD at op {i}")
        elif bk == PAD and pad_op is None:
            pad_op = i
            print(f"    Found PAD at op {i}")

    # Rewire PAD first input to the model input tensor
    if pad_op is not None:
        old_inputs = list(operators[pad_op].inputs)
        old_input = old_inputs[0]
        old_inputs[0] = input_tensor_idx
        operators[pad_op].inputs = old_inputs
        print(f"    Rewired PAD input {old_input} -> {input_tensor_idx}")

    # Remove MUL and ADD ops (in reverse order to preserve indices)
    for op_idx in sorted(ops_to_remove, reverse=True):
        del operators[op_idx]
        print(f"    Removed op {op_idx}")

    # Also update operator codes if needed (remove MUL/ADD codes if they're the only users)
    # This is optional - unused codes are OK

    # --- Pack model ---
    builder = flatbuffers.Builder(10 * 1024 * 1024)  # 10 MB
    packed = model_obj.Pack(builder)
    builder.Finish(packed, b"TFL3")
    new_data = bytes(builder.Output())

    return new_data


# ============================================================
# STEP 7: Verification
# ============================================================

def verify_model(tflite_path, weight_map, act_map):
    """Verify the output model."""
    print("\n=== VERIFICATION ===")

    with open(tflite_path, "rb") as f:
        data = f.read()

    model = tflite_schema.Model.GetRootAs(data, 0)
    subgraph = model.Subgraphs(0)

    print(f"Tensors: {subgraph.TensorsLength()}")
    print(f"Operators: {subgraph.OperatorsLength()}")

    # Count INT8 tensors
    int8_count = 0
    int8_quantized = 0
    pow2_count = 0

    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        if t.Type() != 9:  # INT8
            continue
        int8_count += 1
        q = t.Quantization()
        if q and q.ScaleLength() > 0:
            int8_quantized += 1
            scales = [q.Scale(j) for j in range(q.ScaleLength())]
            for s in scales:
                log2 = np.log2(abs(s))
                if abs(log2 - round(log2)) < 0.01:
                    pow2_count += 1
                    break

    print(f"INT8 tensors: {int8_count}, quantized: {int8_quantized}, with pow2 scales: {pow2_count}")

    # Show sample quantization
    print("\nSample quantization scales:")
    shown = 0
    for i in range(subgraph.TensorsLength()):
        if shown >= 20:
            break
        t = subgraph.Tensors(i)
        q = t.Quantization()
        if q and q.ScaleLength() > 0:
            name = t.Name().decode("utf-8") if t.Name() else f"t_{i}"
            scales = [q.Scale(j) for j in range(min(3, q.ScaleLength()))]
            zps = [q.ZeroPoint(j) for j in range(min(3, q.ZeroPointLength()))]
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            dtype = t.Type()

            is_pow2 = False
            for s in scales:
                log2 = np.log2(abs(s))
                if abs(log2 - round(log2)) < 0.01:
                    is_pow2 = True
                    break
            marker = " [POW2]" if is_pow2 else ""

            n_scales = q.ScaleLength()
            if n_scales == 1:
                print(f"  {name:50s} type={dtype} shape={tuple(shape)} scale={scales[0]:.8f}{marker}")
            else:
                print(f"  {name:50s} type={dtype} shape={tuple(shape)} {n_scales} per-ch, first={scales[0]:.8f}{marker}")
            shown += 1

    # Verify weight data
    print("\nWeight data spot check:")
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        if t.Type() == 9:  # INT8
            name = t.Name().decode("utf-8") if t.Name() else f"t_{i}"
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            q = t.Quantization()
            if q and q.ScaleLength() > 1:
                scales = [q.Scale(j) for j in range(q.ScaleLength())]
                unique_scales = sorted(set([round(s, 10) for s in scales]))
                print(f"  {name}: shape={tuple(shape)}, {len(scales)} scales, unique={unique_scales}")

    return subgraph.TensorsLength()


def test_cosine_similarity(tflite_path):
    """Run cosine similarity test on INT8 model."""
    print("\n=== COSINE SIMILARITY TEST ===")

    import tensorflow as tf

    # Look for test images
    calib_dir = SCRIPT_DIR.parent / "calibration_data" / "qat_112"
    images = sorted(calib_dir.glob("*.jpg")) if calib_dir.exists() else []

    if len(images) < 4:
        print("  Not enough test images, using random data")
        img1 = np.random.randn(1, 112, 112, 3).astype(np.float32)
        img2 = np.random.randn(1, 112, 112, 3).astype(np.float32)
        img3 = np.random.randn(1, 112, 112, 3).astype(np.float32)
        img4 = np.random.randn(1, 112, 112, 3).astype(np.float32)
    else:
        from PIL import Image
        imgs = []
        for idx in [0, 1, 2, 3]:
            if idx < len(images):
                img = Image.open(images[idx * max(1, len(images)//5)]).convert("RGB").resize((112, 112))
                arr = np.array(img, dtype=np.float32)
                arr = (arr / 127.5) - 1.0  # Normalize to [-1, 1]
                imgs.append(np.expand_dims(arr, 0))
            else:
                imgs.append(np.random.randn(1, 112, 112, 3).astype(np.float32))
        img1, img2, img3, img4 = imgs

    try:
        os.environ["TF_ENABLE_XNNPACK"] = "0"
        interp = tf.lite.Interpreter(
            model_path=str(tflite_path),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        )
        interp.allocate_tensors()
        in_details = interp.get_input_details()
        out_details = interp.get_output_details()

        print(f"  Input:  {in_details[0]}")
        print(f"  Output: {out_details[0]}")

        input_scale = in_details[0]["quantization_parameters"]["scales"][0]
        input_zp = in_details[0]["quantization_parameters"]["zero_points"][0]
        output_scale = out_details[0]["quantization_parameters"]["scales"][0]
        output_zp = out_details[0]["quantization_parameters"]["zero_points"][0]

        print(f"  Input  scale={input_scale:.8f}, zp={input_zp}")
        print(f"  Output scale={output_scale:.8f}, zp={output_zp}")

        def get_embedding(float_img):
            q = np.clip(np.round(float_img / input_scale) + input_zp, -128, 127).astype(np.int8)
            interp.set_tensor(in_details[0]["index"], q)
            interp.invoke()
            raw = interp.get_tensor(out_details[0]["index"])
            emb = (raw.astype(np.float32) - output_zp) * output_scale
            emb = emb.flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb

        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        emb3 = get_embedding(img3)
        emb4 = get_embedding(img4)

        print("\n  Pairwise cosine similarities:")
        pairs = [
            ("img0", "img1", emb1, emb2),
            ("img0", "img2", emb1, emb3),
            ("img0", "img3", emb1, emb4),
            ("img1", "img2", emb2, emb3),
            ("img1", "img3", emb2, emb4),
            ("img2", "img3", emb3, emb4),
        ]
        cos_sims = []
        for n1, n2, e1, e2 in pairs:
            cs = float(np.dot(e1, e2))
            cos_sims.append(cs)
            print(f"    {n1} vs {n2}: {cs:.6f}")

        # Self-similarity
        emb1a = get_embedding(img1)
        self_sim = float(np.dot(emb1, emb1a))
        print(f"\n  Self cosine similarity: {self_sim:.6f} (should be ~1.0)")

        avg_cos = np.mean(cos_sims)
        print(f"  Average cosine similarity (different faces): {avg_cos:.6f}")

        # Also compare with float32 if available
        f32_path = SCRIPT_DIR / "mfn_sp_relu_saved_model/mfn_s8_v1_singlepath_relu_float32.tflite"
        if f32_path.exists():
            print("\n  Comparing with float32 model...")
            interp_f32 = tf.lite.Interpreter(model_path=str(f32_path))
            interp_f32.allocate_tensors()
            f32_in = interp_f32.get_input_details()
            f32_out = interp_f32.get_output_details()

            interp_f32.set_tensor(f32_in[0]["index"], img1)
            interp_f32.invoke()
            ref1 = interp_f32.get_tensor(f32_out[0]["index"]).flatten()
            ref1 = ref1 / (np.linalg.norm(ref1) + 1e-8)

            interp_f32.set_tensor(f32_in[0]["index"], img2)
            interp_f32.invoke()
            ref2 = interp_f32.get_tensor(f32_out[0]["index"]).flatten()
            ref2 = ref2 / (np.linalg.norm(ref2) + 1e-8)

            ref_cos = float(np.dot(ref1, ref2))
            cross_cos = float(np.dot(emb1, ref1))
            print(f"    Float32 cosine (0 vs 1): {ref_cos:.6f}")
            print(f"    INT8 vs Float32 cosine (img0): {cross_cos:.6f}")

        # Verdict
        print(f"\n  VERDICT:", end=" ")
        if 0.05 < avg_cos < 0.9 and self_sim > 0.99:
            print(f"MODEL IS WORKING (avg_cos={avg_cos:.4f}, not collapsed)")
        elif avg_cos > 0.95:
            print(f"Model may be COLLAPSED (cos_sim too high: {avg_cos:.4f})")
        elif avg_cos < 0.03:
            print(f"Model may produce near-zero embeddings")
        else:
            print(f"Inconclusive (avg_cos={avg_cos:.4f})")

        return avg_cos, self_sim

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ESP-DL Quantization Fix - BOTH Weights AND Activations")
    print("ReLU Singlepath Architecture")
    print("=" * 70)

    # Step 1: Parse .espdl
    print("\n[1/5] Parsing .espdl...")
    espdl_data = parse_espdl(ESPDL_PATH)
    print(f"  Nodes: {len(espdl_data['nodes'])}")
    print(f"  Activations with exponents: {len(espdl_data['activation_sequence'])}")
    print(f"  Weights with exponents: {len(espdl_data['weight_exps'])}")

    # Show activation exponent distribution
    from collections import Counter
    shape_counts = Counter()
    for act in espdl_data["activation_sequence"]:
        shape_counts[act["dims"]] += 1
    print("  Activation by shape:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        if count > 1:
            exps = sorted(set(a["exp"] for a in espdl_data["activation_sequence"] if a["dims"] == shape))
            print(f"    {shape}: {count} entries, exps={exps}")

    # Step 2: Parse float32 TFLite
    print(f"\n[2/5] Parsing float32 TFLite: {FLOAT32_TFLITE}")
    f32_data = parse_tflite_float32(FLOAT32_TFLITE)
    print(f"  Tensors: {len(f32_data['tensors'])}")
    print(f"  Operators: {len(f32_data['operators'])}")

    # Step 3: Build weight mapping
    print("\n[3/5] Building weight mapping...")
    espdl_weight_by_dims, espdl_weight_by_name = build_espdl_weight_lookup(espdl_data)
    weight_map = build_weight_mapping(
        f32_data["operators"], f32_data["tensors"],
        espdl_weight_by_dims, espdl_weight_by_name, espdl_data
    )

    # Step 4: Build activation mapping
    print("\n[4/5] Building activation mapping...")
    act_map = build_activation_mapping(
        f32_data["operators"], f32_data["tensors"], espdl_data
    )

    # Step 5: Build INT8 model
    print("\n[5/5] Building INT8 TFLite model...")
    int8_data = build_int8_model(f32_data, weight_map, act_map, espdl_data)

    OUT_TFLITE.write_bytes(int8_data)
    print(f"\n  Saved to {OUT_TFLITE} ({len(int8_data) / 1024:.1f} KiB)")

    # Verify
    verify_model(OUT_TFLITE, weight_map, act_map)

    # Test cosine similarity
    test_cosine_similarity(OUT_TFLITE)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    return OUT_TFLITE


if __name__ == "__main__":
    os.chdir(BASE)
    main()
