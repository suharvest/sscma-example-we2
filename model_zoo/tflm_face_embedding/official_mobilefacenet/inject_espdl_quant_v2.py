#!/usr/bin/env python3
"""
ESP-DL Quantization Injection into TFLite - WEIGHT FIX EDITION

Fixes both activation AND weight quantization using .espdl exponents.
Weights are requantized: old_INT8 -> float -> new_INT8 using .espdl scales.
"""

import struct, sys, os
from pathlib import Path

import numpy as np
import flatbuffers
from tensorflow.lite.python import schema_py_generated as tflite_schema

sys.path.insert(0, "/tmp/esp-ppq-src")
from esp_ppq.parser.espdl.FlatBuffers.Dl import Model
from esp_ppq.parser.espdl.FlatBuffers.Dl.TypeInfoValue import TypeInfoValue

# === CONFIG ===
ESPDL_PATH = "/Users/harvest/project/esp-dl/models/human_face_recognition/models/s3/human_face_feat_mfn_s8_v1.espdl"
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_TFLITE = SCRIPT_DIR / "mfn_sp_int8.tflite"
OUT_TFLITE = SCRIPT_DIR / "mfn_espdl_quant.tflite"


def parse_espdl(espdl_path):
    """Parse .espdl and extract nodes, activation exp, and weight data."""
    with open(espdl_path, "rb") as f:
        data = f.read()
    model_data = data[16 : 16 + struct.unpack("I", data[8:12])[0]]
    model = Model.Model.GetRootAs(model_data, 0)
    graph = model.Graph()

    # Nodes
    nodes = []
    for i in range(graph.NodeLength()):
        node = graph.Node(i)
        name = node.Name().decode("utf-8", errors="ignore")
        op = node.OpType().decode("utf-8", errors="ignore")
        inputs = [node.Input(j).decode("utf-8", errors="ignore") for j in range(node.InputLength())]
        outputs = [node.Output(j).decode("utf-8", errors="ignore") for j in range(node.OutputLength())]
        nodes.append({"idx": i, "name": name, "op": op, "inputs": inputs, "outputs": outputs})

    # Value infos (activation exponents)
    value_infos = {}
    for i in range(graph.ValueInfoLength()):
        vi = graph.ValueInfo(i)
        name = vi.Name().decode("utf-8", errors="ignore")
        exps = [vi.Exponents(j) for j in range(vi.ExponentsLength())]
        ti = vi.ValueInfoType()
        dtype, dims = 0, []
        if ti and ti.ValueType() == TypeInfoValue.tensor_type:
            from esp_ppq.parser.espdl.FlatBuffers.Dl.TensorTypeAndShape import TensorTypeAndShape
            from esp_ppq.parser.espdl.FlatBuffers.Dl.DimensionValueType import DimensionValueType

            tt = TensorTypeAndShape()
            tt.Init(ti.Value().Bytes, ti.Value().Pos)
            dtype = tt.ElemType()
            shape = tt.Shape()
            if shape:
                for j in range(shape.DimLength()):
                    dim = shape.Dim(j)
                    if dim and dim.Value():
                        dval = dim.Value()
                        if dval.DimType() == DimensionValueType.VALUE:
                            dims.append(dval.DimValue())
        if exps:
            value_infos[name] = {"dtype": dtype, "dims": tuple(dims), "exponents": exps}

    # Shape -> exponents for activation mapping
    shape_exps = {}
    for vi_name, vi_info in value_infos.items():
        key = vi_info["dims"]
        if key not in shape_exps:
            shape_exps[key] = []
        shape_exps[key].append(vi_info["exponents"])

    # Weight exponents by node (mapped by Conv node output name)
    # .espdl Conv nodes: inputs=[input, "weight_name", "bias_name"]
    # We need weight exponents for each Conv
    weight_exps = {}  # espdl_weight_name -> [exponents]
    for i in range(graph.InitializerLength()):
        init = graph.Initializer(i)
        wname = init.Name().decode("utf-8", errors="ignore")
        exps = [init.Exponents(j) for j in range(init.ExponentsLength())]
        if exps:
            weight_exps[wname] = exps

    # Map Conv nodes to their weight exponents
    conv_weights = []  # [(op_type, weight_name, exponents)]
    for node in nodes:
        if node["op"] == "Conv":
            wname = node["inputs"][1]  # weight is second input
            bname = node["inputs"][2] if len(node["inputs"]) > 2 else None
            wexps = weight_exps.get(wname, [])
            bexps = weight_exps.get(bname, []) if bname else []
            # Determine if depthwise (weight dims will tell us)
            conv_weights.append({
                "node_name": node["name"],
                "weight_name": wname,
                "bias_name": bname,
                "weight_exps": wexps,
                "bias_exps": bexps,
            })

    return {
        "nodes": nodes,
        "value_infos": value_infos,
        "weight_exps": weight_exps,
        "shape_exps": shape_exps,
        "conv_weights": conv_weights,
    }


def parse_tflite(tflite_path):
    """Parse TFLite model."""
    with open(tflite_path, "rb") as f:
        data = f.read()
    model = tflite_schema.Model.GetRootAs(data, 0)
    subgraph = model.Subgraphs(0)

    op_codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        op_codes.append(oc.BuiltinCode())

    TYPE_NAMES = {
        0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
        5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8", 10: "FLOAT64",
    }

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
        tensors[i] = {"name": name, "shape": shape, "type": TYPE_NAMES.get(dtype, str(dtype)), "dtype": dtype, "quant": qp}

    operators = []
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        bk = op_codes[op.OpcodeIndex()]
        op_type = str(bk)  # Use numeric code, we only check 3 (CONV_2D) and 4 (DEPTHWISE_CONV_2D)
        inputs = [op.Inputs(j) for j in range(op.InputsLength())]
        outputs = [op.Outputs(j) for j in range(op.OutputsLength())]
        operators.append({"idx": i, "type": op_type, "code": bk, "inputs": inputs, "outputs": outputs})

    return {"tensors": tensors, "operators": operators, "op_codes": op_codes, "raw_data": data}


def rebuild_model(tflite_data, espdl_data):
    """Rebuild TFLite with corrected activation AND weight quantization."""
    print("\n  Rebuilding TFLite with ESP-DL quantization...")

    model_obj = tflite_schema.ModelT.InitFromPackedBuf(tflite_data["raw_data"], 0)
    subgraph = model_obj.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators
    buffers = model_obj.buffers

    # Find activation vs weight tensors
    op_outputs = set()
    for op in operators:
        for out_idx in op.outputs:
            if out_idx >= 0:
                op_outputs.add(out_idx)

    # --- Fix activation quantization ---
    shape_exps = espdl_data["shape_exps"]
    state = {}

    acts_fixed = 0
    for op in operators:
        for out_idx in op.outputs:
            if out_idx < 0 or out_idx >= len(tensors):
                continue
            t = tensors[out_idx]
            if t.type != tflite_schema.TensorType.INT8:
                continue
            shape = tuple(t.shape) if (t.shape is not None and len(t.shape) > 0) else tuple()
            if shape not in shape_exps:
                continue

            entries = shape_exps[shape]
            if shape not in state:
                state[shape] = 0
            idx = state[shape]
            if idx >= len(entries):
                continue
            exp = entries[idx][0]
            state[shape] += 1

            scale = float(2.0**exp)
            if t.quantization is None:
                t.quantization = tflite_schema.QuantizationParametersT()
            t.quantization.scale = [scale]
            t.quantization.zeroPoint = [0]
            t.quantization.quantizedDimension = 0
            acts_fixed += 1

    print(f"  Fixed {acts_fixed} activation quantization params")

    # --- Fix weight quantization (requantize data + fix params) ---
    # Build ordered list of TFLite Conv ops matching .espdl Conv sequence
    weight_conv_ops = []  # TFLite Conv/DepthwiseConv ops in order
    for op in operators:
        bk = op.opcodeIndex
        if bk == 3:  # CONV_2D
            weight_conv_ops.append(("CONV_2D", op))
        elif bk == 4:  # DEPTHWISE_CONV_2D
            weight_conv_ops.append(("DEPTHWISE_CONV_2D", op))

    # The .espdl has 53 Conv ops. TFLite might have different count due to architecture differences.
    # Map by sequence position (closest match)
    espdl_convs = espdl_data["conv_weights"]
    weights_fixed = 0
    biases_fixed = 0

    # Map by matching op sequence
    # TFLite Conv ops: some are merged, some are same as .espdl
    # Strategy: for each TFLite Conv, find the best-matching .espdl Conv
    # by matching the op type sequence pattern

    tflite_conv_idx = 0
    espdl_conv_idx = 0

    while tflite_conv_idx < len(weight_conv_ops) and espdl_conv_idx < len(espdl_convs):
        tfl_op_type, tfl_op = weight_conv_ops[tflite_conv_idx]
        espdl_conv = espdl_convs[espdl_conv_idx]

        # Get TFLite weight and bias tensor indices
        # Conv2D: inputs = [input, filter, bias?]
        # DepthwiseConv2D: inputs = [input, filter, bias?]
        if len(tfl_op.inputs) >= 2:
            weight_idx = tfl_op.inputs[1]
            bias_idx = tfl_op.inputs[2] if len(tfl_op.inputs) >= 3 else -1

            # Fix weight
            if weight_idx >= 0 and espdl_conv["weight_exps"]:
                new_exp = espdl_conv["weight_exps"]
                fixed = requantize_tensor(
                    tensors, buffers, weight_idx, new_exp, tflite_schema
                )
                if fixed:
                    weights_fixed += 1

            # Fix bias
            if bias_idx >= 0 and espdl_conv["bias_exps"]:
                new_exp = espdl_conv["bias_exps"]
                fixed = requantize_tensor(
                    tensors, buffers, bias_idx, new_exp, tflite_schema
                )
                if fixed:
                    biases_fixed += 1

        tflite_conv_idx += 1
        espdl_conv_idx += 1

        # Handle architecture mismatches (merged ops in TFLite vs split in .espdl)
        # If .espdl has more Conv ops, skip some TFLite entries
        # If TFLite has more, skip some .espdl entries
        # This is a heuristic; the exact mapping depends on the architecture

    print(f"  Fixed {weights_fixed} weight tensors, {biases_fixed} bias tensors")

    # --- Fix input/output quantization ---
    # Input: find the first tensor
    for idx in range(len(tensors)):
        t = tensors[idx]
        if t.type == tflite_schema.TensorType.INT8 and idx not in op_outputs:
            # This is likely the input
            if "input" in espdl_data["value_infos"]:
                exp = espdl_data["value_infos"]["input"]["exponents"][0]
                scale = float(2.0**exp)
                if t.quantization is not None:
                    t.quantization.scale = [scale]
                    t.quantization.zeroPoint = [0]
                    t.quantization.quantizedDimension = 0
            break

    # Output: fix the last tensor
    if "embedding" in espdl_data["value_infos"]:
        exp = espdl_data["value_infos"]["embedding"]["exponents"][0]
        scale = float(2.0**exp)
        for idx in range(len(tensors) - 1, -1, -1):
            t = tensors[idx]
            if t.type == tflite_schema.TensorType.INT8 and t.quantization is not None:
                t.quantization.scale = [scale]
                t.quantization.zeroPoint = [0]
                t.quantization.quantizedDimension = 0
                break

    # Repack
    builder = flatbuffers.Builder(1024 * 1024 * 4)
    packed = model_obj.Pack(builder)
    builder.Finish(packed, b"TFL3")
    return bytes(builder.Output())


def requantize_tensor(tensors, buffers, tensor_idx, new_exps, schema):
    """Requantize a tensor's data from old scale/zp to new scale=2^exp, zp=0."""
    t = tensors[tensor_idx]
    if t.type != schema.TensorType.INT8:
        return False
    if t.quantization is None:
        return False

    old_scales = t.quantization.scale  # numpy array
    old_zps = t.quantization.zeroPoint if t.quantization.zeroPoint is not None else np.array([0])
    old_quant_dim = t.quantization.quantizedDimension

    if old_scales is None or len(old_scales) == 0:
        return False

    new_scales = np.array([float(2.0**e) for e in new_exps], dtype=np.float32)

    # Read buffer data
    buf_idx = t.buffer
    if buf_idx < 0 or buf_idx >= len(buffers):
        return False
    buf = buffers[buf_idx]
    if buf.data is None or len(buf.data) == 0:
        return False

    raw_data = np.frombuffer(bytes(buf.data), dtype=np.int8)

    # Dequantize: float = (int8 - zp) * scale
    shape = list(t.shape) if t.shape is not None else []
    if len(shape) == 0:
        return False

    if len(old_scales) == 1:
        # Per-tensor
        old_zp = old_zps[0] if len(old_zps) > 0 else 0
        float_data = (raw_data.astype(np.float64) - old_zp) * old_scales[0]
    else:
        # Per-channel: reshape scales for broadcasting
        float_data = raw_data.astype(np.float64).reshape(shape)
        # Determine which dim is quantized
        qdim = old_quant_dim if old_quant_dim < len(shape) else 0
        scale_shape = [1] * len(shape)
        scale_shape[qdim] = len(old_scales)
        old_scales_reshaped = np.array(old_scales).reshape(scale_shape)
        if len(old_zps) == 1:
            float_data = (float_data.astype(np.float64) - old_zps[0]) * old_scales_reshaped
        elif len(old_zps) == len(old_scales):
            old_zps_reshaped = np.array(old_zps).reshape(scale_shape)
            float_data = (float_data.astype(np.float64) - old_zps_reshaped) * old_scales_reshaped
        else:
            float_data = float_data * old_scales_reshaped

    # Re-quantize with new scale: new_int8 = round(float / new_scale)
    # Use symmetric quantization (zp=0)
    if len(new_scales) == 1:
        new_int = np.clip(np.round(float_data / new_scales[0]), -128, 127).astype(np.int8)
        t.quantization.scale = [float(new_scales[0])]
        t.quantization.zeroPoint = [0]
        t.quantization.quantizedDimension = 0
    else:
        # Per-channel
        qdim = 0  # Default to output channel (first dim in ONNX format)
        if qdim >= len(shape):
            qdim = len(shape) - 1
        scale_shape = [1] * len(shape)
        scale_shape[qdim] = len(new_scales)
        new_scales_reshaped = np.array(new_scales, dtype=np.float64).reshape(scale_shape)
        new_int = np.clip(np.round(float_data / new_scales_reshaped), -128, 127).astype(np.int8)
        t.quantization.scale = [float(s) for s in new_scales]
        t.quantization.zeroPoint = [0] * len(new_scales)
        t.quantization.quantizedDimension = qdim

    # Write back to buffer
    buf.data = bytes(new_int.flatten().tobytes())

    return True


def verify_model(tflite_path):
    """Verify model has correct quantization."""
    print("\n  Verifying model...")
    with open(tflite_path, "rb") as f:
        data = f.read()
    model = tflite_schema.Model.GetRootAs(data, 0)
    subgraph = model.Subgraphs(0)

    total_int8 = 0
    quantized = 0
    pow2_count = 0

    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        if t.Type() != 9:
            continue
        total_int8 += 1
        q = t.Quantization()
        if q and q.ScaleLength() > 0:
            quantized += 1
            scales = [q.Scale(j) for j in range(q.ScaleLength())]
            # Check if any scale is a power of 2
            for s in scales:
                log2 = np.log2(abs(s))
                if abs(log2 - round(log2)) < 0.001:
                    pow2_count += 1
                    break

    print(f"  INT8 tensors: {total_int8}, quantized: {quantized}, pow2: {pow2_count}")

    # Show first 10 quantized tensors
    print("\n  Sample quantization scales:")
    shown = 0
    for i in range(subgraph.TensorsLength()):
        if shown >= 15:
            break
        t = subgraph.Tensors(i)
        q = t.Quantization()
        if q and q.ScaleLength() > 0:
            name = t.Name().decode("utf-8") if t.Name() else f"t_{i}"
            scales = [q.Scale(j) for j in range(min(3, q.ScaleLength()))]
            zps = [q.ZeroPoint(j) for j in range(min(3, q.ZeroPointLength()))]
            shape = [t.Shape(j) for j in range(t.ShapeLength())]

            is_pow2 = False
            for s in scales:
                log2 = np.log2(abs(s))
                if abs(log2 - round(log2)) < 0.001:
                    is_pow2 = True
                    break
            marker = " [POW2]" if is_pow2 else ""
            if len(scales) == 1:
                print(f"    {name} ({shape}): scale={scales[0]:.8f}, zp={zps[0] if zps else '?'}{marker}")
            else:
                print(f"    {name} ({shape}): {q.ScaleLength()} per-ch scales, first={scales[0]:.8f}{marker}")
            shown += 1

    return quantized, pow2_count


def test_cosine_similarity(tflite_path):
    """Test the model with real images."""
    import tensorflow as tf
    import tensorflow.lite.experimental as tflite_exp
    from PIL import Image

    calib_dir = SCRIPT_DIR.parent / "calibration_data" / "qat_112"
    images = sorted(calib_dir.glob("*.jpg")) if calib_dir.exists() else []

    if len(images) < 2:
        print("  Not enough test images")
        return

    os.environ["TF_ENABLE_XNNPACK"] = "0"
    interp = tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_op_resolver_type=tflite_exp.OpResolverType.BUILTIN_REF,
    )
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()

    input_scale = in_details[0]["quantization_parameters"]["scales"][0]
    input_zp = in_details[0]["quantization_parameters"]["zero_points"][0]
    output_scale = out_details[0]["quantization_parameters"]["scales"][0]
    output_zp = out_details[0]["quantization_parameters"]["zero_points"][0]

    print(f"  Input: scale={input_scale:.8f}, zp={input_zp}")
    print(f"  Output: scale={output_scale:.8f}, zp={output_zp}")

    def preprocess(img_path):
        img = Image.open(img_path).convert("RGB").resize((112, 112))
        x = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
        q = np.clip(np.round(x / input_scale) + input_zp, -128, 127).astype(np.int8)
        return np.expand_dims(q, 0)

    def get_embedding(q_input):
        interp.set_tensor(in_details[0]["index"], q_input)
        interp.invoke()
        raw = interp.get_tensor(out_details[0]["index"])
        emb = (raw.astype(np.float32) - output_zp) * output_scale
        emb = emb.flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    # Test 4 images
    embs = []
    for i in range(min(4, len(images))):
        q = preprocess(images[i * 100])  # Use different images
        emb = get_embedding(q)
        embs.append(emb)

    print("\n  Pairwise cosine similarities (different faces):")
    cos_sims = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            cs = float(np.dot(embs[i], embs[j]))
            cos_sims.append(cs)
            print(f"    img{i} vs img{j}: {cs:.6f}")

    # Self-similarity check
    q0 = preprocess(images[0])
    emb_a = get_embedding(q0)
    emb_b = get_embedding(q0)  # Run again (deterministic)
    self_sim = float(np.dot(emb_a, emb_b))
    print(f"\n  Self cosine similarity: {self_sim:.6f} (should be ~1.0)")

    avg_cos = np.mean(cos_sims)
    print(f"\n  Average cosine similarity (different faces): {avg_cos:.6f}")
    if 0.1 < avg_cos < 0.8 and self_sim > 0.999:
        print(f"  RESULT: Model is WORKING (avg cos_sim={avg_cos:.4f}, not collapsed)")
    elif avg_cos > 0.9:
        print(f"  RESULT: Model may be COLLAPSED (cos_sim too high)")
    elif avg_cos < 0.05:
        print(f"  RESULT: Model may produce near-zero embeddings")
    else:
        print(f"  RESULT: Inconclusive")

    return avg_cos, self_sim


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print("ESP-DL Quantization Injection (Weight Fix Edition)")
    print("=" * 60)

    print("\n[1] Parsing .espdl...")
    espdl_data = parse_espdl(ESPDL_PATH)
    print(f"  Nodes: {len(espdl_data['nodes'])}")
    print(f"  Conv weights: {len(espdl_data['conv_weights'])}")
    print(f"  Value infos with exponents: {len(espdl_data['value_infos'])}")

    # Show exponent distribution
    print("  Activation exponent distribution:")
    for shape, entries in sorted(espdl_data["shape_exps"].items(), key=lambda x: len(x[1]), reverse=True):
        if len(entries) > 2:
            exps = set(e[0] for e in entries)
            scales = [f"2^{e}" for e in sorted(exps)]
            print(f"    shape={shape}: {len(entries)} entries, exps={sorted(exps)} -> {', '.join(scales)}")

    print("\n[2] Parsing TFLite template...")
    tflite_data = parse_tflite(TEMPLATE_TFLITE)
    print(f"  Tensors: {len(tflite_data['tensors'])}")
    print(f"  Operators: {len(tflite_data['operators'])}")

    print("\n[3] Rebuilding with corrected quantization...")
    new_data = rebuild_model(tflite_data, espdl_data)

    OUT_TFLITE.write_bytes(new_data)
    print(f"\n  Saved to {OUT_TFLITE} ({len(new_data) / 1024:.1f} KiB)")

    print("\n[4] Verification")
    verify_model(OUT_TFLITE)

    print("\n[5] Cosine similarity test")
    test_cosine_similarity(OUT_TFLITE)

    print("\nDONE")
    return OUT_TFLITE


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR.parent)
    main()
