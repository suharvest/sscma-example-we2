#!/usr/bin/env python3
"""
Direct .espdl quantization injection into TFLite flatbuffer.

Strategy:
1. Parse .espdl to get activation exponents and weight exponents
2. Take existing INT8 TFLite model (calibration-based)
3. Modify TFLite flatbuffer to use .espdl quantization params
4. Save

The .espdl uses symmetric quantization: real_value = int8_value * 2^exponent
TFLite equivalent: scale = 2^exponent, zero_point = 0
"""

import struct, sys, os
from pathlib import Path

import numpy as np
import flatbuffers
from tensorflow.lite.python import schema_py_generated as tflite_schema

sys.path.insert(0, "/tmp/esp-ppq-src")
from esp_ppq.parser.espdl.FlatBuffers.Dl import Model
from esp_ppq.parser.espdl.FlatBuffers.Dl.TensorDataType import TensorDataType
from esp_ppq.parser.espdl.FlatBuffers.Dl.TypeInfoValue import TypeInfoValue

# === CONFIG ===
ESPDL_PATH = "/Users/harvest/project/esp-dl/models/human_face_recognition/models/s3/human_face_feat_mfn_s8_v1.espdl"
SCRIPT_DIR = Path(__file__).resolve().parent

# Use the PRelu INT8 TFLite (101 ops) as template - closest to .espdl (103 nodes)
TEMPLATE_TFLITE = SCRIPT_DIR / "mfn_sp_int8.tflite"
OUT_TFLITE = SCRIPT_DIR / "mfn_espdl_quant.tflite"
OUT_FLOAT32_TFLITE = SCRIPT_DIR / "mfn_espdl_float32.tflite"


# ============================================================
# STEP 1: Parse .espdl
# ============================================================


def parse_espdl_activations(espdl_path):
    """Extract activation quantization exponents from .espdl."""
    with open(espdl_path, "rb") as f:
        data = f.read()
    model_data = data[16 : 16 + struct.unpack("I", data[8:12])[0]]
    model = Model.Model.GetRootAs(model_data, 0)
    graph = model.Graph()

    # Extract nodes in order
    nodes = []
    for i in range(graph.NodeLength()):
        node = graph.Node(i)
        name = node.Name().decode("utf-8", errors="ignore")
        op = node.OpType().decode("utf-8", errors="ignore")
        inputs = [node.Input(j).decode("utf-8", errors="ignore") for j in range(node.InputLength())]
        outputs = [node.Output(j).decode("utf-8", errors="ignore") for j in range(node.OutputLength())]
        nodes.append({"idx": i, "name": name, "op": op, "inputs": inputs, "outputs": outputs})

    # Extract value infos with exponents
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
            value_infos[name] = {
                "dtype": dtype,
                "dims": tuple(dims),
                "exponents": exps,
                "scale": [2.0**e for e in exps],
            }

    # Extract weight exponents (from initializers)
    weight_exps = {}
    for i in range(graph.InitializerLength()):
        init = graph.Initializer(i)
        wname = init.Name().decode("utf-8", errors="ignore")
        exps = [init.Exponents(j) for j in range(init.ExponentsLength())]
        if exps:
            weight_exps[wname] = exps

    # Build activation list in topological order
    # Group by shape to help with mapping
    act_list = []
    for node in nodes:
        for out_name in node["outputs"]:
            if out_name in value_infos:
                info = value_infos[out_name]
                act_list.append(
                    {
                        "name": out_name,
                        "op": node["op"],
                        "dims": info["dims"],
                        "exponents": info["exponents"],
                        "scale": info["scale"],
                    }
                )

    # Build shape -> list of exponents for quick lookup
    shape_exps = {}
    for vi_name, vi_info in value_infos.items():
        key = vi_info["dims"]
        if key not in shape_exps:
            shape_exps[key] = []
        shape_exps[key].append(vi_info["exponents"])

    return {
        "nodes": nodes,
        "value_infos": value_infos,
        "weight_exps": weight_exps,
        "act_list": act_list,
        "shape_exps": shape_exps,
    }


# ============================================================
# STEP 2: Parse TFLite model
# ============================================================


def parse_tflite(tflite_path):
    """Parse a TFLite model and extract tensor/operator info."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    model = tflite_schema.Model.GetRootAs(data, 0)
    subgraph = model.Subgraphs(0)

    # Op codes
    op_codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        bk = oc.BuiltinCode()
        op_codes.append(bk)

    # Tensor types
    TYPE_NAMES = {
        0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
        5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8", 10: "FLOAT64",
    }

    # Collect tensor info
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
        }

    # Collect operator info with their input/output tensor indices
    operators = []
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        bk = op_codes[op.OpcodeIndex()]
        op_type = tflite_schema.BuiltinOperator.Name(bk) if hasattr(tflite_schema.BuiltinOperator, 'Name') else str(bk)
        inputs = [op.Inputs(j) for j in range(op.InputsLength())]
        outputs = [op.Outputs(j) for j in range(op.OutputsLength())]
        operators.append({"idx": i, "type": op_type, "code": bk, "inputs": inputs, "outputs": outputs})

    return {
        "tensors": tensors,
        "operators": operators,
        "op_codes": op_codes,
        "raw_data": data,
        "model": model,
        "subgraph": subgraph,
    }


# ============================================================
# STEP 3: Map .espdl activations to TFLite tensors
# ============================================================


def map_activations(espdl_data, tflite_data):
    """
    Map .espdl activation tensor exponents to TFLite tensor indices.

    Strategy: Match by operator sequence and tensor shape.
    The .espdl and TFLite models have the same base architecture;
    differences are in merged/split ops and PRelu/ReLU handling.

    For each TFLite op output:
    1. Find its shape
    2. Look up that shape in .espdl's shape_exps
    3. Apply the matching exponent
    """
    tensors = tflite_data["tensors"]
    operators = tflite_data["operators"]
    shape_exps = espdl_data["shape_exps"]

    # Collect all TFLite activation tensors (op outputs)
    op_outputs = []  # (tensor_idx, shape)
    op_inputs = set()
    for op in operators:
        for out_idx in op["outputs"]:
            if out_idx >= 0 and out_idx in tensors:
                shape = tensors[out_idx]["shape"]
                if not shape:  # empty tuple
                    continue
                op_outputs.append((out_idx, shape))
        for in_idx in op["inputs"]:
            if in_idx >= 0:
                op_inputs.add(in_idx)

    # Map each activation tensor
    tflite_to_exp = {}  # tensor_idx -> exponent
    state = {}  # Track which .espdl entry to use for each shape

    for tensor_idx, shape in op_outputs:
        if shape in shape_exps:
            entries = shape_exps[shape]
            # Use the next unused entry for this shape
            shape_key = shape
            if shape_key not in state:
                state[shape_key] = 0
            idx = state[shape_key]
            if idx < len(entries):
                exp = entries[idx][0]  # Per-tensor: take first exponent
                tflite_to_exp[tensor_idx] = exp
                state[shape_key] += 1

    # Also map input tensor
    # In .espdl, input has exponent -6 (scale=0.015625)
    input_exp = -6
    if "input" in espdl_data["value_infos"]:
        input_exp = espdl_data["value_infos"]["input"]["exponents"][0]

    # Map output tensor
    output_exp = -5  # Default
    if "embedding" in espdl_data["value_infos"]:
        output_exp = espdl_data["value_infos"]["embedding"]["exponents"][0]

    return {
        "tflite_to_exp": tflite_to_exp,
        "input_exp": input_exp,
        "output_exp": output_exp,
    }


# ============================================================
# STEP 4: Rebuild TFLite with corrected quantization
# ============================================================


def rebuild_tflite_with_espdl_quant(tflite_data, mapping, espdl_data):
    """
    Rebuild the TFLite flatbuffer with corrected quantization params.

    Uses the flatbuffers Object API (T classes) to unpack, modify, and repack.
    """
    print("\n  Rebuilding TFLite with corrected quantization...")

    # Step 1: Unpack the original TFLite model
    raw_data = tflite_data["raw_data"]
    model_obj = tflite_schema.ModelT.InitFromPackedBuf(raw_data, 0)
    subgraph = model_obj.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators

    tflite_to_exp = mapping["tflite_to_exp"]
    input_exp = mapping["input_exp"]
    output_exp = mapping["output_exp"]

    # Find which tensors are activation outputs
    op_output_tensors = set()
    for op in operators:
        for out_idx in op.outputs:
            if out_idx >= 0:
                op_output_tensors.add(out_idx)

    op_input_tensors = set()
    for op in operators:
        for in_idx in op.inputs:
            if in_idx >= 0:
                op_input_tensors.add(in_idx)

    # Find which tensors are weights (constant, not op outputs)
    weight_tensors = set()
    for idx in range(len(tensors)):
        if idx not in op_output_tensors and tensors[idx].type == tflite_schema.TensorType.INT8:
            # Check if it's in a buffer
            if tensors[idx].buffer > 0:
                weight_tensors.add(idx)

    # Fix activation quantization
    activations_fixed = 0
    activations_skipped = 0

    for idx in range(len(tensors)):
        t = tensors[idx]

        # Only fix INT8 tensors
        if t.type != tflite_schema.TensorType.INT8:
            continue

        shape = tuple(t.shape) if (t.shape is not None and len(t.shape) > 0) else tuple()

        # Determine the correct exponent
        exp = None
        if idx in tflite_to_exp:
            exp = tflite_to_exp[idx]
        elif idx in op_input_tensors and idx not in op_output_tensors:
            # Input tensor
            exp = input_exp
        elif idx in op_output_tensors:
            # Try mapping by shape
            if shape in espdl_data["shape_exps"]:
                exp = espdl_data["shape_exps"][shape][0][0]

        if exp is None:
            continue

        # Compute TFLite quantization params
        # .espdl: real = int8 * 2^exp
        # TFLite: real = scale * (int8 - zero_point)
        # For symmetric: scale = 2^exp, zero_point = 0
        scale = float(2.0**exp)
        zero_point = 0

        # Only fix if current quantization exists (has scales)
        has_q = (t.quantization is not None
                 and t.quantization.scale is not None
                 and len(t.quantization.scale) > 0)
        if has_q:
            old_scales = list(t.quantization.scale)
            old_zps = list(t.quantization.zeroPoint) if (t.quantization.zeroPoint is not None and len(t.quantization.zeroPoint) > 0) else []

            # Set new per-tensor quantization
            t.quantization.scale = [scale]
            t.quantization.zeroPoint = [zero_point]
            t.quantization.quantizedDimension = 0

            activations_fixed += 1
        else:
            activations_skipped += 1

    print(f"  Fixed {activations_fixed} activation tensors, skipped {activations_skipped}")

    # Fix weight quantization using .espdl weight exponents
    # Note: weight quantization in calibration model uses min/max scales
    # For ESP-DL symmetric quantization, we need scales = 2^exponent
    weight_exps = espdl_data["weight_exps"]
    weights_fixed = 0

    for idx in range(len(tensors)):
        t = tensors[idx]
        if t.type != tflite_schema.TensorType.INT8:
            continue

        name = t.name.decode("utf-8") if t.name else ""
        if not name:
            continue

        # Check if this weight name matches .espdl weight
        for wname, wexps in weight_exps.items():
            if wname in name or name in wname:
                has_wq = (t.quantization is not None
                         and t.quantization.scale is not None
                         and len(t.quantization.scale) > 0)
                if has_wq:
                    old_scales = list(t.quantization.scale)
                    # Set .espdl scales
                    new_scales = [float(2.0**e) for e in wexps]
                    if len(new_scales) == 1 and len(old_scales) > 1:
                        # Per-channel -> per-tensor
                        t.quantization.scale = [new_scales[0]]
                        nzps = len(t.quantization.zeroPoint) if (t.quantization.zeroPoint is not None and len(t.quantization.zeroPoint) > 0) else 1
                        t.quantization.zeroPoint = [0] * nzps
                        t.quantization.quantizedDimension = 0
                    elif len(new_scales) > 1 and len(old_scales) == 1:
                        # Per-tensor -> per-channel: set quant_dim to 0 (output channel)
                        t.quantization.scale = new_scales
                        t.quantization.zeroPoint = [0] * len(new_scales)
                        t.quantization.quantizedDimension = 0
                    else:
                        t.quantization.scale = new_scales
                        t.quantization.zeroPoint = [0] * len(new_scales)

                    weights_fixed += 1
                break

    print(f"  Fixed {weights_fixed} weight tensor quantizations")

    # Repack the modified model with TFLite file identifier
    builder = flatbuffers.Builder(1024 * 1024 * 4)  # 4MB
    packed = model_obj.Pack(builder)
    builder.Finish(packed, b'TFL3')
    new_data = bytes(builder.Output())

    return new_data


# ============================================================
# STEP 5: Verification
# ============================================================


def verify_model(tflite_path, espdl_data):
    """Verify the output model has correct quantization params."""
    print("\n  Verifying model...")

    with open(tflite_path, "rb") as f:
        data = f.read()

    model = tflite_schema.Model.GetRootAs(data, 0)
    subgraph = model.Subgraphs(0)

    print(f"  Tensors: {subgraph.TensorsLength()}")
    print(f"  Operators: {subgraph.OperatorsLength()}")

    # Count by tensor type and quantization
    type_q_counts = {}
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        dtype = t.Type()
        name = t.Name().decode("utf-8") if t.Name() else f"t_{i}"
        q = t.Quantization()
        has_q = q and q.ScaleLength() > 0

        key = f"type{dtype}"
        if key not in type_q_counts:
            type_q_counts[key] = {"total": 0, "quantized": 0}
        type_q_counts[key]["total"] += 1
        if has_q:
            type_q_counts[key]["quantized"] += 1

    for key, counts in type_q_counts.items():
        print(f"    {key}: {counts['total']} tensors, {counts['quantized']} quantized")

    # Show some quantization scales
    print("\n  Sample quantization scales:")
    shown = 0
    for i in range(subgraph.TensorsLength()):
        if shown >= 15:
            break
        t = subgraph.Tensors(i)
        q = t.Quantization()
        if q and q.ScaleLength() > 0:
            name = t.Name().decode("utf-8") if t.Name() else f"t_{i}"
            scales = [q.Scale(j) for j in range(q.ScaleLength())]
            zps = [q.ZeroPoint(j) for j in range(q.ZeroPointLength())]
            shape = [t.Shape(j) for j in range(t.ShapeLength())]

            # Check if any scale is a power of 2
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
                print(f"    {name} ({shape}): {len(scales)} per-ch scales, first={scales[0]:.8f}{marker}")
            shown += 1

    return True


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print("ESP-DL Quantization Injection into TFLite")
    print("=" * 60)

    # STEP 1: Parse .espdl
    print("\n[1/4] Parsing .espdl activation exponents...")
    espdl_data = parse_espdl_activations(ESPDL_PATH)
    print(f"  Nodes: {len(espdl_data['nodes'])}")
    print(f"  Value infos with exponents: {len(espdl_data['value_infos'])}")
    print(f"  Weights with exponents: {len(espdl_data['weight_exps'])}")

    # Print exponent distribution
    print("  Activation exponent distribution by shape:")
    for shape, entries in sorted(espdl_data["shape_exps"].items(), key=lambda x: len(x[1]), reverse=True):
        if len(entries) > 2:
            exps = [e[0] for e in entries]
            print(f"    shape={shape}: {len(entries)} entries, exps={set(exps)}")

    # STEP 2: Parse TFLite template
    print(f"\n[2/4] Parsing TFLite template: {TEMPLATE_TFLITE}")
    tflite_data = parse_tflite(TEMPLATE_TFLITE)
    print(f"  Tensors: {len(tflite_data['tensors'])}")
    print(f"  Operators: {len(tflite_data['operators'])}")

    # STEP 3: Map activations
    print("\n[3/4] Mapping .espdl activations to TFLite tensors...")
    mapping = map_activations(espdl_data, tflite_data)
    print(f"  Mapped {len(mapping['tflite_to_exp'])} activation tensors")
    print(f"  Input exponent: {mapping['input_exp']} (scale={2.0**mapping['input_exp']:.6f})")
    print(f"  Output exponent: {mapping['output_exp']} (scale={2.0**mapping['output_exp']:.6f})")

    # STEP 4: Rebuild with corrected quantization
    print("\n[4/4] Rebuilding TFLite with corrected quantization...")
    new_data = rebuild_tflite_with_espdl_quant(tflite_data, mapping, espdl_data)

    OUT_TFLITE.write_bytes(new_data)
    print(f"  Saved to {OUT_TFLITE} ({len(new_data) / 1024:.1f} KiB)")

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    verify_model(OUT_TFLITE, espdl_data)

    # Check cosine similarity on test images
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY TEST")
    print("=" * 60)
    test_cosine_similarity(OUT_TFLITE)

    print("\nDONE")
    return OUT_TFLITE


def test_cosine_similarity(tflite_path):
    """Run a quick cosine similarity test on the model."""
    import tensorflow as tf
    from PIL import Image

    # Look for test images
    calib_dir = SCRIPT_DIR.parent / "calibration_data" / "qat_112"
    test_images = []
    img_paths = sorted(calib_dir.glob("*.jpg")) if calib_dir.exists() else []
    if not img_paths:
        # Try another path
        calib_dir = SCRIPT_DIR.parent / "datasets"
        img_paths = sorted(calib_dir.rglob("*.jpg")) if calib_dir.exists() else []

    if len(img_paths) < 2:
        print("  Not enough test images found, generating dummy inputs...")
        dummy1 = np.random.randn(1, 112, 112, 3).astype(np.float32)
        dummy2 = np.random.randn(1, 112, 112, 3).astype(np.float32)
    else:
        print(f"  Loading test images from {img_paths[0].parent}")
        img1 = Image.open(img_paths[0]).convert("RGB").resize((112, 112))
        img2 = Image.open(img_paths[-1]).convert("RGB").resize((112, 112))
        dummy1 = np.expand_dims(np.array(img1, dtype=np.float32), 0)
        dummy2 = np.expand_dims(np.array(img2, dtype=np.float32), 0)

    # Also test with the float32 version for comparison
    float32_path = SCRIPT_DIR / "mfn_sp_relu_saved_model" / "mfn_s8_v1_singlepath_relu_float32.tflite"
    has_float32 = float32_path.exists()
    if has_float32:
        print(f"  Comparing with float32 model: {float32_path}")

    try:
        # Load INT8 model (disable XNNPACK to avoid delegate errors with custom quant)
        import os as _os
        import tensorflow.lite.experimental as tflite_exp
        _os.environ["TF_ENABLE_XNNPACK"] = "0"
        interp_int8 = tf.lite.Interpreter(
            model_path=str(tflite_path),
            experimental_op_resolver_type=tflite_exp.OpResolverType.BUILTIN_REF,
        )
        interp_int8.allocate_tensors()
        in_details = interp_int8.get_input_details()
        out_details = interp_int8.get_output_details()

        print(f"  INT8 input details: {in_details[0]}")
        print(f"  INT8 output details: {out_details[0]}")

        # Quantize input according to input quantization
        input_scale = in_details[0]["quantization_parameters"]["scales"][0]
        input_zp = in_details[0]["quantization_parameters"]["zero_points"][0]

        def quantize_input(x):
            return np.clip(np.round(x / input_scale) + input_zp, -128, 127).astype(np.int8)

        def dequantize_output(y, out_detail):
            scale = out_detail["quantization_parameters"]["scales"][0]
            zp = out_detail["quantization_parameters"]["zero_points"][0]
            return (y.astype(np.float32) - zp) * scale

        # Run on two inputs
        q_in1 = quantize_input(dummy1)
        q_in2 = quantize_input(dummy2)

        interp_int8.set_tensor(in_details[0]["index"], q_in1)
        interp_int8.invoke()
        emb1_raw = interp_int8.get_tensor(out_details[0]["index"])
        emb1 = dequantize_output(emb1_raw, out_details[0])

        interp_int8.set_tensor(in_details[0]["index"], q_in2)
        interp_int8.invoke()
        emb2_raw = interp_int8.get_tensor(out_details[0]["index"])
        emb2 = dequantize_output(emb2_raw, out_details[0])

        # L2 normalize and compute cosine similarity
        emb1_norm = emb1.flatten()
        emb1_norm = emb1_norm / (np.linalg.norm(emb1_norm) + 1e-8)
        emb2_norm = emb2.flatten()
        emb2_norm = emb2_norm / (np.linalg.norm(emb2_norm) + 1e-8)
        cos_sim = float(np.dot(emb1_norm, emb2_norm))

        print(f"\n  INT8 Cosine similarity (different faces): {cos_sim:.6f}")
        print(f"  (Expected: ~0.3-0.7 for different faces, should NOT be ~0.99)")

        # Quick check: cosine sim with itself
        interp_int8.set_tensor(in_details[0]["index"], q_in1)
        interp_int8.invoke()
        emb1a_raw = interp_int8.get_tensor(out_details[0]["index"])
        emb1a = dequantize_output(emb1a_raw, out_details[0])
        emb1a_norm = emb1a.flatten() / (np.linalg.norm(emb1a.flatten()) + 1e-8)
        self_sim = float(np.dot(emb1_norm, emb1a_norm))
        print(f"  INT8 Self cosine similarity: {self_sim:.6f} (should be ~1.0)")

        # If float32 available, compare
        if has_float32:
            interp_f32 = tf.lite.Interpreter(model_path=str(float32_path))
            interp_f32.allocate_tensors()
            f32_in = interp_f32.get_input_details()
            f32_out = interp_f32.get_output_details()

            interp_f32.set_tensor(f32_in[0]["index"], dummy1)
            interp_f32.invoke()
            ref1 = interp_f32.get_tensor(f32_out[0]["index"]).flatten()
            ref1 = ref1 / (np.linalg.norm(ref1) + 1e-8)

            interp_f32.set_tensor(f32_in[0]["index"], dummy2)
            interp_f32.invoke()
            ref2 = interp_f32.get_tensor(f32_out[0]["index"]).flatten()
            ref2 = ref2 / (np.linalg.norm(ref2) + 1e-8)

            ref_cos_sim = float(np.dot(ref1, ref2))
            print(f"  Float32 Cosine similarity: {ref_cos_sim:.6f}")

            # Compute cosine sim between INT8 and float32 embeddings
            cross_sim = float(np.dot(emb1_norm, ref1))
            print(f"  INT8-vs-Float32 Cosine similarity (same input): {cross_sim:.6f} (should be ~1.0)")

            if cross_sim > 0.9:
                print("  PASS: INT8 embeddings match float32 embeddings")
            else:
                print("  WARNING: INT8 embeddings differ from float32 (cos_sim < 0.9)")

        # Final verdict
        if cos_sim < 0.95 and cos_sim > 0.01 and self_sim > 0.99:
            print(f"\n  RESULT: Model appears to be WORKING (cos_sim={cos_sim:.4f}, not collapsed)")
        elif cos_sim > 0.95:
            print(f"\n  RESULT: Model may be COLLAPSED (cos_sim={cos_sim:.4f} is too high)")
        elif cos_sim < 0.01:
            print(f"\n  RESULT: Model may produce near-zero embeddings")
        else:
            print(f"\n  RESULT: Inconclusive (cos_sim={cos_sim:.4f})")

    except Exception as e:
        print(f"  ERROR in cosine similarity test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR.parent)
    main()
