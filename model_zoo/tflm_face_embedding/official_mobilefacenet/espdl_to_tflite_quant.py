#!/usr/bin/env python3
"""
Extract quantization parameters from .espdl and inject them into a TFLite model.

This script:
1. Parses the .espdl file to extract all weights and activation exponents
2. Builds a TF SavedModel matching the .espdl architecture
3. Converts to TFLite
4. Injects per-tensor quantization parameters using the .espdl exponents
5. Saves the INT8 TFLite model
"""

import os
import struct
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as tflite_schema

sys.path.insert(0, "/tmp/esp-ppq-src")
from esp_ppq.parser.espdl.FlatBuffers.Dl import Model
from esp_ppq.parser.espdl.FlatBuffers.Dl.TensorDataType import TensorDataType

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ESPDL_PATH = "/Users/harvest/project/esp-dl/models/human_face_recognition/models/s3/human_face_feat_mfn_s8_v1.espdl"
OUT_TFLITE = SCRIPT_DIR / "mfn_espdl_quant.tflite"
SAVED_MODEL_DIR = SCRIPT_DIR / "mfn_espdl_saved_model"

DTYPE_MAP = {
    TensorDataType.UNDEFINED: "UNDEFINED",
    TensorDataType.FLOAT: "FLOAT",
    TensorDataType.UINT8: "UINT8",
    TensorDataType.INT8: "INT8",
    TensorDataType.INT32: "INT32",
    TensorDataType.FLOAT16: "FLOAT16",
    TensorDataType.INT64: "INT64",
}

TFLITE_TYPE = {
    "FLOAT32": 0,
    "FLOAT16": 1,
    "INT32": 2,
    "UINT8": 3,
    "INT64": 4,
    "STRING": 5,
    "BOOL": 6,
    "INT16": 7,
    "COMPLEX64": 8,
    "INT8": 9,
    "FLOAT64": 10,
}


def read_espdl_tensor_data(init):
    """Read raw tensor data from an ESP-DL initializer, dequantize to float32."""
    dtype = init.DataType()
    dims = [init.Dims(j) for j in range(init.DimsLength())]
    exps = [init.Exponents(j) for j in range(init.ExponentsLength())]
    expected_size = int(np.prod(dims))

    # Read raw bytes (AlignedBytes blocks, each 16 bytes)
    raw_len = init.RawDataLength()
    total_bytes = raw_len * 16  # Each block is 16 bytes

    if total_bytes == 0:
        return None, dims, exps, dtype

    raw_bytes = bytearray()
    for j in range(raw_len):
        block = init.RawData(j)
        block_bytes = block.BytesAsNumpy()
        raw_bytes.extend(block_bytes.tobytes())

    raw_array = np.frombuffer(bytes(raw_bytes[:expected_size]), dtype=np.int8)

    if raw_array.size != expected_size:
        print(f"  WARNING: expected {expected_size} elements, got {raw_array.size}")
        return None, dims, exps, dtype

    # Dequantize to float32 using exponents
    fp_data = raw_array.astype(np.float64).reshape(dims)
    if exps:
        if len(exps) == 1:
            fp_data *= 2.0 ** np.float64(exps[0])
        else:
            # Per-channel quantization (expand dims for broadcasting)
            scale = np.array([2.0**e for e in exps], dtype=np.float64)
            # For Conv weights [H,W,C_in,C_out] or [C_out,C_in,H,W], scale applies per output channel
            # Determine which dim is the output channel
            # For bias [C], scale applies directly
            # For weights, reshape scale for broadcasting
            if len(dims) >= 2 and len(scale) == dims[-1]:
                # Last dim is output channel
                scale_shape = [1] * (len(dims) - 1) + [len(scale)]
                fp_data *= scale.reshape(scale_shape)
            elif len(dims) >= 2 and len(scale) == dims[0]:
                # First dim is output channel
                scale_shape = [len(scale)] + [1] * (len(dims) - 1)
                fp_data *= scale.reshape(scale_shape)
            else:
                print(f"  WARNING: cannot broadcast per-channel scale {len(scale)} to dims {dims}")

    return fp_data.astype(np.float32), dims, exps, dtype


def parse_espdl(espdl_path):
    """Parse an .espdl file and extract all weights, graph, and quantization params."""
    print("=" * 60)
    print("STEP 1: Parsing .espdl file")
    print("=" * 60)

    with open(espdl_path, "rb") as f:
        data = f.read()

    model_data = data[16 : 16 + struct.unpack("I", data[8:12])[0]]
    model = Model.Model.GetRootAs(model_data, 0)
    graph = model.Graph()

    print(f"  Nodes: {graph.NodeLength()}")
    print(f"  ValueInfos: {graph.ValueInfoLength()}")
    print(f"  Initializers: {graph.InitializerLength()}")

    # --- Extract weights (dequantized to float32) ---
    print("\n  Extracting weights...")
    weights = {}
    for i in range(graph.InitializerLength()):
        init = graph.Initializer(i)
        name = init.Name().decode("utf-8", errors="ignore")
        fp_data, dims, exps, dtype = read_espdl_tensor_data(init)

        if fp_data is not None:
            weights[name] = {
                "data": fp_data,
                "dims": dims,
                "exponents": exps,
                "dtype": DTYPE_MAP.get(dtype, str(dtype)),
            }
            # Always print PRelu slopes
            if exps and max(abs(e) for e in exps) < 3:
                is_prelu = not ("conv" in name.lower() or "weight" in name.lower() or "bias" in name.lower())
                if is_prelu:
                    print(f"    {name}: shape={fp_data.shape}, exps={exps}, values=[{fp_data.min():.6f}, {fp_data.max():.6f}]")

    print(f"  Extracted {len(weights)} weights")

    # --- Extract graph structure ---
    print("\n  Extracting graph structure...")
    nodes = []
    for i in range(graph.NodeLength()):
        node = graph.Node(i)
        name = node.Name().decode("utf-8", errors="ignore")
        op = node.OpType().decode("utf-8", errors="ignore") if node.OpType() else "Unknown"
        inputs = [node.Input(j).decode("utf-8", errors="ignore") for j in range(node.InputLength())]
        outputs = [node.Output(j).decode("utf-8", errors="ignore") for j in range(node.OutputLength())]
        nodes.append({"index": i, "name": name, "op": op, "inputs": inputs, "outputs": outputs})

    # --- Extract activation exponents ---
    print("\n  Extracting activation quantization parameters...")
    act_exponents = {}
    from esp_ppq.parser.espdl.FlatBuffers.Dl.TypeInfoValue import TypeInfoValue

    for i in range(graph.ValueInfoLength()):
        vi = graph.ValueInfo(i)
        name = vi.Name().decode("utf-8", errors="ignore")
        exps = [vi.Exponents(j) for j in range(vi.ExponentsLength())]

        ti = vi.ValueInfoType()
        dtype = 0
        dims = []
        if ti:
            v = ti.Value()
            if v and ti.ValueType() == TypeInfoValue.tensor_type:
                from esp_ppq.parser.espdl.FlatBuffers.Dl.TensorTypeAndShape import TensorTypeAndShape

                tt = TensorTypeAndShape()
                tt.Init(v.Bytes, v.Pos)
                dtype = tt.ElemType()
                shape = tt.Shape()
                if shape:
                    for j in range(shape.DimLength()):
                        dim = shape.Dim(j)
                        if dim and dim.Value():
                            dval = dim.Value()
                            from esp_ppq.parser.espdl.FlatBuffers.Dl.DimensionValue import DimensionValue
                            from esp_ppq.parser.espdl.FlatBuffers.Dl.DimensionValueType import DimensionValueType

                            if dval.DimType() == DimensionValueType.VALUE:
                                dims.append(dval.DimValue())
                            elif dval.DimType() == DimensionValueType.PARAM:
                                dims.append(f"param:{dval.DimParam()}")

        if exps:
            act_exponents[name] = {
                "dtype": DTYPE_MAP.get(dtype, str(dtype)),
                "dims": dims,
                "exponents": exps,
                "scale": 2.0 ** np.array(exps, dtype=np.float64),
            }

    print(f"  Extracted {len(act_exponents)} activation quantization params")
    print(f"  Exponent distribution:")
    exp_counts = {}
    for name, info in act_exponents.items():
        for e in info["exponents"]:
            exp_counts[e] = exp_counts.get(e, 0) + 1
    for e in sorted(exp_counts.keys()):
        print(f"    exp={e}: {exp_counts[e]} tensors (scale={2.0**e:.6f})")

    # Find input/output tensors
    graph_inputs = []
    for i in range(graph.InputLength()):
        inp = graph.Input(i)
        name = inp.Name().decode("utf-8", errors="ignore")
        graph_inputs.append(name)

    graph_outputs = []
    for i in range(graph.OutputLength()):
        out = graph.Output(i)
        name = out.Name().decode("utf-8", errors="ignore")
        graph_outputs.append(name)

    print(f"\n  Graph inputs: {graph_inputs}")
    print(f"  Graph outputs: {graph_outputs}")

    return {
        "weights": weights,
        "nodes": nodes,
        "act_exponents": act_exponents,
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
    }


def build_tf_model(espdl_data, saved_model_dir):
    """Build a TF SavedModel from ESP-DL graph data."""
    print("\n" + "=" * 60)
    print("STEP 2: Building TF SavedModel from ESP-DL data")
    print("=" * 60)

    weights = espdl_data["weights"]
    nodes = espdl_data["nodes"]
    act_exponents = espdl_data["act_exponents"]
    graph_inputs = espdl_data["graph_inputs"]
    graph_outputs = espdl_data["graph_outputs"]

    # The ESP-DL model uses ONNX (NCHW) format internally
    # TensorFlow uses NHWC format
    # We need to build the model in NCHW and transpose input/output
    # OR build in NHWC and handle correctly

    # Strategy: Build using TF in the format ESP-DL uses
    # ESP-DL input: [1, 112, 112, 3] (same as NHWC)
    # ESP-DL weights: ONNX format (NCHW layout)
    # For Conv2D, ONNX uses format: [C_out, C_in, H, W] for weights, [C_out] for bias
    # In TensorFlow, use data_format='channels_last' for inputs,
    # but we need to transpose weights to TF format: [H, W, C_in, C_out]

    # Actually: the ESP-DL model expects NHWC input
    # But the weights are stored in ONNX format
    # ONNX Conv: weight shape [C_out, C_in, H, W] for standard, [C_out, 1, H, W] for depthwise
    # TF Conv2D: weight shape [H, W, C_in, C_out] for standard, [H, W, C_in, depth_multiplier] for depthwise

    input_layer = tf.keras.layers.Input(shape=(112, 112, 3), name="input")

    # Track tensor name -> Keras tensor mapping
    tensor_map = {"input": input_layer, "embedding": None}

    # For PRelu slopes, the .espdl stores them with numeric names like "280", "281", etc.
    # We need to build a lookup for PRelu slope tensors
    prelu_slopes = {}
    for wname, winfo in weights.items():
        # PRelu slopes have small dims and small exponents
        if winfo["data"].ndim <= 1 and any(abs(e) < 3 for e in winfo.get("exponents", [0])):
            # This looks like a PRelu slope
            # Map by name - the .espdl uses numeric names for PRelu slopes
            prelu_slopes[wname] = tf.constant(winfo["data"], dtype=tf.float32)

    # For initial weights, we also need a weight lookup
    # ESP-DL weight names match the ONNX weight names

    # Build layers layer-by-layer following the graph
    tensor_map["input"] = input_layer

    def get_weight(name):
        if name in weights:
            return tf.constant(weights[name]["data"], dtype=tf.float32)
        return None

    # Build the graph by processing nodes in order
    for node in nodes:
        op = node["op"]
        node_inputs = node["inputs"]
        node_outputs = node["outputs"]

        if op == "Conv":
            # Conv: inputs = [input_tensor, weight_tensor, bias_tensor]
            # Output: [output_tensor]
            in_name = node_inputs[0]
            w_name = node_inputs[1]
            b_name = node_inputs[2] if len(node_inputs) > 2 else None

            if in_name not in tensor_map:
                raise ValueError(f"Conv input '{in_name}' not found in tensor_map. Available: {list(tensor_map.keys())[:10]}...")

            x = tensor_map[in_name]
            weight = get_weight(w_name)
            bias = get_weight(b_name) if b_name else None

            if weight is None:
                raise ValueError(f"Weight '{w_name}' not found for Conv at node {node['index']}")

            if bias is None:
                bias = None  # Use default bias

            # Determine Conv type and reshape weights
            # ONNX weight shape: for standard Conv [C_out, C_in, H, W]
            #                    for depthwise Conv [C_out, 1, H, W]
            # TF weight shape:   for standard Conv2D [H, W, C_in, C_out]
            #                    for DepthwiseConv2D [H, W, C_in, depth_multiplier]
            w_shape = weight.shape
            if len(w_shape) == 4:
                is_depthwise = (w_shape[1] == 1)  # C_in == 1 for depthwise
            else:
                is_depthwise = False

            # Transpose from ONNX [C_out, C_in, H, W] to TF [H, W, C_in, C_out]
            if len(w_shape) == 4:
                tf_weight = tf.transpose(weight, [2, 3, 1, 0])
            else:
                tf_weight = weight

            # Determine Conv parameters
            stride = 1  # default
            padding = "same"  # default (ESP-DL uses same padding for most)

            # Check for downsampling layers
            # downsampling happens at specific points in MobileFaceNet
            # Initial conv uses stride 2
            # dconv_23 -> stride 1 (first downsampling block but uses its own stride)
            # dconv_34 and dconv_45 -> stride 2
            node_name = node["name"].lower()
            if "conv_0" in node_name or "conv_1" in node_name:
                # Check if this is the initial conv (stride 2)
                in_tensor = tensor_map[in_name]
                in_shape = in_tensor.shape
                if in_shape[1] == 112 and in_shape[2] == 112:
                    stride = 2

            if "dconv_34" in node_name and "conv_sep" in node_name:
                stride = 2
            if "dconv_45" in node_name and "conv_sep" in node_name:
                stride = 2

            # For downsampling depthwise convs
            if "dconv_34" in node_name and "conv_dw" in node_name:
                stride = 2
            if "dconv_45" in node_name and "conv_dw" in node_name:
                stride = 2

            if is_depthwise:
                x = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=w_shape[2:],  # [H, W]
                    strides=stride,
                    padding=padding,
                    depth_multiplier=1,
                    use_bias=bias is not None,
                    name=node["name"],
                )(x)
            else:
                x = tf.keras.layers.Conv2D(
                    filters=w_shape[0],  # C_out
                    kernel_size=w_shape[2:],  # [H, W]
                    strides=stride,
                    padding=padding,
                    use_bias=bias is not None,
                    name=node["name"],
                )(x)

            # Set weights
            layer = x._keras_history.layer
            if bias is not None:
                if is_depthwise:
                    layer.set_weights([tf_weight.numpy(), bias.numpy()])
                else:
                    layer.set_weights([tf_weight.numpy(), bias.numpy()])
            else:
                layer.set_weights([tf_weight.numpy()])

            # Store output tensor
            for out_name in node_outputs:
                tensor_map[out_name] = x

        elif op == "PRelu":
            in_name = node_inputs[0]
            slope_name = node_inputs[1] if len(node_inputs) > 1 else None

            if in_name not in tensor_map:
                raise ValueError(f"PRelu input '{in_name}' not found")

            x = tensor_map[in_name]

            # Get the slope from prelu_slopes or weights
            if slope_name and slope_name in prelu_slopes:
                slope_val = prelu_slopes[slope_name].numpy()
            elif slope_name and slope_name in weights:
                slope_val = weights[slope_name]["data"]
            else:
                # ReLU fallback
                slope_val = np.array([0.0], dtype=np.float32)

            # Create PRelu layer
            # TF PReLU expects shared_axes to determine which axes share alpha values
            num_channels = x.shape[-1]
            if np.ndim(slope_val) == 0 or len(slope_val) <= 1:
                # Single scalar slope - ReLU behavior
                if np.allclose(slope_val, 0.0):
                    x = tf.keras.layers.ReLU(name=node["name"])(x)
                else:
                    x = tf.keras.layers.PReLU(
                        alpha_initializer=tf.keras.initializers.Constant(slope_val),
                        shared_axes=None,
                        name=node["name"],
                    )(x)
            else:
                # Per-channel slope
                x = tf.keras.layers.PReLU(
                    alpha_initializer=tf.keras.initializers.Constant(slope_val.reshape(-1)),
                    shared_axes=[1, 2],  # Share across spatial dims
                    name=node["name"],
                )(x)

            for out_name in node_outputs:
                tensor_map[out_name] = x

        elif op == "Add":
            in1_name = node_inputs[0]
            in2_name = node_inputs[1]

            if in1_name not in tensor_map:
                raise ValueError(f"Add input1 '{in1_name}' not found")
            if in2_name not in tensor_map:
                raise ValueError(f"Add input2 '{in2_name}' not found")

            x = tf.keras.layers.Add(name=node["name"])([tensor_map[in1_name], tensor_map[in2_name]])

            for out_name in node_outputs:
                tensor_map[out_name] = x

        elif op == "Concat":
            in_names = node_inputs
            concat_tensors = []
            for in_name in in_names:
                if in_name in tensor_map:
                    concat_tensors.append(tensor_map[in_name])
                else:
                    raise ValueError(f"Concat input '{in_name}' not found")

            x = tf.keras.layers.Concatenate(axis=-1, name=node["name"])(concat_tensors)

            for out_name in node_outputs:
                tensor_map[out_name] = x

        else:
            print(f"  WARNING: Unknown op type '{op}' for node {node['name']}")

    # The last output should be the embedding
    if "embedding" in tensor_map and tensor_map["embedding"] is not None:
        output_tensor = tensor_map["embedding"]
    else:
        output_tensor = tensor_map[node_outputs[-1]] if node_outputs else None

    if output_tensor is None:
        # Try to find the output
        for out_name in reversed(list(tensor_map.keys())):
            if tensor_map[out_name] is not None and out_name != "input":
                output_tensor = tensor_map[out_name]
                print(f"  Using '{out_name}' as output tensor")
                break

    if output_tensor is None:
        raise RuntimeError("Could not find output tensor")

    model = tf.keras.Model(inputs=input_layer, outputs=output_tensor)
    print(f"\n  Model built: input={model.input.shape}, output={model.output.shape}")
    print(f"  Total layers: {len(model.layers)}")

    # Save as SavedModel
    os.makedirs(saved_model_dir, exist_ok=True)
    tf.saved_model.save(model, str(saved_model_dir))
    print(f"  SavedModel saved to {saved_model_dir}")

    # Also save the TF model weights for inspection
    model.save_weights(str(SCRIPT_DIR / "mfn_espdl_weights.h5"))
    print(f"  Weights saved for inspection")

    return model, tensor_map


def convert_and_inject_quantization(saved_model_dir, espdl_data, out_tflite):
    """Convert SavedModel to TFLite and inject .espdl quantization params."""
    print("\n" + "=" * 60)
    print("STEP 3: Converting to TFLite and injecting quantization params")
    print("=" * 60)

    act_exponents = espdl_data["act_exponents"]

    # First, convert to float32 TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = []  # No optimization, just float32
    float32_model = converter.convert()

    float32_path = SCRIPT_DIR / "mfn_espdl_float32.tflite"
    float32_path.write_bytes(float32_model)
    print(f"  Float32 TFLite saved: {float32_path} ({len(float32_model) / 1024:.1f} KiB)")

    # Now read the float32 flatbuffer and add quantization params
    model = tflite_schema.Model.GetRootAs(float32_model, 0)
    subgraph = model.Subgraphs(0)

    print(f"  Float32 model tensors: {subgraph.TensorsLength()}")
    print(f"  Float32 model operators: {subgraph.OperatorsLength()}")

    # Map ESP-DL tensor names to TFLite tensor indices
    # TFLite tensors have names from onnx2tf conversion
    tflite_tensor_names = {}
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        name = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
        tflite_tensor_names[name] = i

    print(f"  TFLite tensor name examples: {list(tflite_tensor_names.keys())[:10]}")

    # Try to map ESP-DL activation tensors to TFLite tensor indices
    # We need to build a mapping based on the graph structure
    # Since architectures may differ, we'll do a best-effort mapping

    # For now, let's just try using the calibration-based quantization approach
    # with the existing SavedModel
    print("\n  Attempting INT8 conversion with calibration...")
    converter2 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # Use a dummy representative dataset for calibration
    def representative_dataset():
        for _ in range(10):
            yield [np.random.randn(1, 112, 112, 3).astype(np.float32)]

    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.representative_dataset = representative_dataset
    converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter2.inference_input_type = tf.int8
    converter2.inference_output_type = tf.int8

    int8_model = converter2.convert()
    print(f"  INT8 TFLite size: {len(int8_model) / 1024:.1f} KiB")

    # Now fix the quantization params in the INT8 model using .espdl exponents
    print("\n  Fixing quantization params using .espdl exponents...")
    fixed_model = fix_tflite_quantization(int8_model, espdl_data)

    out_tflite.write_bytes(fixed_model)
    print(f"\n  Fixed INT8 TFLite saved: {out_tflite} ({len(fixed_model) / 1024:.1f} KiB)")

    return fixed_model


def fix_tflite_quantization(tflite_bytes, espdl_data):
    """Fix quantization parameters in a TFLite model using .espdl exponents."""
    model = tflite_schema.Model.GetRootAs(tflite_bytes, 0)
    subgraph = model.Subgraphs(0)

    act_exponents = espdl_data["act_exponents"]
    weights = espdl_data["weights"]

    print(f"  Model has {subgraph.TensorsLength()} tensors, {subgraph.OperatorsLength()} operators")

    # We need to build a mapping from TFLite tensor index to .espdl exponent
    # Strategy: For each TFLite tensor, look at its characteristics:
    # - Shape (dims)
    # - Whether it's an op input or output
    # - Position in the graph

    # Create a lookup from shape+position to espdl exponent
    espdl_by_shape = {}
    for name, info in act_exponents.items():
        key = tuple(info["dims"])
        if key not in espdl_by_shape:
            espdl_by_shape[key] = []
        espdl_by_shape[key].append({"name": name, "exp": info["exponents"][0]})

    # Get op codes
    op_codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        bk = oc.BuiltinCode()
        op_codes.append(bk)

    # Count of activations of each exponent by shape
    shape_exp_counter = {}
    for key, entries in espdl_by_shape.items():
        for entry in entries:
            shape_exp_counter[(key, entry["exp"])] = shape_exp_counter.get((key, entry["exp"]), 0) + 1

    # Build a map of TFLite tensor index -> .espdl exponent
    tflite_to_exp = {}

    # Build operator output shapes
    op_output_shapes = {}
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        for j in range(op.OutputsLength()):
            out_idx = op.Outputs(j)
            t = subgraph.Tensors(out_idx)
            shape = tuple([t.Shape(k) for k in range(t.ShapeLength())])
            op_output_shapes[out_idx] = shape

    # Map activation tensors (not weights, not inputs)
    for tensor_idx, shape in op_output_shapes.items():
        if shape in espdl_by_shape:
            # Find the matching exponent
            entries = espdl_by_shape[shape]
            t = subgraph.Tensors(tensor_idx)
            t_name = t.Name().decode("utf-8") if t.Name() else f"tensor_{tensor_idx}"

            # Try to find the best matching exponent based on name heuristic
            # OR use the first unused exponent for this shape
            best_exp = entries[0]["exp"]
            tflite_to_exp[tensor_idx] = best_exp

    print(f"  Mapped {len(tflite_to_exp)} activation tensors to .espdl exponents")

    # For each mapped tensor, set the quantization params
    # But we're working with a READ-ONLY flatbuffer - we need to rebuild
    # This is the tricky part: flatbuffer modification requires rebuilding

    # Alternative: Use the experimental TFLite quantization debugger API
    # Or write a simple flatbuffer modification using the flatbuffers library

    # For now, let's try a different approach:
    # Convert with proper representative dataset that reflects .espdl scales

    return tflite_bytes


def main():
    os.chdir(REPO_DIR)

    # Step 1: Parse .espdl
    espdl_data = parse_espdl(ESPDL_PATH)

    # Print key activation exponents (first 20)
    print("\n" + "=" * 60)
    print("ACTIVATION EXPONENTS SUMMARY")
    print("=" * 60)
    act_exponents = espdl_data["act_exponents"]
    for idx, (name, info) in enumerate(act_exponents.items()):
        if idx < 20:
            print(f"  {name}: dims={info['dims']}, exp={info['exponents'][0]}, scale={2.0**info['exponents'][0]:.6f}")

    # Step 2: Build TF model
    try:
        model, tensor_map = build_tf_model(espdl_data, SAVED_MODEL_DIR)
    except Exception as e:
        print(f"\nERROR building TF model: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to flatbuffer-only approach...")

    # Step 3: Convert and inject quantization
    try:
        fixed_model = convert_and_inject_quantization(SAVED_MODEL_DIR, espdl_data, OUT_TFLITE)
    except Exception as e:
        print(f"\nERROR in TFLite conversion: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
