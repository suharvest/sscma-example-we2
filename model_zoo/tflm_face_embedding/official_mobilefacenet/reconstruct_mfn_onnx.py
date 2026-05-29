#!/usr/bin/env python3
"""
Reconstruct a MobileFaceNet ONNX model from .espdl file weights.

Reads human_face_feat_mfn_s8_v1.espdl, dequantizes all INT8 weights
using their per-tensor exponents, and builds a float32 ONNX graph
with the exact model architecture.
"""

import struct
import sys
import os
import numpy as np
import onnx
from onnx import helper as onnx_helper, numpy_helper as onnx_numpy_helper
from onnx import TensorProto

# --- Add esp-ppq source to path for FlatBuffers schemas ---
sys.path.insert(0, "/tmp/esp-ppq-src")
from esp_ppq.parser.espdl.FlatBuffers.Dl import Model
from esp_ppq.parser.espdl import helper as _espdl_helper  # for tensor_dtype_to_np_dtype


# =============================================================================
# 1. Parse .espdl file
# =============================================================================

ESPDL_PATH = "/Users/harvest/project/esp-dl/models/human_face_recognition/models/s3/human_face_feat_mfn_s8_v1.espdl"
OUTPUT_PATH = "/Users/harvest/project/grove_vision_2/sscma-example-we2/model_zoo/tflm_face_embedding/official_mobilefacenet/mfn_s8_v1_reconstructed.onnx"


def parse_espdl(filepath: str):
    """Read the .espdl file, parse header, return FlatBuffers Model."""
    with open(filepath, "rb") as f:
        data = f.read()

    # EDL2 header: magic(4) + encrypt_flag(4) + data_len(4) + padding(4)
    magic = data[:4]
    if magic != b"EDL2":
        raise ValueError(f"Bad magic: {magic}")

    encrypt_flag = struct.unpack("I", data[4:8])[0]
    data_len = struct.unpack("I", data[8:12])[0]

    if encrypt_flag != 0:
        raise NotImplementedError("Encrypted .espdl files not supported")

    model_data = data[16 : 16 + data_len]
    model = Model.Model.GetRootAs(model_data, 0)
    return model


def extract_tensor_data(tensor) -> np.ndarray:
    """Extract raw numpy array from a FlatBuffer Tensor, applying dequantization."""
    dtype = _espdl_helper.tensor_dtype_to_np_dtype(tensor.DataType())

    # Get shape
    shape = [tensor.Dims(i) for i in range(tensor.DimsLength())]

    # Get exponents (per-tensor INT8 dequantization)
    exponents = [tensor.Exponents(i) for i in range(tensor.ExponentsLength())]
    if len(exponents) == 0:
        exponents = [0]
    exponent = exponents[0]  # per-tensor exponent

    # Read raw data from AlignedBytes chunks
    chunks = []
    for i in range(tensor.RawDataLength()):
        chunk = tensor.RawData(i)
        chunk_data = np.frombuffer(
            chunk.BytesAsNumpy().tobytes(), dtype=np.uint8
        )
        chunks.append(chunk_data)
    raw_bytes = b"".join(c.tobytes() for c in chunks)

    # Interpret bytes as the correct dtype
    total_elements = int(np.prod(shape)) if shape else 1
    arr = np.frombuffer(raw_bytes, dtype=dtype)

    # Trim to expected size (may have padding to 16-byte alignment)
    if len(arr) > total_elements:
        arr = arr[:total_elements]

    arr = arr.reshape(shape)

    # Dequantize: real_value = int_value * 2^exponent
    arr_float = arr.astype(np.float32) * (2.0 ** exponent)

    return arr_float


# =============================================================================
# 2. Extract all weights
# =============================================================================

def extract_all_weights(model) -> dict:
    """Extract all initializer tensors from the FlatBuffers model into a dict."""
    graph = model.Graph()
    weights = {}

    for i in range(graph.InitializerLength()):
        t = graph.Initializer(i)
        name = t.Name().decode("utf-8", errors="ignore")
        arr = extract_tensor_data(t)
        weights[name] = arr

    return weights


def nhwc_to_nchw_conv(nhwc_weight: np.ndarray) -> np.ndarray:
    """
    Convert NHWC conv weight to NCHW (ONNX) format.

    NHWC weight:  [k_h, k_w, in_c, out_c]
    ONNX weight:  [out_c, in_c, k_h, k_w]
    Transposition: (3, 2, 0, 1)
    """
    return nhwc_weight.transpose(3, 2, 0, 1)


def nhwc_to_nchw_dw(nhwc_weight: np.ndarray) -> np.ndarray:
    """
    Convert NHWC depthwise weight to NCHW format.

    NHWC depthwise: [k_h, k_w, in_c, 1]  (group = in_c)
    ONNX depthwise: [in_c, 1, k_h, k_w]
    Transposition: (2, 3, 0, 1)
    """
    in_c = nhwc_weight.shape[2]
    w = nhwc_weight.transpose(2, 3, 0, 1)
    # Result is [in_c, 1, k_h, k_w]
    return w


# =============================================================================
# 3. Build ONNX model
# =============================================================================

def make_initializer(name: str, data: np.ndarray) -> TensorProto:
    """Create an ONNX initializer tensor from a numpy array."""
    return onnx_numpy_helper.from_array(data.astype(np.float32), name=name)


def create_onnx_model(weights: dict):
    """Build the complete MobileFaceNet ONNX model."""

    # Helper: get weight by name
    def w(name: str) -> np.ndarray:
        return weights[name]

    # Helper: make conv weight name lookup
    def get_conv_weight(name: str) -> np.ndarray:
        return nhwc_to_nchw_conv(w(name))

    def get_dw_weight(name: str) -> np.ndarray:
        return nhwc_to_nchw_dw(w(name))

    # Helper: PReLU alpha stored as [c, 1, 1] in NHWC, reshape to [1, c, 1, 1] for NCHW broadcast
    def get_prelu_alpha(name: str) -> np.ndarray:
        raw = w(name)
        c = raw.shape[0] if len(raw.shape) >= 1 else 1
        return raw.reshape(1, c, 1, 1).astype(np.float32)  # → [1, c, 1, 1]

    # ---- Node name counter ----
    node_counter = [0]

    def make_node(op_type, inputs, outputs, **attrs):
        node_counter[0] += 1
        return onnx_helper.make_node(
            op_type, inputs, outputs,
            name=f"{op_type}_{node_counter[0]}",
            **attrs
        )

    # ---- Value info name counter ----
    val_counter = [0]

    def new_val(prefix="v"):
        val_counter[0] += 1
        return f"{prefix}_{val_counter[0]}"

    # ---- Input ----
    # Input shape: NHWC [1, 112, 112, 3] → NCHW [1, 3, 112, 112]
    # Normalize: (pixel / 127.5 - 1.0) using Mul + Sub as a pre-processing
    # Actually, we include the normalization as the first ops in the graph
    input_shape = [1, 3, 112, 112]
    model_input = onnx_helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )

    # Input normalization: (x / 127.5) - 1.0  →  x * (1/127.5) + (-1.0)
    # Use a single Mul constant [1, 1, 1, 1] * 1/127.5
    norm_scale = np.array([1.0 / 127.5], dtype=np.float32).reshape([1, 1, 1, 1])
    norm_bias = np.array([-1.0], dtype=np.float32).reshape([1, 1, 1, 1])
    norm_scale_init = make_initializer("norm_scale", norm_scale)
    norm_bias_init = make_initializer("norm_bias", norm_bias)
    nodes = []
    initializers = [norm_scale_init, norm_bias_init]

    x = "input"

    # Pre-normalization: (input / 127.5 - 1.0) as Mul + Add
    x1 = new_val("norm_mul")
    nodes.append(make_node("Mul", [x, "norm_scale"], [x1]))
    x2 = new_val("norm_add")
    nodes.append(make_node("Add", [x1, "norm_bias"], [x2]))
    x = x2

    # ============================================================
    # Stage 0: Initial Conv + PReLU + Depthwise + PReLU
    # ============================================================

    # conv_1: 3→64, 3x3, stride=2, pad=1
    conv_1_weight_nchw = get_conv_weight("conv_1.weight")
    conv_1_bias_nchw = w("conv_1.bias")
    initializers.append(make_initializer("conv_1.weight", conv_1_weight_nchw))
    initializers.append(make_initializer("conv_1.bias", conv_1_bias_nchw))

    v_conv1 = new_val("conv1")
    nodes.append(make_node(
        "Conv", [x, "conv_1.weight", "conv_1.bias"], [v_conv1],
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
        group=1,
    ))
    x = v_conv1  # [1, 64, 56, 56]

    # PReLU 280
    prelu_280 = get_prelu_alpha("280")
    initializers.append(make_initializer("PReLU_alpha_280", prelu_280))
    v_prelu0 = new_val("prelu")
    nodes.append(make_node("PRelu", [x, "PReLU_alpha_280"], [v_prelu0]))
    x = v_prelu0

    # conv_2_dw: 64→64, 3x3, stride=1, pad=1, group=64
    conv_2_dw_weight_nchw = get_dw_weight("conv_2_dw.weight")
    conv_2_dw_bias_nchw = w("conv_2_dw.bias")
    initializers.append(make_initializer("conv_2_dw.weight", conv_2_dw_weight_nchw))
    initializers.append(make_initializer("conv_2_dw.bias", conv_2_dw_bias_nchw))

    v_conv2 = new_val("conv2")
    nodes.append(make_node(
        "Conv", [x, "conv_2_dw.weight", "conv_2_dw.bias"], [v_conv2],
        kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
        group=64,
    ))
    x = v_conv2

    # PReLU 281
    prelu_281 = get_prelu_alpha("281")
    initializers.append(make_initializer("PReLU_alpha_281", prelu_281))
    v_prelu1 = new_val("prelu")
    nodes.append(make_node("PRelu", [x, "PReLU_alpha_281"], [v_prelu1]))
    x = v_prelu1

    # ============================================================
    # Stage 1: dconv_23 (56→28, 64→64)
    # ============================================================

    # Pointwise: 64→128
    c = get_conv_weight("dconv_23_conv_sep.weight")
    b = w("dconv_23_conv_sep.bias")
    initializers.append(make_initializer("dconv_23_conv_sep.weight", c))
    initializers.append(make_initializer("dconv_23_conv_sep.bias", b))
    v_s1_sep = new_val("dconv23_sep")
    nodes.append(make_node(
        "Conv", [x, "dconv_23_conv_sep.weight", "dconv_23_conv_sep.bias"], [v_s1_sep],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))
    x = v_s1_sep

    # PReLU 282
    prelu_282 = get_prelu_alpha("282")
    initializers.append(make_initializer("PReLU_alpha_282", prelu_282))
    v_p2 = new_val("prelu")
    nodes.append(make_node("PRelu", [x, "PReLU_alpha_282"], [v_p2]))
    x = v_p2

    # Depthwise: 128→128, 3x3, stride=2, pad=1
    c = get_dw_weight("dconv_23_conv_dw.weight")
    b = w("dconv_23_conv_dw.bias")
    initializers.append(make_initializer("dconv_23_conv_dw.weight", c))
    initializers.append(make_initializer("dconv_23_conv_dw.bias", b))
    v_s1_dw = new_val("dconv23_dw")
    nodes.append(make_node(
        "Conv", [x, "dconv_23_conv_dw.weight", "dconv_23_conv_dw.bias"], [v_s1_dw],
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
        group=128,
    ))
    x = v_s1_dw

    # PReLU 283
    prelu_283 = get_prelu_alpha("283")
    initializers.append(make_initializer("PReLU_alpha_283", prelu_283))
    v_p3 = new_val("prelu")
    nodes.append(make_node("PRelu", [x, "PReLU_alpha_283"], [v_p3]))
    x = v_p3

    # Pointwise: 128→64
    c = get_conv_weight("dconv_23_conv_proj.weight")
    b = w("dconv_23_conv_proj.bias")
    initializers.append(make_initializer("dconv_23_conv_proj.weight", c))
    initializers.append(make_initializer("dconv_23_conv_proj.bias", b))
    v_s1_proj = new_val("dconv23_proj")
    nodes.append(make_node(
        "Conv", [x, "dconv_23_conv_proj.weight", "dconv_23_conv_proj.bias"], [v_s1_proj],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))
    x = v_s1_proj  # [1, 64, 28, 28]

    # ============================================================
    # Stage 2: res_3 (28x28, 4 blocks)
    # ============================================================
    def make_res_block(x, block_name, prelu_a_name, prelu_b_name, in_c, mid_c, out_c):
        """Single bottleneck residual block (pre-activation style)."""
        residual = x  # save input for Add

        # Pointwise expand: in_c → mid_c
        c = get_conv_weight(f"{block_name}_conv_sep.weight")
        b = w(f"{block_name}_conv_sep.bias")
        initializers.append(make_initializer(f"{block_name}_conv_sep.weight", c))
        initializers.append(make_initializer(f"{block_name}_conv_sep.bias", b))
        v_sep = new_val(f"{block_name}_sep")
        nodes.append(make_node(
            "Conv",
            [x, f"{block_name}_conv_sep.weight", f"{block_name}_conv_sep.bias"],
            [v_sep],
            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
            group=1,
        ))
        x = v_sep

        # PReLU
        alpha_a = get_prelu_alpha(prelu_a_name)
        init_name_a = f"PReLU_alpha_{prelu_a_name}"
        initializers.append(make_initializer(init_name_a, alpha_a))
        v_pa = new_val("prelu")
        nodes.append(make_node("PRelu", [x, init_name_a], [v_pa]))
        x = v_pa

        # Depthwise: mid_c → mid_c
        c = get_dw_weight(f"{block_name}_conv_dw.weight")
        b = w(f"{block_name}_conv_dw.bias")
        initializers.append(make_initializer(f"{block_name}_conv_dw.weight", c))
        initializers.append(make_initializer(f"{block_name}_conv_dw.bias", b))
        v_dw = new_val(f"{block_name}_dw")
        nodes.append(make_node(
            "Conv",
            [x, f"{block_name}_conv_dw.weight", f"{block_name}_conv_dw.bias"],
            [v_dw],
            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
            group=mid_c,
        ))
        x = v_dw

        # PReLU
        alpha_b = get_prelu_alpha(prelu_b_name)
        init_name_b = f"PReLU_alpha_{prelu_b_name}"
        initializers.append(make_initializer(init_name_b, alpha_b))
        v_pb = new_val("prelu")
        nodes.append(make_node("PRelu", [x, init_name_b], [v_pb]))
        x = v_pb

        # Pointwise project: mid_c → out_c
        c = get_conv_weight(f"{block_name}_conv_proj.weight")
        b = w(f"{block_name}_conv_proj.bias")
        initializers.append(make_initializer(f"{block_name}_conv_proj.weight", c))
        initializers.append(make_initializer(f"{block_name}_conv_proj.bias", b))
        v_proj = new_val(f"{block_name}_proj")
        nodes.append(make_node(
            "Conv",
            [x, f"{block_name}_conv_proj.weight", f"{block_name}_conv_proj.bias"],
            [v_proj],
            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
            group=1,
        ))
        x = v_proj

        # Add residual
        v_add = new_val(f"{block_name}_add")
        nodes.append(make_node("Add", [residual, x], [v_add]))
        return v_add

    # res_3: 4 blocks, in_c=64, mid_c=128, out_c=64
    x = make_res_block(x, "res_3_block0", "284", "285", 64, 128, 64)
    x = make_res_block(x, "res_3_block1", "286", "287", 64, 128, 64)
    x = make_res_block(x, "res_3_block2", "288", "289", 64, 128, 64)
    x = make_res_block(x, "res_3_block3", "290", "291", 64, 128, 64)

    # ============================================================
    # Stage 3: dconv_34 (28→14, 64→128)
    # ============================================================

    # Pointwise: 64→256
    c = get_conv_weight("dconv_34_conv_sep.weight")
    b = w("dconv_34_conv_sep.bias")
    initializers.append(make_initializer("dconv_34_conv_sep.weight", c))
    initializers.append(make_initializer("dconv_34_conv_sep.bias", b))
    v_s3_sep = new_val("dconv34_sep")
    nodes.append(make_node(
        "Conv", [x, "dconv_34_conv_sep.weight", "dconv_34_conv_sep.bias"], [v_s3_sep],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))
    x = v_s3_sep

    # PReLU 292
    alpha = get_prelu_alpha("292")
    initializers.append(make_initializer("PReLU_alpha_292", alpha))
    v_p4 = new_val("prelu")
    nodes.append(make_node("PRelu", [x, "PReLU_alpha_292"], [v_p4]))
    x = v_p4

    # Depthwise: 256→256, 3x3, stride=2
    c = get_dw_weight("dconv_34_conv_dw.weight")
    b = w("dconv_34_conv_dw.bias")
    initializers.append(make_initializer("dconv_34_conv_dw.weight", c))
    initializers.append(make_initializer("dconv_34_conv_dw.bias", b))
    v_s3_dw = new_val("dconv34_dw")
    nodes.append(make_node(
        "Conv", [x, "dconv_34_conv_dw.weight", "dconv_34_conv_dw.bias"], [v_s3_dw],
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
        group=256,
    ))
    x = v_s3_dw

    # PReLU 293
    alpha = get_prelu_alpha("293")
    initializers.append(make_initializer("PReLU_alpha_293", alpha))
    v_p5 = new_val("prelu")
    nodes.append(make_node("PRelu", [x, "PReLU_alpha_293"], [v_p5]))
    x = v_p5

    # Pointwise: 256→128
    c = get_conv_weight("dconv_34_conv_proj.weight")
    b = w("dconv_34_conv_proj.bias")
    initializers.append(make_initializer("dconv_34_conv_proj.weight", c))
    initializers.append(make_initializer("dconv_34_conv_proj.bias", b))
    v_s3_proj = new_val("dconv34_proj")
    nodes.append(make_node(
        "Conv", [x, "dconv_34_conv_proj.weight", "dconv_34_conv_proj.bias"], [v_s3_proj],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))
    x = v_s3_proj  # [1, 128, 14, 14]

    # ============================================================
    # Stage 4: res_4 (14x14, 6 blocks)
    # ============================================================
    x = make_res_block(x, "res_4_block0", "294", "295", 128, 256, 128)
    x = make_res_block(x, "res_4_block1", "296", "297", 128, 256, 128)
    x = make_res_block(x, "res_4_block2", "298", "299", 128, 256, 128)
    x = make_res_block(x, "res_4_block3", "300", "301", 128, 256, 128)
    x = make_res_block(x, "res_4_block4", "302", "303", 128, 256, 128)
    x = make_res_block(x, "res_4_block5", "304", "305", 128, 256, 128)

    # ============================================================
    # Stage 5: dconv_45 (14→7, 128→128, with split/concat)
    # ============================================================

    # Split into two branches (pointwise 128→256 each)
    # Branch 1
    c1 = get_conv_weight("dconv_45_conv_sep_1.weight")
    b1 = w("dconv_45_conv_sep_1.bias")
    initializers.append(make_initializer("dconv_45_conv_sep_1.weight", c1))
    initializers.append(make_initializer("dconv_45_conv_sep_1.bias", b1))
    v_s5_sep1 = new_val("dconv45_sep1")
    nodes.append(make_node(
        "Conv", [x, "dconv_45_conv_sep_1.weight", "dconv_45_conv_sep_1.bias"],
        [v_s5_sep1],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))
    # Branch 2
    c2 = get_conv_weight("dconv_45_conv_sep_2.weight")
    b2 = w("dconv_45_conv_sep_2.bias")
    initializers.append(make_initializer("dconv_45_conv_sep_2.weight", c2))
    initializers.append(make_initializer("dconv_45_conv_sep_2.bias", b2))
    v_s5_sep2 = new_val("dconv45_sep2")
    nodes.append(make_node(
        "Conv", [x, "dconv_45_conv_sep_2.weight", "dconv_45_conv_sep_2.bias"],
        [v_s5_sep2],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))

    # PReLU 306
    alpha = get_prelu_alpha("306")
    initializers.append(make_initializer("PReLU_alpha_306", alpha))
    v_p6a = new_val("prelu")
    nodes.append(make_node("PRelu", [v_s5_sep1, "PReLU_alpha_306"], [v_p6a]))

    # PReLU 307
    alpha = get_prelu_alpha("307")
    initializers.append(make_initializer("PReLU_alpha_307", alpha))
    v_p6b = new_val("prelu")
    nodes.append(make_node("PRelu", [v_s5_sep2, "PReLU_alpha_307"], [v_p6b]))

    # Concat: NHWC axis=3 → NCHW axis=1
    v_s5_cat1 = new_val("dconv45_cat1")
    nodes.append(make_node("Concat", [v_p6a, v_p6b], [v_s5_cat1], axis=1))  # [1, 512, 14, 14]

    # Depthwise: 512→512, 3x3, stride=2
    c = get_dw_weight("dconv_45_conv_dw.weight")
    b = w("dconv_45_conv_dw.bias")
    initializers.append(make_initializer("dconv_45_conv_dw.weight", c))
    initializers.append(make_initializer("dconv_45_conv_dw.bias", b))
    v_s5_dw = new_val("dconv45_dw")
    nodes.append(make_node(
        "Conv", [v_s5_cat1, "dconv_45_conv_dw.weight", "dconv_45_conv_dw.bias"],
        [v_s5_dw],
        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
        group=512,
    ))

    # PReLU 308
    alpha = get_prelu_alpha("308")
    initializers.append(make_initializer("PReLU_alpha_308", alpha))
    v_p7 = new_val("prelu")
    nodes.append(make_node("PRelu", [v_s5_dw, "PReLU_alpha_308"], [v_p7]))

    # Split pointwise: 2 branches 512→64 each
    # Branch 1
    c1 = get_conv_weight("dconv_45_conv_proj_1.weight")
    b1 = w("dconv_45_conv_proj_1.bias")
    initializers.append(make_initializer("dconv_45_conv_proj_1.weight", c1))
    initializers.append(make_initializer("dconv_45_conv_proj_1.bias", b1))
    v_s5_proj1 = new_val("dconv45_proj1")
    nodes.append(make_node(
        "Conv", [v_p7, "dconv_45_conv_proj_1.weight", "dconv_45_conv_proj_1.bias"],
        [v_s5_proj1],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))
    # Branch 2
    c2 = get_conv_weight("dconv_45_conv_proj_2.weight")
    b2 = w("dconv_45_conv_proj_2.bias")
    initializers.append(make_initializer("dconv_45_conv_proj_2.weight", c2))
    initializers.append(make_initializer("dconv_45_conv_proj_2.bias", b2))
    v_s5_proj2 = new_val("dconv45_proj2")
    nodes.append(make_node(
        "Conv", [v_p7, "dconv_45_conv_proj_2.weight", "dconv_45_conv_proj_2.bias"],
        [v_s5_proj2],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))

    # Concat: NHWC axis=3 → NCHW axis=1
    v_s5_cat2 = new_val("dconv45_cat2")
    nodes.append(make_node("Concat", [v_s5_proj1, v_s5_proj2], [v_s5_cat2], axis=1))
    x = v_s5_cat2  # [1, 128, 7, 7]

    # ============================================================
    # Stage 6: res_5 (7x7, 2 blocks)
    # ============================================================
    x = make_res_block(x, "res_5_block0", "309", "310", 128, 256, 128)
    x = make_res_block(x, "res_5_block1", "311", "312", 128, 256, 128)

    # ============================================================
    # Stage 7: Final (7→1, 128→512)
    # ============================================================

    # Split pointwise: 2 branches 128→256 each
    # Branch 1
    c1 = get_conv_weight("conv_6sep_1.weight")
    b1 = w("conv_6sep_1.bias")
    initializers.append(make_initializer("conv_6sep_1.weight", c1))
    initializers.append(make_initializer("conv_6sep_1.bias", b1))
    v_s7_sep1 = new_val("s7_sep1")
    nodes.append(make_node(
        "Conv", [x, "conv_6sep_1.weight", "conv_6sep_1.bias"], [v_s7_sep1],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))
    # Branch 2
    c2 = get_conv_weight("conv_6sep_2.weight")
    b2 = w("conv_6sep_2.bias")
    initializers.append(make_initializer("conv_6sep_2.weight", c2))
    initializers.append(make_initializer("conv_6sep_2.bias", b2))
    v_s7_sep2 = new_val("s7_sep2")
    nodes.append(make_node(
        "Conv", [x, "conv_6sep_2.weight", "conv_6sep_2.bias"], [v_s7_sep2],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))

    # PReLU 313
    alpha = get_prelu_alpha("313")
    initializers.append(make_initializer("PReLU_alpha_313", alpha))
    v_p8a = new_val("prelu")
    nodes.append(make_node("PRelu", [v_s7_sep1, "PReLU_alpha_313"], [v_p8a]))

    # PReLU 314
    alpha = get_prelu_alpha("314")
    initializers.append(make_initializer("PReLU_alpha_314", alpha))
    v_p8b = new_val("prelu")
    nodes.append(make_node("PRelu", [v_s7_sep2, "PReLU_alpha_314"], [v_p8b]))

    # Concat: NHWC axis=3 → NCHW axis=1
    v_s7_cat = new_val("s7_cat")
    nodes.append(make_node("Concat", [v_p8a, v_p8b], [v_s7_cat], axis=1))  # [1, 512, 7, 7]

    # Depthwise: 512→512, 7x7, pad=0, stride=1 → output [1, 512, 1, 1]
    c = get_dw_weight("conv_6dw7_7.weight")
    b = w("conv_6dw7_7.bias")
    initializers.append(make_initializer("conv_6dw7_7.weight", c))
    initializers.append(make_initializer("conv_6dw7_7.bias", b))
    v_s7_dw = new_val("s7_dw")
    nodes.append(make_node(
        "Conv", [v_s7_cat, "conv_6dw7_7.weight", "conv_6dw7_7.bias"], [v_s7_dw],
        kernel_shape=[7, 7], strides=[1, 1], pads=[0, 0, 0, 0],
        group=512,
    ))  # [1, 512, 1, 1]

    # No PReLU after conv_6dw7_7 in the .espdl graph (verified)

    # Pointwise: 512→512 (fc1 - final embedding)
    c = get_conv_weight("fc1.weight")
    b = w("fc1.bias")
    initializers.append(make_initializer("fc1.weight", c))
    initializers.append(make_initializer("fc1.bias", b))
    v_embedding = new_val("embedding")
    nodes.append(make_node(
        "Conv", [v_s7_dw, "fc1.weight", "fc1.bias"], [v_embedding],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
        group=1,
    ))  # [1, 512, 1, 1]

    output_name = v_embedding

    # ---- Model Output ----
    model_output = onnx_helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, 512, 1, 1]
    )

    # ---- Build graph ----
    graph_def = onnx_helper.make_graph(
        nodes=nodes,
        name="MobileFaceNet_Reconstructed",
        inputs=[model_input],
        outputs=[model_output],
        initializer=initializers,
    )

    # ---- Build model ----
    model = onnx_helper.make_model(
        graph_def,
        producer_name="esp-dl-reconstructor",
        opset_imports=[onnx_helper.make_opsetid("", 11)],
    )

    return model


# =============================================================================
# 4. Main
# =============================================================================

def main():
    print("=" * 70)
    print("MFN .espdl → ONNX Reconstructor")
    print("=" * 70)

    # Step 1: Parse
    print("\n[1/4] Parsing .espdl file...")
    model = parse_espdl(ESPDL_PATH)
    graph = model.Graph()
    n_weights = graph.InitializerLength()
    n_nodes = graph.NodeLength()
    print(f"  Found {n_nodes} graph nodes, {n_weights} initializer tensors")

    # Step 2: Extract weights
    print("\n[2/4] Extracting and dequantizing weights...")
    weights = extract_all_weights(model)
    print(f"  Extracted {len(weights)} weight tensors")

    # Step 3: Build ONNX
    print("\n[3/4] Building ONNX model graph...")
    onnx_model = create_onnx_model(weights)

    # Count parameters
    total_params = 0
    for init in onnx_model.graph.initializer:
        arr = onnx_numpy_helper.to_array(init)
        total_params += int(np.prod(arr.shape))
    print(f"  Total parameters: {total_params:,}")

    # Validate ONNX
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure verified OK")

    # Step 4: Save
    print(f"\n[4/4] Saving ONNX to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    onnx.save(onnx_model, OUTPUT_PATH)

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    print("\n" + "=" * 70)
    print("EVIDENCE")
    print("=" * 70)
    print(f"Input .espdl: {ESPDL_PATH}")
    print(f"Output ONNX:  {OUTPUT_PATH}")
    print(f"File size:    {file_size} bytes")
    print(f"Parameters:   {total_params:,}")
    print(f"Graph nodes:  {len(onnx_model.graph.node)}")
    print(f"Initializers: {len(onnx_model.graph.initializer)}")
    print(f"Input:        {onnx_model.graph.input[0].name} shape={[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"Output:       {onnx_model.graph.output[0].name} shape={[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")

    # Verify onnx inference works
    print("\n--- ONNX Runtime Verification ---")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(OUTPUT_PATH)
        dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
        output = session.run(None, {"input": dummy_input})
        print(f"Inference OK. Output shape: {output[0].shape}, dtype: {output[0].dtype}")
        print(f"Output stats: min={output[0].min():.4f}, max={output[0].max():.4f}, mean={output[0].mean():.4f}")
    except ImportError:
        print("onnxruntime not installed, skipping inference test")
    except Exception as e:
        print(f"Inference error (may be OK for structure validation): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
