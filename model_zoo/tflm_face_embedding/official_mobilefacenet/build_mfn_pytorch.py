#!/usr/bin/env python3
"""
Build a standard MobileFaceNet in PyTorch, load dequantized weights from .espdl,
verify outputs, and export to ONNX (PReLU + ReLU variants).

Architecture: Standard MobileFaceNet with MERGED single-path dconv_45 and conv_6.
No BatchNorm (folded into Conv weights during quantization).
No final L2 normalization (done in post-processing by esp-dl).

Weight format: NHWC in .espdl → NCHW for PyTorch/ONNX.
Split weights in dconv_45 and conv_6 are merged before NCHW conversion.
"""

import struct
import sys
import os
import numpy as np
import torch
import torch.nn as nn

# --- Add esp-ppq source to path ---
sys.path.insert(0, "/tmp/esp-ppq-src")
from esp_ppq.parser.espdl.FlatBuffers.Dl import Model
from esp_ppq.parser.espdl import helper as _espdl_helper


# =============================================================================
# Configuration
# =============================================================================

ESPDL_PATH = "/Users/harvest/project/esp-dl/models/human_face_recognition/models/s3/human_face_feat_mfn_s8_v1.espdl"
OUTPUT_DIR = "/Users/harvest/project/grove_vision_2/sscma-example-we2/model_zoo/tflm_face_embedding/official_mobilefacenet"
PTH_PATH = os.path.join(OUTPUT_DIR, "mfn_s8_v1_pytorch.pth")
ONNX_PRELU_PATH = os.path.join(OUTPUT_DIR, "mfn_s8_v1_pytorch.onnx")
ONNX_RELU_PATH = os.path.join(OUTPUT_DIR, "mfn_s8_v1_pytorch_relu.onnx")


# =============================================================================
# 1. Parse .espdl and extract weights
# =============================================================================

def parse_espdl(filepath: str):
    """Read the .espdl file and return the FlatBuffers Model."""
    with open(filepath, "rb") as f:
        data = f.read()
    magic = data[:4]
    if magic != b"EDL2":
        raise ValueError(f"Bad magic: {magic}")
    encrypt_flag = struct.unpack("I", data[4:8])[0]
    data_len = struct.unpack("I", data[8:12])[0]
    if encrypt_flag != 0:
        raise NotImplementedError("Encrypted .espdl files not supported")
    model_data = data[16 : 16 + data_len]
    return Model.Model.GetRootAs(model_data, 0)


def extract_tensor_data(tensor) -> np.ndarray:
    """Extract raw numpy array from a FlatBuffer Tensor, applying dequantization."""
    dtype = _espdl_helper.tensor_dtype_to_np_dtype(tensor.DataType())
    shape = [tensor.Dims(i) for i in range(tensor.DimsLength())]
    exponents = [tensor.Exponents(i) for i in range(tensor.ExponentsLength())]
    exponent = exponents[0] if exponents else 0

    # Read raw bytes from AlignedBytes chunks
    chunks = []
    for i in range(tensor.RawDataLength()):
        chunk = tensor.RawData(i)
        chunks.append(np.frombuffer(chunk.BytesAsNumpy().tobytes(), dtype=np.uint8))
    raw_bytes = b"".join(c.tobytes() for c in chunks)

    total_elements = int(np.prod(shape)) if shape else 1
    arr = np.frombuffer(raw_bytes, dtype=dtype)
    if len(arr) > total_elements:
        arr = arr[:total_elements]
    arr = arr.reshape(shape)

    # Dequantize: real_value = int_value * 2^exponent
    arr_float = arr.astype(np.float32) * (2.0 ** exponent)
    return arr_float


def extract_all_weights(model) -> dict:
    """Extract all initializer tensors into a dict: name → dequantized float32 ndarray."""
    graph = model.Graph()
    weights = {}
    for i in range(graph.InitializerLength()):
        t = graph.Initializer(i)
        name = t.Name().decode("utf-8", errors="ignore")
        arr = extract_tensor_data(t)
        weights[name] = arr
    return weights


# =============================================================================
# 2. NHWC → NCHW conversion utilities
# =============================================================================

def nhwc_to_nchw_conv(nhwc_weight: np.ndarray) -> np.ndarray:
    """
    Convert NHWC conv weight to NCHW format.
    NHWC: [k_h, k_w, in_c, out_c] → NCHW: [out_c, in_c, k_h, k_w]
    """
    return nhwc_weight.transpose(3, 2, 0, 1)


def nhwc_to_nchw_dw(nhwc_weight: np.ndarray) -> np.ndarray:
    """
    Convert NHWC depthwise weight to NCHW format.
    NHWC: [k_h, k_w, in_c, 1] → NCHW: [in_c, 1, k_h, k_w]
    """
    in_c = nhwc_weight.shape[2]
    return nhwc_weight.transpose(2, 3, 0, 1)


# =============================================================================
# 3. PyTorch Model Definition
# =============================================================================

class Bottleneck(nn.Module):
    """MobileFaceNet bottleneck residual block (no BatchNorm, no PReLU after proj)."""

    def __init__(self, in_c: int, mid_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.sep = nn.Conv2d(in_c, mid_c, 1, stride=1, bias=True)  # expand
        self.prelu_sep = nn.PReLU(mid_c)
        self.dw = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1,
                            groups=mid_c, bias=True)
        self.prelu_dw = nn.PReLU(mid_c)
        self.proj = nn.Conv2d(mid_c, out_c, 1, stride=1, bias=True)  # project

    def forward(self, x):
        residual = x
        x = self.sep(x)
        x = self.prelu_sep(x)
        x = self.dw(x)
        x = self.prelu_dw(x)
        x = self.proj(x)
        return x + residual


class MobileFaceNet(nn.Module):
    """
    Standard MobileFaceNet with merged single-path dconv_45 and conv_6.
    No BatchNorm. No final L2 normalization.
    Input: [N, 3, 112, 112] in range [0, 255].
    Output: [N, 512, 1, 1]
    """

    def __init__(self):
        super().__init__()

        # --- Stage 0: 112→56 ---
        self.conv_1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=True)
        self.prelu_0 = nn.PReLU(64)
        self.conv_2_dw = nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64, bias=True)
        self.prelu_1 = nn.PReLU(64)

        # --- Stage 1: dconv_23, 56→28, 64→64 ---
        self.dconv23_sep = nn.Conv2d(64, 128, 1, bias=True)
        self.prelu_2 = nn.PReLU(128)
        self.dconv23_dw = nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=True)
        self.prelu_3 = nn.PReLU(128)
        self.dconv23_proj = nn.Conv2d(128, 64, 1, bias=True)

        # --- Stage 2: res_3, 4 blocks at 28×28, 64→128→64 ---
        self.res3_blocks = nn.ModuleList([
            Bottleneck(64, 128, 64, stride=1) for _ in range(4)
        ])

        # --- Stage 3: dconv_34, 28→14, 64→128 ---
        self.dconv34_sep = nn.Conv2d(64, 256, 1, bias=True)
        self.prelu_4 = nn.PReLU(256)
        self.dconv34_dw = nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256, bias=True)
        self.prelu_5 = nn.PReLU(256)
        self.dconv34_proj = nn.Conv2d(256, 128, 1, bias=True)

        # --- Stage 4: res_4, 6 blocks at 14×14, 128→256→128 ---
        self.res4_blocks = nn.ModuleList([
            Bottleneck(128, 256, 128, stride=1) for _ in range(6)
        ])

        # --- Stage 5: dconv_45 (MERGED), 14→7, 128→128 ---
        self.dconv45_sep = nn.Conv2d(128, 512, 1, bias=True)
        self.prelu_6 = nn.PReLU(512)
        self.dconv45_dw = nn.Conv2d(512, 512, 3, stride=2, padding=1, groups=512, bias=True)
        self.prelu_7 = nn.PReLU(512)
        self.dconv45_proj = nn.Conv2d(512, 128, 1, bias=True)

        # --- Stage 6: res_5, 2 blocks at 7×7, 128→256→128 ---
        self.res5_blocks = nn.ModuleList([
            Bottleneck(128, 256, 128, stride=1) for _ in range(2)
        ])

        # --- Stage 7: conv_6 (MERGED), 7→1, 128→512 ---
        self.conv6_sep = nn.Conv2d(128, 512, 1, bias=True)
        self.prelu_8 = nn.PReLU(512)
        # DW conv 7×7, pad=0, stride=1 → 1×1 output
        self.conv6_dw = nn.Conv2d(512, 512, 7, stride=1, padding=0, groups=512, bias=True)
        # No PReLU after DW in the .espdl graph
        self.fc1 = nn.Conv2d(512, 512, 1, bias=True)

    def forward(self, x):
        # Input normalization: (pixel / 127.5 - 1.0)
        x = x / 127.5 - 1.0

        # Stage 0
        x = self.prelu_0(self.conv_1(x))
        x = self.prelu_1(self.conv_2_dw(x))

        # Stage 1
        x = self.prelu_2(self.dconv23_sep(x))
        x = self.prelu_3(self.dconv23_dw(x))
        x = self.dconv23_proj(x)

        # Stage 2
        for blk in self.res3_blocks:
            x = blk(x)

        # Stage 3
        x = self.prelu_4(self.dconv34_sep(x))
        x = self.prelu_5(self.dconv34_dw(x))
        x = self.dconv34_proj(x)

        # Stage 4
        for blk in self.res4_blocks:
            x = blk(x)

        # Stage 5 (merged)
        x = self.prelu_6(self.dconv45_sep(x))
        x = self.prelu_7(self.dconv45_dw(x))
        x = self.dconv45_proj(x)

        # Stage 6
        for blk in self.res5_blocks:
            x = blk(x)

        # Stage 7 (merged)
        x = self.prelu_8(self.conv6_sep(x))
        x = self.conv6_dw(x)          # no PReLU after DW
        x = self.fc1(x)

        return x  # [N, 512, 1, 1]


# =============================================================================
# 4. Weight Loading
# =============================================================================

def _set_conv_weight(conv_layer: nn.Conv2d, nhwc_weight: np.ndarray, bias: np.ndarray):
    """Set pointwise conv weight + bias from NHWC format."""
    nchw_w = torch.from_numpy(nhwc_to_nchw_conv(nhwc_weight).copy())
    conv_layer.weight.data.copy_(nchw_w)
    if bias is not None:
        conv_layer.bias.data.copy_(torch.from_numpy(bias.astype(np.float32).copy()))


def _set_dw_weight(conv_layer: nn.Conv2d, nhwc_weight: np.ndarray, bias: np.ndarray):
    """Set depthwise conv weight + bias from NHWC format."""
    nchw_w = torch.from_numpy(nhwc_to_nchw_dw(nhwc_weight).copy())
    conv_layer.weight.data.copy_(nchw_w)
    if bias is not None:
        conv_layer.bias.data.copy_(torch.from_numpy(bias.astype(np.float32).copy()))


def _set_prelu(prelu_layer: nn.PReLU, alpha_raw: np.ndarray):
    """
    Set PReLU alpha. The raw alpha from .espdl has shape [c, 1, 1] (NHWC).
    PyTorch PReLU expects shape [c] or [1].
    """
    c = alpha_raw.shape[0]
    alpha = alpha_raw.flatten().astype(np.float32).copy()
    prelu_layer.weight.data.copy_(torch.from_numpy(alpha))


def load_weights(model: MobileFaceNet, weights: dict):
    """Load all dequantized weights into the PyTorch model."""

    w = weights  # shorthand

    # --- Stage 0 ---
    _set_conv_weight(model.conv_1, w["conv_1.weight"], w["conv_1.bias"])
    _set_prelu(model.prelu_0, w["280"])
    _set_dw_weight(model.conv_2_dw, w["conv_2_dw.weight"], w["conv_2_dw.bias"])
    _set_prelu(model.prelu_1, w["281"])

    # --- Stage 1 ---
    _set_conv_weight(model.dconv23_sep, w["dconv_23_conv_sep.weight"], w["dconv_23_conv_sep.bias"])
    _set_prelu(model.prelu_2, w["282"])
    _set_dw_weight(model.dconv23_dw, w["dconv_23_conv_dw.weight"], w["dconv_23_conv_dw.bias"])
    _set_prelu(model.prelu_3, w["283"])
    _set_conv_weight(model.dconv23_proj, w["dconv_23_conv_proj.weight"], w["dconv_23_conv_proj.bias"])

    # --- Stage 2: res_3 (4 blocks) ---
    for n in range(4):
        blk = model.res3_blocks[n]
        _set_conv_weight(blk.sep, w[f"res_3_block{n}_conv_sep.weight"],
                         w[f"res_3_block{n}_conv_sep.bias"])
        _set_prelu(blk.prelu_sep, w[str(284 + 2 * n)])
        _set_dw_weight(blk.dw, w[f"res_3_block{n}_conv_dw.weight"],
                       w[f"res_3_block{n}_conv_dw.bias"])
        _set_prelu(blk.prelu_dw, w[str(285 + 2 * n)])
        _set_conv_weight(blk.proj, w[f"res_3_block{n}_conv_proj.weight"],
                         w[f"res_3_block{n}_conv_proj.bias"])

    # --- Stage 3 ---
    _set_conv_weight(model.dconv34_sep, w["dconv_34_conv_sep.weight"], w["dconv_34_conv_sep.bias"])
    _set_prelu(model.prelu_4, w["292"])
    _set_dw_weight(model.dconv34_dw, w["dconv_34_conv_dw.weight"], w["dconv_34_conv_dw.bias"])
    _set_prelu(model.prelu_5, w["293"])
    _set_conv_weight(model.dconv34_proj, w["dconv_34_conv_proj.weight"], w["dconv_34_conv_proj.bias"])

    # --- Stage 4: res_4 (6 blocks) ---
    for n in range(6):
        blk = model.res4_blocks[n]
        _set_conv_weight(blk.sep, w[f"res_4_block{n}_conv_sep.weight"],
                         w[f"res_4_block{n}_conv_sep.bias"])
        _set_prelu(blk.prelu_sep, w[str(294 + 2 * n)])
        _set_dw_weight(blk.dw, w[f"res_4_block{n}_conv_dw.weight"],
                       w[f"res_4_block{n}_conv_dw.bias"])
        _set_prelu(blk.prelu_dw, w[str(295 + 2 * n)])
        _set_conv_weight(blk.proj, w[f"res_4_block{n}_conv_proj.weight"],
                         w[f"res_4_block{n}_conv_proj.bias"])

    # --- Stage 5: dconv_45 (MERGED: sep split + proj split) ---
    # Merge sep_1 + sep_2 along NHWC axis=3 (output channel)
    sep1_nhwc = w["dconv_45_conv_sep_1.weight"]   # [1,1,128,256]
    sep2_nhwc = w["dconv_45_conv_sep_2.weight"]   # [1,1,128,256]
    sep_merged_nhwc = np.concatenate([sep1_nhwc, sep2_nhwc], axis=3)  # [1,1,128,512]
    sep1_b = w["dconv_45_conv_sep_1.bias"]
    sep2_b = w["dconv_45_conv_sep_2.bias"]
    sep_merged_b = np.concatenate([sep1_b, sep2_b], axis=0)  # [512]

    _set_conv_weight(model.dconv45_sep, sep_merged_nhwc, sep_merged_b)
    # Merge PReLU alphas 306 + 307
    alpha_306_307 = np.concatenate([w["306"].flatten(), w["307"].flatten()], axis=0)  # [512]
    model.prelu_6.weight.data.copy_(torch.from_numpy(alpha_306_307.astype(np.float32)))

    # DW conv is unchanged
    _set_dw_weight(model.dconv45_dw, w["dconv_45_conv_dw.weight"], w["dconv_45_conv_dw.bias"])
    _set_prelu(model.prelu_7, w["308"])

    # Merge proj_1 + proj_2 along NHWC axis=3
    proj1_nhwc = w["dconv_45_conv_proj_1.weight"]  # [1,1,512,64]
    proj2_nhwc = w["dconv_45_conv_proj_2.weight"]  # [1,1,512,64]
    proj_merged_nhwc = np.concatenate([proj1_nhwc, proj2_nhwc], axis=3)  # [1,1,512,128]
    proj1_b = w["dconv_45_conv_proj_1.bias"]
    proj2_b = w["dconv_45_conv_proj_2.bias"]
    proj_merged_b = np.concatenate([proj1_b, proj2_b], axis=0)  # [128]

    _set_conv_weight(model.dconv45_proj, proj_merged_nhwc, proj_merged_b)

    # --- Stage 6: res_5 (2 blocks) ---
    for n in range(2):
        blk = model.res5_blocks[n]
        _set_conv_weight(blk.sep, w[f"res_5_block{n}_conv_sep.weight"],
                         w[f"res_5_block{n}_conv_sep.bias"])
        _set_prelu(blk.prelu_sep, w[str(309 + 2 * n)])
        _set_dw_weight(blk.dw, w[f"res_5_block{n}_conv_dw.weight"],
                       w[f"res_5_block{n}_conv_dw.bias"])
        _set_prelu(blk.prelu_dw, w[str(310 + 2 * n)])
        _set_conv_weight(blk.proj, w[f"res_5_block{n}_conv_proj.weight"],
                         w[f"res_5_block{n}_conv_proj.bias"])

    # --- Stage 7: conv_6 (MERGED: sep split) ---
    # Merge sep_1 + sep_2
    sep1_nhwc = w["conv_6sep_1.weight"]   # [1,1,128,256]
    sep2_nhwc = w["conv_6sep_2.weight"]   # [1,1,128,256]
    sep_merged_nhwc = np.concatenate([sep1_nhwc, sep2_nhwc], axis=3)  # [1,1,128,512]
    sep1_b = w["conv_6sep_1.bias"]
    sep2_b = w["conv_6sep_2.bias"]
    sep_merged_b = np.concatenate([sep1_b, sep2_b], axis=0)  # [512]

    _set_conv_weight(model.conv6_sep, sep_merged_nhwc, sep_merged_b)
    # Merge PReLU alphas 313 + 314
    alpha_313_314 = np.concatenate([w["313"].flatten(), w["314"].flatten()], axis=0)  # [512]
    model.prelu_8.weight.data.copy_(torch.from_numpy(alpha_313_314.astype(np.float32)))

    # DW conv and fc1 are unchanged
    _set_dw_weight(model.conv6_dw, w["conv_6dw7_7.weight"], w["conv_6dw7_7.bias"])
    _set_conv_weight(model.fc1, w["fc1.weight"], w["fc1.bias"])


# =============================================================================
# 5. Verification
# =============================================================================

def count_params(model: nn.Module) -> int:
    """Count total parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def verify_model(model: MobileFaceNet):
    """Verify model structure and output."""
    model.eval()

    print("\n" + "=" * 70)
    print("MODEL VERIFICATION")
    print("=" * 70)

    # Param count
    n_params = count_params(model)
    print(f"  Parameter count: {n_params:,} (expected: 1,190,338)")
    if n_params == 1190338:
        print("  => MATCH")
    else:
        print(f"  => MISMATCH (diff: {n_params - 1190338:+,})")

    # Layer-by-layer param counts
    print("\n  Layer parameter breakdown:")
    total_conv = 0
    total_prelu = 0
    for name, param in model.named_parameters():
        n = param.numel()
        if 'prelu' in name:
            total_prelu += n
        else:
            total_conv += n
        if n > 10000:
            print(f"    {name}: {param.shape} = {n:,}")
    print(f"  Conv params: {total_conv:,}, PReLU params: {total_prelu:,}")
    print(f"  Total: {total_conv + total_prelu:,}")

    # Test forward pass
    torch.manual_seed(42)
    dummy_input = torch.rand(1, 3, 112, 112) * 255.0  # raw pixels [0, 255]
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n  Forward pass:")
    print(f"    Input shape:  {dummy_input.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Output min:   {output.min().item():.6f}")
    print(f"    Output max:   {output.max().item():.6f}")
    print(f"    Output mean:  {output.mean().item():.6f}")
    print(f"    Output std:   {output.std().item():.6f}")
    print(f"    Output L2:    {output.norm().item():.4f}")

    # Cosine similarity between two different inputs
    torch.manual_seed(42)
    x1 = torch.rand(1, 3, 112, 112) * 255.0
    torch.manual_seed(43)
    x2 = torch.rand(1, 3, 112, 112) * 255.0
    with torch.no_grad():
        e1 = model(x1).flatten()
        e2 = model(x2).flatten()
        e1_n = e1 / e1.norm()
        e2_n = e2 / e2.norm()
        cos_sim = (e1_n * e2_n).sum().item()

    print(f"\n    Cosine similarity (random input 1 vs 2): {cos_sim:.6f}")
    if 0.1 < cos_sim < 0.9:
        print(f"    => HEALTHY (expected ~0.3-0.7, NOT 0.99)")
    else:
        print(f"    => WARNING: cos_sim={cos_sim:.6f} is outside expected range [0.1, 0.9]")

    # Test with ALL-ZERO input (should not crash)
    with torch.no_grad():
        zero_out = model(torch.zeros(1, 3, 112, 112))
    print(f"\n    Zero-input output L2: {zero_out.norm().item():.4f}")

    # Test with ALL-ONES input
    with torch.no_grad():
        ones_out = model(torch.ones(1, 3, 112, 112) * 255.0)
    print(f"    Full-bright (255) output L2: {ones_out.norm().item():.4f}")

    return output, cos_sim


# =============================================================================
# 6. Compare with existing ONNX model
# =============================================================================

def compare_with_onnx(model: MobileFaceNet):
    """Compare PyTorch output with the existing singlepath ONNX model."""
    onnx_path = os.path.join(OUTPUT_DIR, "mfn_s8_v1_singlepath.onnx")
    if not os.path.exists(onnx_path):
        print("\n  ONNX comparison skipped (singlepath ONNX not found)")
        return

    print("\n" + "=" * 70)
    print("ONNX COMPARISON")
    print("=" * 70)

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping")
        return

    # Create same input
    torch.manual_seed(42)
    pt_input = torch.rand(1, 3, 112, 112) * 255.0  # raw pixels

    # PyTorch output
    model.eval()
    with torch.no_grad():
        pt_out = model(pt_input).flatten().numpy()

    # ONNX output (singlepath model also has built-in normalization, expects raw pixels)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_inp_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {onnx_inp_name: pt_input.numpy().astype(np.float32)})[0].flatten()

    # Compare
    max_diff = np.abs(pt_out - onnx_out).max()
    mean_diff = np.abs(pt_out - onnx_out).mean()
    cos_sim = np.dot(pt_out, onnx_out) / (np.linalg.norm(pt_out) * np.linalg.norm(onnx_out))

    print(f"  PyTorch output L2: {np.linalg.norm(pt_out):.4f}")
    print(f"  ONNX output L2:    {np.linalg.norm(onnx_out):.4f}")
    print(f"  Max absolute diff: {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")
    print(f"  Cosine similarity: {cos_sim:.10f}")

    if max_diff < 1e-4:
        print("  => OUTPUTS MATCH (within tolerance)")
    else:
        print(f"  => OUTPUTS DIFFER (max_diff={max_diff}), showing first 10 values:")
        print(f"    PT:   {pt_out[:10]}")
        print(f"    ONNX: {onnx_out[:10]}")


# =============================================================================
# 7. Export to ONNX
# =============================================================================

def export_to_onnx(model: MobileFaceNet, output_path: str):
    """Export PyTorch model to ONNX at opset 14 (first opset with full PRelu support)."""
    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )

    # Verify exported ONNX
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(output_path)
    n_params = sum(
        int(np.prod(onnx.numpy_helper.to_array(init).shape))
        for init in onnx_model.graph.initializer
    )

    # Determine actual opset
    actual_opset = onnx_model.opset_import[0].version
    print(f"  Exported: {output_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  ONNX params: {n_params:,}")
    print(f"  ONNX nodes:  {len(onnx_model.graph.node)}")
    print(f"  ONNX opset:  {actual_opset}")
    print(f"  ONNX checker: PASS")
    return onnx_model


def export_relu_variant(prelu_onnx_path: str, relu_onnx_path: str):
    """Convert PReLU ONNX model to ReLU by replacing PRelu ops with Relu."""
    import onnx
    from onnx import helper

    model = onnx.load(prelu_onnx_path)
    graph = model.graph

    prelu_nodes = [n for n in graph.node if n.op_type == "PRelu"]
    alpha_names = set()
    for n in prelu_nodes:
        alpha_names.add(n.input[1])

    # Replace PReLU with ReLU
    for node in graph.node:
        if node.op_type == "PRelu":
            node.op_type = "Relu"
            del node.input[1:]

    # Remove alpha initializers
    new_inits = [i for i in graph.initializer if i.name not in alpha_names]
    removed = len(graph.initializer) - len(new_inits)
    del graph.initializer[:]
    graph.initializer.extend(new_inits)

    # Clean inputs
    for node in graph.node:
        node.input[:] = [inp for inp in node.input if inp not in alpha_names]

    # Preserve original opset (don't let onnx.save upgrade it)
    onnx.checker.check_model(model)
    onnx.save(model, relu_onnx_path)

    file_size = os.path.getsize(relu_onnx_path)
    n_params = sum(
        int(np.prod(onnx.numpy_helper.to_array(init).shape))
        for init in model.graph.initializer
    )

    print(f"\n  ReLU variant exported: {relu_onnx_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  ONNX params: {n_params:,}")
    print(f"  Replaced {len(prelu_nodes)} PReLU → ReLU, removed {removed} alpha initializers")


# =============================================================================
# 8. Final Validation
# =============================================================================

def final_validation():
    """Run final ONNX validation: cos_sim should be reasonable (~0.3-0.7)."""
    print("\n" + "=" * 70)
    print("FINAL ONNX VALIDATION")
    print("=" * 70)

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping")
        return

    for label, path in [("PReLU", ONNX_PRELU_PATH), ("ReLU", ONNX_RELU_PATH)]:
        if not os.path.exists(path):
            print(f"  {label} ONNX not found, skipping")
            continue

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name

        np.random.seed(42)
        x1 = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0)
        np.random.seed(43)
        x2 = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0)

        e1 = sess.run(None, {inp_name: x1})[0].flatten()
        e2 = sess.run(None, {inp_name: x2})[0].flatten()

        e1_n = e1 / np.linalg.norm(e1)
        e2_n = e2 / np.linalg.norm(e2)
        cos_sim = np.dot(e1_n, e2_n)

        print(f"\n  {label} ONNX:")
        print(f"    Output shape: {sess.get_outputs()[0].shape}")
        print(f"    Output L2 (x1): {np.linalg.norm(e1):.4f}")
        print(f"    Output L2 (x2): {np.linalg.norm(e2):.4f}")
        print(f"    Cosine similarity: {cos_sim:.6f}")

        if 0.1 < cos_sim < 0.9:
            print(f"    => HEALTHY (expected ~0.3-0.7)")
        else:
            print(f"    => WARNING: cos_sim={cos_sim:.6f} outside expected range")

        # Also test that the model differentiates between images
        # Compute cos_sim on more pairs
        cos_sims = []
        for s in range(5):
            np.random.seed(100 + s * 2)
            a = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0)
            np.random.seed(101 + s * 2)
            b = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0)
            ea = sess.run(None, {inp_name: a})[0].flatten()
            eb = sess.run(None, {inp_name: b})[0].flatten()
            ea_n = ea / np.linalg.norm(ea)
            eb_n = eb / np.linalg.norm(eb)
            cos_sims.append(np.dot(ea_n, eb_n))

        cos_sims = np.array(cos_sims)
        print(f"    Cos sim across 5 pairs: mean={cos_sims.mean():.4f}, "
              f"std={cos_sims.std():.4f}, min={cos_sims.min():.4f}, max={cos_sims.max():.4f}")


# =============================================================================
# 9. Main
# =============================================================================

def main():
    print("=" * 70)
    print("MobileFaceNet PyTorch Builder (from .espdl weights)")
    print("=" * 70)

    # Step 1: Parse .espdl
    print("\n[1/7] Parsing .espdl file...")
    fb_model = parse_espdl(ESPDL_PATH)
    graph = fb_model.Graph()
    n_weights = graph.InitializerLength()
    n_nodes = graph.NodeLength()
    print(f"  Found {n_nodes} graph nodes, {n_weights} initializer tensors")

    # Step 2: Extract and dequantize weights
    print("\n[2/7] Extracting and dequantizing weights...")
    weights = extract_all_weights(fb_model)
    print(f"  Extracted {len(weights)} weight tensors")

    # Show weight name summary
    weight_keys = sorted(weights.keys())
    print(f"\n  Weight names ({len(weight_keys)} total):")
    for k in weight_keys:
        v = weights[k]
        print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}, "
              f"range=[{v.min():.4f}, {v.max():.4f}]")

    # Step 3: Build PyTorch model
    print("\n[3/7] Building PyTorch MobileFaceNet...")
    model = MobileFaceNet()

    # Step 4: Load weights
    print("\n[4/7] Loading weights into PyTorch model...")
    load_weights(model, weights)
    print("  All weights loaded successfully")

    # Step 5: Verify
    print("\n[5/7] Verifying model...")
    pt_output, pt_cos_sim = verify_model(model)

    # Compare with existing ONNX
    compare_with_onnx(model)

    # Step 6: Save checkpoint and export ONNX
    print("\n[6/7] Saving checkpoint and exporting ONNX...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save PyTorch checkpoint
    torch.save(model.state_dict(), PTH_PATH)
    pth_size = os.path.getsize(PTH_PATH)
    print(f"  PyTorch checkpoint: {PTH_PATH}")
    print(f"    Size: {pth_size:,} bytes ({pth_size / 1024 / 1024:.2f} MB)")

    # Export ONNX PReLU variant
    onnx_model = export_to_onnx(model, ONNX_PRELU_PATH)

    # Export ONNX ReLU variant
    export_relu_variant(ONNX_PRELU_PATH, ONNX_RELU_PATH)

    # Step 7: Final validation
    print("\n[7/7] Final ONNX validation...")
    final_validation()

    # ===================================================================
    # EVIDENCE
    # ===================================================================
    print("\n" + "=" * 70)
    print("EVIDENCE")
    print("=" * 70)

    import onnx
    import onnxruntime as ort

    n_params = count_params(model)

    print(f"\n  Parameter count: {n_params:,}")
    print(f"  Expected:         1,190,338")
    print(f"  Match:            {'YES' if n_params == 1190338 else f'NO (diff={n_params - 1190338:+,})'}")

    print(f"\n  PyTorch Model:")
    print(f"    Architecture:   MobileFaceNet (merged single-path)")
    print(f"    Input:          [N, 3, 112, 112] raw pixels [0, 255]")
    print(f"    Output:         [N, 512, 1, 1]")
    print(f"    Normalization:  x / 127.5 - 1.0 (inside forward())")
    print(f"    Activations:    PReLU (per-channel)")
    print(f"    BatchNorm:      None (folded into Conv)")
    print(f"    L2 Norm:        None (done externally by esp-dl)")

    # ONNX file verification
    for path, label in [(ONNX_PRELU_PATH, "PReLU"), (ONNX_RELU_PATH, "ReLU")]:
        if os.path.exists(path):
            m = onnx.load(path)
            n = sum(int(np.prod(onnx.numpy_helper.to_array(i).shape)) for i in m.graph.initializer)
            nodes = len(m.graph.node)
            size = os.path.getsize(path)
            ops = {}
            for nd in m.graph.node:
                ops[nd.op_type] = ops.get(nd.op_type, 0) + 1

            print(f"\n  {label} ONNX: {path}")
            print(f"    File size:    {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
            print(f"    Parameters:   {n:,}")
            print(f"    Graph nodes:  {nodes}")
            print(f"    Operators:    {ops}")

    # Final runtime test
    print(f"\n  ONNX Runtime Test (PReLU):")
    sess = ort.InferenceSession(ONNX_PRELU_PATH, providers=["CPUExecutionProvider"])
    np.random.seed(42)
    test_x = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0)
    test_out = sess.run(None, {"input": test_x})[0]
    print(f"    Input shape:  {test_x.shape} (raw [0, 255] pixels)")
    print(f"    Output shape: {test_out.shape}")
    print(f"    Output min:   {test_out.min():.6f}")
    print(f"    Output max:   {test_out.max():.6f}")
    print(f"    Output mean:  {test_out.mean():.6f}")
    print(f"    Output L2:    {np.linalg.norm(test_out.flatten()):.4f}")

    # Cosine similarity test with raw [0,255] inputs
    np.random.seed(42)
    a = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0)
    np.random.seed(99)
    b = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0)
    ea = sess.run(None, {"input": a})[0].flatten()
    eb = sess.run(None, {"input": b})[0].flatten()
    ea_n = ea / np.linalg.norm(ea)
    eb_n = eb / np.linalg.norm(eb)
    cos = np.dot(ea_n, eb_n)
    print(f"\n  Cosine similarity (random pairs, raw [0,255] inputs):")
    print(f"    cos_sim = {cos:.6f}")
    if 0.1 < cos < 0.9:
        print(f"    VERDICT: HEALTHY (reasonable diversity, NOT collapsed)")
    else:
        print(f"    VERDICT: WARNING - cos_sim={cos:.6f} (expected ~0.3-0.7)")

    # Compare with the broken ONNX model
    broken_path = os.path.join(OUTPUT_DIR, "mfn_s8_v1_reconstructed.onnx")
    if os.path.exists(broken_path):
        sess_broken = ort.InferenceSession(broken_path, providers=["CPUExecutionProvider"])
        inp_b = sess_broken.get_inputs()[0].name
        ea_broken = sess_broken.run(None, {inp_b: a})[0].flatten()
        eb_broken = sess_broken.run(None, {inp_b: b})[0].flatten()
        ea_bn = ea_broken / np.linalg.norm(ea_broken)
        eb_bn = eb_broken / np.linalg.norm(eb_broken)
        cos_broken = np.dot(ea_bn, eb_bn)
        print(f"\n  Broken reconstructed ONNX cos_sim: {cos_broken:.6f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
