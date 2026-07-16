#!/usr/bin/env python3
"""
Face embedding computation that exactly mirrors the Himax firmware pipeline.

Firmware reference: EPII_CM55M_APP_S/app/scenario_app/sscma_face/cvapp_face_embedding.cpp

Pipeline:
  1. Load image, convert to BGR planar (matching camera output format)
  2. Direct resize BGR planar → 160x160 RGB interleaved (no letterbox)
  3. Quantize to INT8: dst = src + input_zero_point
  4. SCRFD_500M_KPS inference → detect face + 5-point landmarks
  5. Decode: dequantize, distance-based bbox, anchor-corner landmarks
  6. NMS (intra-stride IoU + cross-stride center suppression)
  7. Face alignment: similarity transform from eyes → ArcFace canonical
  8. Affine warp (backward bilinear): BGR planar src → 112x112 RGB dst
  9. Quantize to INT8 using the embedding model's input scale/zero-point
  10. MobileFaceNet/GhostFaceNet inference → embedding (INT8 output)
  11. Dequantize: float = (int8_val - output_zp) * output_scale
  12. L2 normalize

Usage:
  python compute_embedding.py <image_path> [--output json] [--debug]
  python compute_embedding.py <image1> <image2> --compare
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants — must match common_config.h and face_alignment.h
# ---------------------------------------------------------------------------

FD_INPUT_W = 160
FD_INPUT_H = 160
FD_INPUT_C = 3

EMB_INPUT_W = 112
EMB_INPUT_H = 112
EMB_INPUT_C = 3
EMB_OUTPUT_DIM = 128

SCRFD_NUM_STRIDES = 3
SCRFD_NUM_ANCHORS = 2
SCRFD_NUM_LANDMARKS = 5
STRIDES = (8, 16, 32)

FACE_CONF_THRESHOLD = 0.70
FACE_NMS_THRESHOLD = 0.40
MIN_FACE_SIZE = 40
CENTER_DIST_THRESH_RATIO = 0.3
MAX_FACE_RATIO = 0.6
ALLOW_CENTER_CROP_FALLBACK = False

# ArcFace canonical reference landmarks for 112x112 (face_alignment.c:21-27)
REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],   # Left eye
        [73.5318, 51.5014],   # Right eye
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Left mouth corner
        [70.7299, 92.2041],   # Right mouth corner
    ],
    dtype=np.float32,
)

LM_LEFT_EYE = 0
LM_RIGHT_EYE = 1
LM_NOSE = 2
LM_LEFT_MOUTH = 3
LM_RIGHT_MOUTH = 4

# ---------------------------------------------------------------------------
# Model paths (relative to this script or absolute)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# SCRFD INT8 TFLite — input zp=-128, matches firmware: dst = src + zp = src - 128
SCRFD_TFLITE = SCRIPT_DIR / "scrfd" / "models" / "scrfd_500m_kps_int8.tflite"

# MobileFaceNet-128D float32 TFLite.
MOBILEFACENET_FLOAT32_TFLITE = (
    SCRIPT_DIR / "foamliu_mobilefacenet_128d" / "foamliu_mobilefacenet_128d_float32.tflite"
)

# Fallback: 512D GhostFaceNet INT8 TFLite (firmware uses only first 128 dims)
GHOSTFACENET_INT8_TFLITE = SCRIPT_DIR / "ghostfacenet_fixed_int8.tflite"

# MobileFaceNet-128D QAT INT8 TFLite (zp=-1 input, zp=10 output)
MOBILEFACENET_INT8_TFLITE = (
    SCRIPT_DIR / "foamliu_mobilefacenet_128d" / "foamliu_mobilefacenet_128d_qat_int8.tflite"
)

# DEPLOYED model — the pre-Vela INT8 twin of what the device flashes at 0x510000
# (qat_distill_v2_relu6_128d). Use this so PC embeddings match the device
# (on-device Vela reproduces this model to cos 0.99). Input zp=-1 scale 0.00784
# (same as firmware pixel-129), output 128D int8 zp=18 scale 0.01266.
DEPLOYED_EMBEDDING_TFLITE = (
    SCRIPT_DIR / "qat_distill_v2_relu6_128d" / "model_128d.int8.tflite"
)

# ONNX fallbacks (float32)
SCRFD_ONNX = SCRIPT_DIR / "scrfd" / "models" / "scrfd_500m_kps.onnx"
MOBILEFACENET_ONNX = SCRIPT_DIR / "foamliu_mobilefacenet_128d" / "foamliu_mobilefacenet_128d.onnx"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tflite_model(path: str):
    """Load a TFLite model. Prefer tflite_runtime, fall back to tensorflow."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=path)
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path)


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """Match C roundf/TFLite-style integer quantization."""
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5))


def quantize_embedding_input_rgb(
    aligned_face: np.ndarray, input_scale: float, input_zp: int
) -> np.ndarray:
    """Quantize 112x112 RGB input exactly like firmware.

    MobileFaceNet/ArcFace expects RGB pixels normalized to [-1, 1]:
        real = pixel / 127.5 - 1
        q = round(real / input_scale + input_zero_point)
    """
    if input_zp == -1 and abs(float(input_scale) - (1.0 / 127.5)) < 1e-5:
        a = aligned_face.astype(np.int32)
        q = np.where(a > 128, a - 128, a - 129)
        return np.clip(q, -128, 127).astype(np.int8)

    if input_scale <= 0:
        q = aligned_face.astype(np.int32) - 128
    else:
        real = aligned_face.astype(np.float32) / 127.5 - 1.0
        q = round_half_away_from_zero(real / float(input_scale) + int(input_zp))
    return np.clip(q, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def image_to_bgr_planar(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to BGR planar format matching the camera output.

    Camera format (cisdp_cfg.h): 3 separate planes in B, G, R order.
    Each plane is W*H bytes.

    Returns: uint8 ndarray of shape (3, H, W) in BGR plane order.
    """
    rgb = pil_image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)  # (H, W, 3) RGB interleaved

    # Convert RGB → BGR and interleaved → planar
    # BGR planar: [B_plane | G_plane | R_plane]
    b = arr[:, :, 2]  # B channel
    g = arr[:, :, 1]  # G channel
    r = arr[:, :, 0]  # R channel

    return np.stack([b, g, r], axis=0)  # (3, H, W)


def bgr_planar_to_rgb_interleaved(bgr_planar: np.ndarray) -> np.ndarray:
    """
    Convert BGR planar to RGB interleaved.
    Input:  (3, H, W) BGR planar
    Output: (H, W, 3) RGB interleaved
    """
    b, g, r = bgr_planar[0], bgr_planar[1], bgr_planar[2]
    return np.stack([r, g, b], axis=-1)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Step 1+2: Direct resize BGR planar → 160x160 RGB interleaved
# ---------------------------------------------------------------------------

def direct_resize_bgr_planar_to_rgb(
    bgr_planar: np.ndarray, dst_w: int, dst_h: int
) -> np.ndarray:
    """
    Replicates hx_lib_image_resize_BGR8U3C_to_RGB24_helium().

    Input:  (3, H, W) uint8, BGR planar
    Output: (dst_h, dst_w, 3) uint8, RGB interleaved

    Uses OpenCV resize on each channel for speed, then reorders.
    The firmware uses a HW accelerator but bilinear resize is equivalent.
    """
    import cv2

    b, g, r = bgr_planar[0], bgr_planar[1], bgr_planar[2]

    b_resized = cv2.resize(b, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    g_resized = cv2.resize(g, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    r_resized = cv2.resize(r, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    # Reorder to RGB interleaved
    return np.stack([r_resized, g_resized, b_resized], axis=-1)


# ---------------------------------------------------------------------------
# Step 3: Quantize RGB uint8 → INT8 for SCRFD input
# ---------------------------------------------------------------------------

def quantize_uint8_to_int8(rgb: np.ndarray, zero_point: int) -> np.ndarray:
    """
    Replicates firmware: dst[i] = clamp(src[i] + zp, -128, 127)
    """
    val = rgb.astype(np.int32) + zero_point
    return np.clip(val, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Step 4-5: SCRFD post-processing
# ---------------------------------------------------------------------------

def dequantize_int8(data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Replicates firmware dequantize(): (value - zp) * scale"""
    return (data.astype(np.float32) - zero_point) * scale


def scrfd_decode_and_nms(
    score_tensors: list,
    bbox_tensors: list,
    kps_tensors: list,
    score_scales: list,
    score_zps: list,
    bbox_scales: list,
    bbox_zps: list,
    kps_scales: list,
    kps_zps: list,
    input_w: int,
    input_h: int,
    img_w: int,
    img_h: int,
    score_thresh: float,
    nms_thresh: float,
) -> list[dict]:
    """
    Exact replica of scrfd_detect() + scrfd_nms() + cross_stride_suppress().

    Returns list of face dicts, each with: bbox(x,y,w,h), score, landmarks[5](x,y).
    Coordinates are in ORIGINAL image space.
    """
    scale_x = img_w / input_w
    scale_y = img_h / input_h

    all_dets = []  # list of (stride_idx, det_dict)

    for s in range(SCRFD_NUM_STRIDES):
        stride = STRIDES[s]
        grid_h = input_h // stride
        grid_w = input_w // stride

        # Dequantize
        score_f = dequantize_int8(score_tensors[s], score_scales[s], score_zps[s])
        bbox_f = dequantize_int8(bbox_tensors[s], bbox_scales[s], bbox_zps[s])
        kps_f = dequantize_int8(kps_tensors[s], kps_scales[s], kps_zps[s])

        # Vela output format: [num_elements, channels] = [H*W*A, C]
        # score: [N, 1], bbox: [N, 4], kps: [N, 10]
        score_f = score_f.flatten()

        for h in range(grid_h):
            for w in range(grid_w):
                for a in range(SCRFD_NUM_ANCHORS):
                    row = (h * grid_w + w) * SCRFD_NUM_ANCHORS + a

                    score = score_f[row]
                    score = min(1.0, max(0.0, score))

                    if score < score_thresh:
                        continue

                    # Distance-based bbox decoding (scrfd_postprocessing.cc:300-321)
                    d_left = bbox_f[row, 0]
                    d_top = bbox_f[row, 1]
                    d_right = bbox_f[row, 2]
                    d_bottom = bbox_f[row, 3]

                    # Anchor corner in model space
                    cx = w * stride
                    cy = h * stride

                    x1 = cx - d_left * stride
                    y1 = cy - d_top * stride
                    x2 = cx + d_right * stride
                    y2 = cy + d_bottom * stride

                    x1 = max(0.0, min(float(input_w), x1))
                    y1 = max(0.0, min(float(input_h), y1))
                    x2 = max(0.0, min(float(input_w), x2))
                    y2 = max(0.0, min(float(input_h), y2))

                    # Map to original image space: orig = (model - pad) * scale
                    orig_x1 = x1 * scale_x
                    orig_x2 = x2 * scale_x
                    orig_y1 = y1 * scale_y
                    orig_y2 = y2 * scale_y

                    orig_x1 = max(0.0, min(float(img_w), orig_x1))
                    orig_y1 = max(0.0, min(float(img_h), orig_y1))
                    orig_x2 = max(0.0, min(float(img_w), orig_x2))
                    orig_y2 = max(0.0, min(float(img_h), orig_y2))

                    # Landmarks: anchor corner + kp_dx * stride
                    landmarks = []
                    for k in range(SCRFD_NUM_LANDMARKS):
                        kp_dx = kps_f[row, k * 2]
                        kp_dy = kps_f[row, k * 2 + 1]
                        lm_x = w * stride + kp_dx * stride
                        lm_y = h * stride + kp_dy * stride
                        lm_x = lm_x * scale_x
                        lm_y = lm_y * scale_y
                        landmarks.append((lm_x, lm_y))

                    det = {
                        "bbox": (orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1),
                        "score": float(score),
                        "landmarks": landmarks,
                        "stride_idx": s,
                        "bbox_model": (x1, y1, x2 - x1, y2 - y1),
                    }
                    all_dets.append((s, det))

    # Intra-stride NMS (scrfd_postprocessing.cc:465-483)
    for s in range(SCRFD_NUM_STRIDES):
        stride_dets = [d for si, d in all_dets if si == s and d["score"] > 0]
        stride_dets.sort(key=lambda d: d["score"], reverse=True)

        for i in range(len(stride_dets)):
            if stride_dets[i]["score"] <= 0:
                continue
            for j in range(i + 1, len(stride_dets)):
                if stride_dets[j]["score"] <= 0:
                    continue
                iou = _compute_iou(stride_dets[i]["bbox"], stride_dets[j]["bbox"])
                if iou > nms_thresh:
                    stride_dets[j]["score"] = 0.0

    # Collect surviving dets
    dets = [d for _, d in all_dets if d["score"] > 0]

    # Cross-stride suppression (scrfd_postprocessing.cc:105-124)
    if len(dets) > 1:
        dets.sort(key=lambda d: d["bbox"][2] * d["bbox"][3])  # sort by area asc
        for i in range(len(dets)):
            if dets[i]["score"] <= 0:
                continue
            for j in range(i + 1, len(dets)):
                if dets[j]["score"] <= 0:
                    continue
                if _is_same_face_by_center(dets[i], dets[j], CENTER_DIST_THRESH_RATIO):
                    dets[j]["score"] = 0.0

    dets = [d for d in dets if d["score"] > 0]

    # Filter oversized faces (scrfd_postprocessing.cc:444-455)
    max_w = img_w * MAX_FACE_RATIO
    max_h = img_h * MAX_FACE_RATIO
    dets = [d for d in dets if d["bbox"][2] <= max_w and d["bbox"][3] <= max_h]

    return dets


def _compute_iou(bbox_a, bbox_b):
    """IoU between (x, y, w, h) boxes."""
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _is_same_face_by_center(det_a, det_b, thresh_ratio):
    """scrfd_postprocessing.cc:92-97"""
    ax, ay, aw, ah = det_a["bbox"]
    bx, by, bw, bh = det_b["bbox"]
    cx_a = ax + aw / 2
    cy_a = ay + ah / 2
    cx_b = bx + bw / 2
    cy_b = by + bh / 2
    dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    ref_size = min(min(aw, ah), min(bw, bh))
    return dist < ref_size * thresh_ratio


# ---------------------------------------------------------------------------
# Step 6: Get best face (scrfd_get_best_face logic without temporal smoothing)
# ---------------------------------------------------------------------------

def get_best_face(dets: list[dict], min_size: int) -> Optional[dict]:
    """Select highest-score face meeting min size requirement."""
    best = None
    best_score = -1.0
    for d in dets:
        if d["score"] <= 0:
            continue
        if d["bbox"][2] < min_size or d["bbox"][3] < min_size:
            continue
        if d["score"] > best_score:
            best_score = d["score"]
            best = d
    return best


# ---------------------------------------------------------------------------
# Step 7: Face alignment — compute similarity transform
# ---------------------------------------------------------------------------

def compute_face_alignment(landmarks: list) -> np.ndarray:
    """
    Exact replica of compute_face_alignment() in face_alignment.c:29-108.

    Uses eye positions to compute a similarity transform (rotation + uniform
    scale + translation) that maps detected landmarks to ArcFace reference.

    Returns 2x3 affine matrix M where:
      x' = M[0,0]*x + M[0,1]*y + M[0,2]
      y' = M[1,0]*x + M[1,1]*y + M[1,2]
    """
    src_left = np.array(landmarks[LM_LEFT_EYE], dtype=np.float32)
    src_right = np.array(landmarks[LM_RIGHT_EYE], dtype=np.float32)

    src_center = (src_left + src_right) / 2.0
    src_vec = src_right - src_left
    src_dist = float(np.linalg.norm(src_vec))

    dst_left = REFERENCE_LANDMARKS[LM_LEFT_EYE]
    dst_right = REFERENCE_LANDMARKS[LM_RIGHT_EYE]

    dst_center = (dst_left + dst_right) / 2.0
    dst_vec = dst_right - dst_left
    dst_dist = float(np.linalg.norm(dst_vec))

    scale = dst_dist / (src_dist + 1e-6)

    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
    angle = dst_angle - src_angle

    cos_a = np.cos(angle) * scale
    sin_a = np.sin(angle) * scale

    # M such that: x' = cos_a*(x - cx_src) - sin_a*(y - cy_src) + cx_dst
    #              y' = sin_a*(x - cx_src) + cos_a*(y - cy_src) + cy_dst
    M = np.array(
        [
            [cos_a, -sin_a, dst_center[0] - cos_a * src_center[0] + sin_a * src_center[1]],
            [sin_a, cos_a, dst_center[1] - sin_a * src_center[0] - cos_a * src_center[1]],
        ],
        dtype=np.float32,
    )
    return M


def invert_affine(M: np.ndarray) -> np.ndarray:
    """
    Exact replica of invert_affine_transform() in face_alignment.c:110-145.
    """
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]

    det = a * e - b * d
    inv_det = 1.0 / (det + 1e-8)

    inv = np.zeros((2, 3), dtype=np.float32)
    inv[0, 0] = e * inv_det
    inv[0, 1] = -b * inv_det
    inv[0, 2] = (b * f - e * c) * inv_det
    inv[1, 0] = -d * inv_det
    inv[1, 1] = a * inv_det
    inv[1, 2] = (d * c - a * f) * inv_det
    return inv


# ---------------------------------------------------------------------------
# Step 8: Affine warp — BGR planar src → 112x112 RGB interleaved dst
# ---------------------------------------------------------------------------

def apply_face_alignment(
    bgr_planar: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """
    Exact replica of apply_face_alignment() in face_alignment.c:148-239.

    Input:  (3, H, W) uint8, BGR planar
    Output: (112, 112, 3) uint8, RGB interleaved

    Backward mapping with bilinear interpolation. Out-of-bounds → black.
    """
    src_h, src_w = bgr_planar.shape[1], bgr_planar.shape[2]
    b_plane, g_plane, r_plane = bgr_planar[0], bgr_planar[1], bgr_planar[2]

    inv_M = invert_affine(M)

    dst = np.zeros((EMB_INPUT_H, EMB_INPUT_W, 3), dtype=np.uint8)

    # Compute all source coordinates at once (vectorized)
    dy_grid, dx_grid = np.mgrid[0:EMB_INPUT_H, 0:EMB_INPUT_W]
    sx = inv_M[0, 0] * dx_grid + inv_M[0, 1] * dy_grid + inv_M[0, 2]
    sy = inv_M[1, 0] * dx_grid + inv_M[1, 1] * dy_grid + inv_M[1, 2]

    ix = np.floor(sx).astype(np.int32)
    iy = np.floor(sy).astype(np.int32)

    fx = sx - ix
    fy = sy - iy

    # Valid mask: interpolation kernel within source bounds
    valid = (ix >= 0) & (ix < src_w - 1) & (iy >= 0) & (iy < src_h - 1)

    w00 = (1.0 - fx) * (1.0 - fy)
    w01 = fx * (1.0 - fy)
    w10 = (1.0 - fx) * fy
    w11 = fx * fy

    # For valid pixels, do bilinear interpolation
    idx00 = iy * src_w + ix
    idx01 = idx00 + 1
    idx10 = idx00 + src_w
    idx11 = idx10 + 1

    v = valid
    dst[dy_grid[v], dx_grid[v], 0] = np.clip(
        w00[v] * r_plane.flat[idx00[v]]
        + w01[v] * r_plane.flat[idx01[v]]
        + w10[v] * r_plane.flat[idx10[v]]
        + w11[v] * r_plane.flat[idx11[v]]
        + 0.5,
        0, 255,
    ).astype(np.uint8)

    dst[dy_grid[v], dx_grid[v], 1] = np.clip(
        w00[v] * g_plane.flat[idx00[v]]
        + w01[v] * g_plane.flat[idx01[v]]
        + w10[v] * g_plane.flat[idx10[v]]
        + w11[v] * g_plane.flat[idx11[v]]
        + 0.5,
        0, 255,
    ).astype(np.uint8)

    dst[dy_grid[v], dx_grid[v], 2] = np.clip(
        w00[v] * b_plane.flat[idx00[v]]
        + w01[v] * b_plane.flat[idx01[v]]
        + w10[v] * b_plane.flat[idx10[v]]
        + w11[v] * b_plane.flat[idx11[v]]
        + 0.5,
        0, 255,
    ).astype(np.uint8)

    return dst


def center_crop_resize_rgb(bgr_planar: np.ndarray) -> np.ndarray:
    """
    Fallback for already-cropped benchmark faces when detector/landmarks fail.

    This is intentionally opt-in and is not firmware-equivalent. It keeps CFP-FP
    profile crops usable for embedding evaluation/training instead of treating
    detector failure as recognition failure.
    """
    src_h, src_w = bgr_planar.shape[1], bgr_planar.shape[2]
    side = min(src_w, src_h)
    x0 = max((src_w - side) // 2, 0)
    y0 = max((src_h - side) // 2, 0)
    crop_bgr = bgr_planar[:, y0:y0 + side, x0:x0 + side]
    crop_rgb = np.transpose(crop_bgr[::-1], (1, 2, 0))
    return np.asarray(
        Image.fromarray(crop_rgb).resize((EMB_INPUT_W, EMB_INPUT_H), Image.BILINEAR),
        dtype=np.uint8,
    )


# ---------------------------------------------------------------------------
# Step 11: L2 normalization
# ---------------------------------------------------------------------------

def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """
    Exact replica of normalize_embedding() in face_embedding_protocol.c:131-145.
    """
    norm = float(np.linalg.norm(embedding))
    if norm > 1e-6:
        return embedding / norm
    return embedding


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class FaceEmbeddingPipeline:
    """Complete face embedding pipeline matching the Himax firmware."""

    def __init__(self, scrfd_model: str, embedding_model: str, backend: str = "tflite"):
        self.backend = backend
        if backend == "tflite":
            self._init_tflite(scrfd_model, embedding_model)
        elif backend == "onnx":
            self._init_onnx(scrfd_model, embedding_model)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_tflite(self, scrfd_path: str, emb_path: str):
        print(f"Loading SCRFD: {scrfd_path}")
        self.fd_interp = load_tflite_model(scrfd_path)
        self.fd_interp.allocate_tensors()
        self.fd_input_idx = self.fd_interp.get_input_details()[0]["index"]
        self.fd_input_dtype = self.fd_interp.get_input_details()[0]["dtype"]
        fd_input_q = self.fd_interp.get_input_details()[0].get("quantization_parameters", {})
        self.fd_input_zp = fd_input_q.get("zero_point", np.array([-128]))
        if hasattr(self.fd_input_zp, "__len__"):
            self.fd_input_zp = int(np.asarray(self.fd_input_zp).flat[0])

        # Map outputs
        self.fd_score_tensors = []
        self.fd_bbox_tensors = []
        self.fd_kps_tensors = []
        self.fd_score_scales = []
        self.fd_score_zps = []
        self.fd_bbox_scales = []
        self.fd_bbox_zps = []
        self.fd_kps_scales = []
        self.fd_kps_zps = []

        output_details = self.fd_interp.get_output_details()
        for od in output_details:
            shape = od["shape"]
            name = od["name"]
            qp = od.get("quantization_parameters", {})
            scale = float(np.asarray(qp.get("scales", [1.0])).flat[0])
            zp = int(np.asarray(qp.get("zero_points", [0])).flat[0])

            # Determine stride and type from shape
            if len(shape) == 4:  # [1, H, W, C]
                h, w, c = shape[1], shape[2], shape[3]
                if h == 20 and w == 20:
                    s_idx = 0
                elif h == 10 and w == 10:
                    s_idx = 1
                elif h == 5 and w == 5:
                    s_idx = 2
                else:
                    continue
                n_elements = h * w * SCRFD_NUM_ANCHORS
            elif len(shape) == 2:  # [N, C]
                n_elements = shape[0]
                channels = shape[1]
                if n_elements == 800:
                    s_idx = 0
                elif n_elements == 200:
                    s_idx = 1
                elif n_elements == 50:
                    s_idx = 2
                else:
                    continue
                if channels == 1:
                    pass  # score
                elif channels == 4:
                    pass  # bbox
                elif channels == 10:
                    pass  # kps
                else:
                    continue
            else:
                continue

            # Identify tensor type from channel count
            ch = shape[-1]
            if ch == 1 or (len(shape) == 4 and shape[3] == 2):
                while len(self.fd_score_tensors) <= s_idx:
                    self.fd_score_tensors.append(None)
                    self.fd_score_scales.append(1.0)
                    self.fd_score_zps.append(0)
                self.fd_score_tensors[s_idx] = od["index"]
                self.fd_score_scales[s_idx] = scale
                self.fd_score_zps[s_idx] = zp
            elif ch == 4 or (len(shape) == 4 and shape[3] == 8):
                while len(self.fd_bbox_tensors) <= s_idx:
                    self.fd_bbox_tensors.append(None)
                    self.fd_bbox_scales.append(1.0)
                    self.fd_bbox_zps.append(0)
                self.fd_bbox_tensors[s_idx] = od["index"]
                self.fd_bbox_scales[s_idx] = scale
                self.fd_bbox_zps[s_idx] = zp
            elif ch == 10 or (len(shape) == 4 and shape[3] == 20):
                while len(self.fd_kps_tensors) <= s_idx:
                    self.fd_kps_tensors.append(None)
                    self.fd_kps_scales.append(1.0)
                    self.fd_kps_zps.append(0)
                self.fd_kps_tensors[s_idx] = od["index"]
                self.fd_kps_scales[s_idx] = scale
                self.fd_kps_zps[s_idx] = zp

        print(f"  SCRFD input: {self.fd_interp.get_input_details()[0]['shape']}, "
              f"dtype={self.fd_input_dtype}, zp={self.fd_input_zp}")

        # Load embedding model
        print(f"Loading embedding model: {emb_path}")
        self.emb_interp = load_tflite_model(emb_path)
        self.emb_interp.allocate_tensors()
        self.emb_input_idx = self.emb_interp.get_input_details()[0]["index"]
        self.emb_input_dtype = self.emb_interp.get_input_details()[0]["dtype"]
        emb_in_shape = self.emb_interp.get_input_details()[0]["shape"]
        self.emb_output_idx = self.emb_interp.get_output_details()[0]["index"]
        emb_out_dtype = self.emb_interp.get_output_details()[0]["dtype"]
        emb_out_shape = self.emb_interp.get_output_details()[0]["shape"]
        emb_out_q = self.emb_interp.get_output_details()[0].get("quantization_parameters", {})
        out_scales = np.asarray(emb_out_q.get("scales", [1.0])).flatten()
        out_zps = np.asarray(emb_out_q.get("zero_points", [0])).flatten()
        self.emb_out_scale = float(out_scales[0]) if len(out_scales) > 0 else 1.0
        self.emb_out_zp = int(out_zps[0]) if len(out_zps) > 0 else 0
        print(f"  Embedding model input: {emb_in_shape}, dtype={self.emb_input_dtype}")
        print(f"  Embedding model output: {emb_out_shape}, dtype={emb_out_dtype}, "
              f"scale={self.emb_out_scale}, zp={self.emb_out_zp}")

    def _init_onnx(self, scrfd_path: str, emb_path: str):
        import onnxruntime as ort

        print(f"Loading SCRFD ONNX: {scrfd_path}")
        self.fd_session = ort.InferenceSession(scrfd_path)
        self.fd_input_name = self.fd_session.get_inputs()[0].name
        print(f"  SCRFD input: {self.fd_session.get_inputs()[0].shape}")

        print(f"Loading MobileFaceNet ONNX: {emb_path}")
        self.emb_session = ort.InferenceSession(emb_path)
        self.emb_input_name = self.emb_session.get_inputs()[0].name
        print(f"  MobileFaceNet input: {self.emb_session.get_inputs()[0].shape}")

    def compute(self, image: Image.Image, debug: bool = False) -> dict:
        """
        Run the complete pipeline on a PIL image.

        Returns dict with:
          - embedding: 128D float32 L2-normalized array
          - bbox: (x, y, w, h) in original image space
          - score: detection confidence
          - landmarks: list of 5 (x, y) tuples
          - aligned_face: (112, 112, 3) uint8 RGB image
          - quality: face quality score
          - pose: dict with yaw, pitch, roll
        """
        img_w, img_h = image.size  # PIL: (width, height)

        # Step 1: Convert to BGR planar (camera format)
        bgr_planar = image_to_bgr_planar(image)  # (3, H, W)

        # Step 2: Direct resize BGR planar → 160x160 RGB interleaved
        fd_input_rgb = direct_resize_bgr_planar_to_rgb(bgr_planar, FD_INPUT_W, FD_INPUT_H)

        if self.backend == "tflite":
            embedding = self._run_tflite(bgr_planar, fd_input_rgb, img_w, img_h, debug)
        else:
            embedding = self._run_onnx(bgr_planar, fd_input_rgb, img_w, img_h, debug)

        return embedding

    def _run_tflite(
        self, bgr_planar: np.ndarray, fd_input_rgb: np.ndarray,
        img_w: int, img_h: int, debug: bool
    ) -> dict:
        # Step 3: Quantize to INT8 and run SCRFD
        if self.fd_input_dtype == np.int8:
            fd_input = quantize_uint8_to_int8(fd_input_rgb, self.fd_input_zp)
        else:
            fd_input = fd_input_rgb

        # Add batch dimension [H, W, C] → [1, H, W, C]
        fd_input = fd_input[np.newaxis, ...]

        self.fd_interp.set_tensor(self.fd_input_idx, fd_input)
        self.fd_interp.invoke()

        # Collect output tensors
        score_data = []
        bbox_data = []
        kps_data = []
        for s in range(SCRFD_NUM_STRIDES):
            score_data.append(self.fd_interp.get_tensor(self.fd_score_tensors[s]))
            bbox_data.append(self.fd_interp.get_tensor(self.fd_bbox_tensors[s]))
            kps_data.append(self.fd_interp.get_tensor(self.fd_kps_tensors[s]))

        if debug:
            for s in range(SCRFD_NUM_STRIDES):
                s_max = dequantize_int8(
                    score_data[s].flatten(), self.fd_score_scales[s], self.fd_score_zps[s]
                ).max()
                print(f"  SCRFD stride {STRIDES[s]}: max_score={s_max:.4f}")

        # Step 4-5: Decode + NMS
        dets = scrfd_decode_and_nms(
            score_data, bbox_data, kps_data,
            self.fd_score_scales, self.fd_score_zps,
            self.fd_bbox_scales, self.fd_bbox_zps,
            self.fd_kps_scales, self.fd_kps_zps,
            FD_INPUT_W, FD_INPUT_H, img_w, img_h,
            FACE_CONF_THRESHOLD, FACE_NMS_THRESHOLD,
        )

        if debug:
            print(f"  Detected {len(dets)} face(s) after NMS")
            for i, d in enumerate(dets):
                print(f"    Face {i}: bbox=({d['bbox'][0]:.0f},{d['bbox'][1]:.0f},"
                      f"{d['bbox'][2]:.0f}x{d['bbox'][3]:.0f}) score={d['score']:.3f}")

        # Step 6: Get best face
        best = get_best_face(dets, MIN_FACE_SIZE)
        used_fallback = False
        if best is None:
            if not ALLOW_CENTER_CROP_FALLBACK:
                raise RuntimeError("No face detected (try a clearer front-facing photo)")
            aligned_face = center_crop_resize_rgb(bgr_planar)
            best = {
                "bbox": (0.0, 0.0, float(img_w), float(img_h)),
                "score": 0.0,
                "landmarks": [(0.0, 0.0)] * SCRFD_NUM_LANDMARKS,
            }
            used_fallback = True

        if debug:
            if used_fallback:
                print("  No SCRFD face; using center-crop fallback")
            else:
                print(f"  Best face: bbox=({best['bbox'][0]:.0f},{best['bbox'][1]:.0f},"
                      f"{best['bbox'][2]:.0f}x{best['bbox'][3]:.0f}) score={best['score']:.3f}")

        # Validate face
        if not used_fallback and (best["bbox"][2] < MIN_FACE_SIZE or best["bbox"][3] < MIN_FACE_SIZE):
            raise RuntimeError(f"Face too small: {best['bbox'][2]:.0f}x{best['bbox'][3]:.0f}")

        if not used_fallback:
            # Step 7: Compute alignment transform
            M = compute_face_alignment(best["landmarks"])

            # Step 8: Affine warp
            aligned_face = apply_face_alignment(bgr_planar, M)

        # Step 9: Prepare embedding model input
        if self.emb_input_dtype == np.int8:
            # INT8 model: mirror firmware quantization from RGB [-1, 1] to int8.
            emb_in_q = self.emb_interp.get_input_details()[0].get(
                "quantization_parameters", {}
            )
            emb_in_zp = int(
                np.asarray(emb_in_q.get("zero_points", [-128])).flat[0]
            )
            emb_in_scale = float(
                np.asarray(emb_in_q.get("scales", [1.0])).flat[0]
            )
            emb_input = quantize_embedding_input_rgb(
                aligned_face, emb_in_scale, emb_in_zp
            )
            # Add batch dimension
            emb_input = emb_input[np.newaxis, ...]
            if debug:
                print(f"  Emb input: dtype=int8, zp={emb_in_zp}, "
                      f"range=[{emb_input.min()},{emb_input.max()}]")
        elif self.emb_input_dtype == np.float32:
            # Float32 model — normalize to [-1, 1] (standard ArcFace/MobileFaceNet)
            emb_input = (aligned_face.astype(np.float32) / 127.5) - 1.0
            emb_input = emb_input[np.newaxis, ...]  # add batch dim
            if debug:
                print(f"  Emb input: dtype=float32, "
                      f"range=[{emb_input.min():.3f},{emb_input.max():.3f}]")
        else:
            emb_input = aligned_face

        # Step 10: Run embedding model
        self.emb_interp.set_tensor(self.emb_input_idx, emb_input)
        self.emb_interp.invoke()

        # Step 11: Dequantize
        emb_output = self.emb_interp.get_tensor(self.emb_output_idx)
        emb_out_dtype = self.emb_interp.get_output_details()[0]["dtype"]

        if emb_out_dtype in (np.int8, np.uint8):
            embedding = (
                emb_output.astype(np.float32) - self.emb_out_zp
            ) * self.emb_out_scale
        else:
            embedding = emb_output.astype(np.float32)

        embedding = embedding.flatten()[:EMB_OUTPUT_DIM]

        # Step 12: L2 normalize
        embedding = l2_normalize(embedding)

        # Quality and pose
        quality = 0.0 if used_fallback else estimate_face_quality(best["landmarks"])
        pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0} if used_fallback else estimate_face_pose(best["landmarks"])

        return {
            "embedding": embedding,
            "bbox": best["bbox"],
            "score": best["score"],
            "landmarks": best["landmarks"],
            "aligned_face": aligned_face,
            "quality": quality,
            "pose": pose,
            "fallback": used_fallback,
        }

    def _run_onnx(
        self, bgr_planar: np.ndarray, fd_input_rgb: np.ndarray,
        img_w: int, img_h: int, debug: bool
    ) -> dict:
        # ONNX SCRFD expects float32 input, normalize from [0,255] to [0,1] or [-1,1]
        # Most SCRFD models expect [0, 1] range
        fd_input = fd_input_rgb.astype(np.float32) / 255.0
        fd_input = np.transpose(fd_input, (2, 0, 1))  # HWC → CHW
        fd_input = fd_input[np.newaxis, ...]  # add batch dim

        fd_outputs = self.fd_session.run(None, {self.fd_input_name: fd_input})

        # SCRFD ONNX typically has 9 outputs (3 strides x 3 types)
        # Parse similarly to TFLite
        if debug:
            print(f"  SCRFD ONNX outputs: {len(fd_outputs)} tensors")
            for i, o in enumerate(fd_outputs):
                print(f"    output[{i}]: shape={o.shape}")

        # ONNX output parsing — the exact format depends on the exported model.
        # For the SCRFD ONNX model in this repo, outputs are:
        # scores (3), bboxes (3), kps (3) — each as float32
        # We need to handle the specific ONNX output format.
        #
        # Create temporary TFLite interpreter for post-processing
        # (ONNX model outputs need different decoding)
        # For now, convert ONNX outputs to the format expected by scrfd_decode_and_nms

        num_outputs = len(fd_outputs)
        score_data = []
        bbox_data = []
        kps_data = []
        score_scales = [1.0] * 3
        score_zps = [0] * 3
        bbox_scales = [1.0] * 3
        bbox_zps = [0] * 3
        kps_scales = [1.0] * 3
        kps_zps = [0] * 3

        for i, out in enumerate(fd_outputs):
            s_idx = i // 3
            t_idx = i % 3
            if t_idx == 0:
                score_data.append(out)
            elif t_idx == 1:
                bbox_data.append(out)
            else:
                kps_data.append(out)

        # Ensure we have 3 strides
        while len(score_data) < 3:
            score_data.append(np.zeros((1,), dtype=np.float32))
            bbox_data.append(np.zeros((1, 4), dtype=np.float32))
            kps_data.append(np.zeros((1, 10), dtype=np.float32))

        dets = scrfd_decode_and_nms(
            score_data, bbox_data, kps_data,
            score_scales, score_zps,
            bbox_scales, bbox_zps,
            kps_scales, kps_zps,
            FD_INPUT_W, FD_INPUT_H, img_w, img_h,
            FACE_CONF_THRESHOLD, FACE_NMS_THRESHOLD,
        )

        best = get_best_face(dets, MIN_FACE_SIZE)
        if best is None:
            raise RuntimeError("No face detected")

        # Alignment
        M = compute_face_alignment(best["landmarks"])
        aligned_face = apply_face_alignment(bgr_planar, M)

        # GhostFaceNet ONNX expects float32, normalize
        emb_input = aligned_face.astype(np.float32)
        # MobileFaceNet trained with [-1, 1] normalization
        emb_input = (emb_input / 127.5) - 1.0
        emb_input = np.transpose(emb_input, (2, 0, 1))  # HWC → CHW
        emb_input = emb_input[np.newaxis, ...]

        emb_outputs = self.emb_session.run(None, {self.emb_input_name: emb_input})
        embedding = emb_outputs[0].flatten()[:EMB_OUTPUT_DIM]
        embedding = l2_normalize(embedding)

        quality = estimate_face_quality(best["landmarks"])
        pose = estimate_face_pose(best["landmarks"])

        return {
            "embedding": embedding,
            "bbox": best["bbox"],
            "score": best["score"],
            "landmarks": best["landmarks"],
            "aligned_face": aligned_face,
            "quality": quality,
            "pose": pose,
        }


# ---------------------------------------------------------------------------
# Face quality and pose estimation (from face_alignment.c)
# ---------------------------------------------------------------------------

def estimate_face_quality(landmarks: list) -> float:
    """Replica of estimate_face_quality() in face_alignment.c:242-268"""
    left_eye = np.array(landmarks[LM_LEFT_EYE])
    right_eye = np.array(landmarks[LM_RIGHT_EYE])
    nose = np.array(landmarks[LM_NOSE])
    left_mouth = np.array(landmarks[LM_LEFT_MOUTH])
    right_mouth = np.array(landmarks[LM_RIGHT_MOUTH])

    eye_dy = abs(left_eye[1] - right_eye[1])
    eye_dx = abs(left_eye[0] - right_eye[0])
    roll_ratio = eye_dy / (eye_dx + 1e-6)
    roll_score = max(0.0, 1.0 - roll_ratio * 3.0)
    quality = roll_score

    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    nose_offset = abs(nose[0] - eye_center_x)
    nose_range = eye_dx / 2.0
    yaw_score = max(0.0, 1.0 - nose_offset / (nose_range + 1e-6))
    quality *= yaw_score

    mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2.0
    mouth_offset = abs(mouth_center_x - eye_center_x)
    mouth_score = max(0.0, 1.0 - mouth_offset / (nose_range + 1e-6))
    quality *= mouth_score

    return float(quality)


def estimate_face_pose(landmarks: list) -> dict:
    """Replica of estimate_face_pose() in face_alignment.c:270-306"""
    left_eye = np.array(landmarks[LM_LEFT_EYE])
    right_eye = np.array(landmarks[LM_RIGHT_EYE])
    nose = np.array(landmarks[LM_NOSE])

    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    roll = np.arctan2(eye_dy, eye_dx) * 180.0 / np.pi

    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_width = np.sqrt(eye_dx ** 2 + eye_dy ** 2)

    nose_offset = nose[0] - eye_center_x
    norm_offset = nose_offset / (eye_width / 2.0 + 1e-6)
    yaw = norm_offset * 45.0

    nose_eye_dist = nose[1] - eye_center_y
    expected_dist = eye_width * 0.6
    ratio = nose_eye_dist / (expected_dist + 1e-6)
    pitch = (ratio - 1.0) * 30.0

    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Since embeddings are L2-normalized, cosine = dot product."""
    return float(np.dot(emb1, emb2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Face embedding computation — exact replica of Himax firmware pipeline"
    )
    parser.add_argument(
        "images", nargs="+", help="Image file path(s)"
    )
    parser.add_argument(
        "--output", "-o", choices=["json", "npz", "bin"], default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--compare", "-c", action="store_true",
        help="Compare two images and print cosine similarity"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Print debug information"
    )
    parser.add_argument(
        "--backend", choices=["tflite", "onnx"], default="tflite",
        help="Inference backend (default: tflite)"
    )
    parser.add_argument(
        "--scrfd-model", help="Path to SCRFD model (overrides default)"
    )
    parser.add_argument(
        "--embedding-model", help="Path to embedding model (overrides default)"
    )
    parser.add_argument(
        "--save-aligned", metavar="PATH",
        help="Save aligned face image to PATH"
    )
    args = parser.parse_args()

    # Resolve models
    if args.backend == "tflite":
        scrfd_model = args.scrfd_model or str(SCRFD_TFLITE)
        emb_model = args.embedding_model or str(DEPLOYED_EMBEDDING_TFLITE)
    else:
        scrfd_model = args.scrfd_model or str(SCRFD_ONNX)
        emb_model = args.embedding_model or str(MOBILEFACENET_ONNX)

    for path in [scrfd_model, emb_model]:
        if not os.path.exists(path):
            print(f"ERROR: Model not found: {path}")
            sys.exit(1)

    # Initialize pipeline
    pipeline = FaceEmbeddingPipeline(scrfd_model, emb_model, args.backend)

    if args.compare and len(args.images) == 2:
        # Compare mode
        print(f"Image 1: {args.images[0]}")
        result1 = pipeline.compute(Image.open(args.images[0]), debug=args.debug)
        print(f"  Confidence: {result1['score']:.3f}, Quality: {result1['quality']:.3f}")

        print(f"Image 2: {args.images[1]}")
        result2 = pipeline.compute(Image.open(args.images[1]), debug=args.debug)
        print(f"  Confidence: {result2['score']:.3f}, Quality: {result2['quality']:.3f}")

        sim = cosine_similarity(result1["embedding"], result2["embedding"])
        print(f"\nCosine similarity: {sim:.6f}")
        print("→ SAME person" if sim > 0.5 else "→ DIFFERENT person")

    else:
        for img_path in args.images:
            print(f"Processing: {img_path}")
            result = pipeline.compute(Image.open(img_path), debug=args.debug)

            print(f"  Confidence: {result['score']:.3f}")
            print(f"  Quality:   {result['quality']:.3f}")
            print(f"  Pose:      yaw={result['pose']['yaw']:.1f}, "
                  f"pitch={result['pose']['pitch']:.1f}, roll={result['pose']['roll']:.1f}")
            print(f"  Embedding: {EMB_OUTPUT_DIM}D (norm={np.linalg.norm(result['embedding']):.6f})")

            if args.output == "json":
                out = {
                    "embedding": result["embedding"].tolist(),
                    "bbox": [float(v) for v in result["bbox"]],
                    "score": float(result["score"]),
                    "quality": float(result["quality"]),
                    "pose": result["pose"],
                }
                print(json.dumps(out, indent=2))

            elif args.output == "npz":
                out_path = os.path.splitext(img_path)[0] + "_embedding.npz"
                np.savez(out_path, embedding=result["embedding"])
                print(f"  Saved: {out_path}")

            elif args.output == "bin":
                out_path = os.path.splitext(img_path)[0] + "_embedding.bin"
                result["embedding"].astype(np.float32).tofile(out_path)
                print(f"  Saved: {out_path} ({EMB_OUTPUT_DIM * 4} bytes)")

            if args.save_aligned:
                Image.fromarray(result["aligned_face"]).save(args.save_aligned)
                print(f"  Aligned face saved: {args.save_aligned}")


if __name__ == "__main__":
    main()
