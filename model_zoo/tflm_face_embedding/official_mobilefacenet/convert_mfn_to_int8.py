#!/usr/bin/env python3
"""Convert MFN SavedModel to INT8 TFLite for Ethos-U55 deployment.

Follows the pattern in sface/convert_to_int8.py and export_w600k_dense128.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
SAVED_MODEL_DIR = SCRIPT_DIR / "mfn_saved_model"
CALIB_DIR = REPO_DIR / "calibration_data" / "qat_112"
OUT_PATH = SCRIPT_DIR / "mfn_s8_v1_int8.tflite"
NUM_CALIB = 500


def load_calibration_images(limit: int = NUM_CALIB) -> np.ndarray:
    """Load calibration images as raw float32 pixels [0, 255].

    The ONNX model's Sub/Mul normalization is in-graph (folded by onnx2tf),
    so the SavedModel input expects raw pixel values.
    """
    images = []
    for path in sorted(CALIB_DIR.glob("*.jpg"))[:limit]:
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (112, 112):
                img = img.resize((112, 112), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            images.append(arr)
        except Exception as exc:
            print(f"  Skipping {path.name}: {exc}")
    if not images:
        raise RuntimeError(f"No calibration images found in {CALIB_DIR}")
    print(f"Loaded {len(images)} calibration images, shape={images[0].shape}")
    return np.stack(images, axis=0)


def inspect_saved_model():
    """Inspect the SavedModel signature to understand input/output shapes."""
    print("\n--- SavedModel Signature ---")
    loaded = tf.saved_model.load(str(SAVED_MODEL_DIR))
    sig = loaded.signatures["serving_default"]
    for key, spec in sig.structured_input_signature[1].items():
        print(f"  Input  '{key}': shape={spec.shape}, dtype={spec.dtype}")
    for key, spec in sig.structured_outputs.items():
        print(f"  Output '{key}': shape={spec.shape}, dtype={spec.dtype}")
    return sig


def convert():
    calib_images = load_calibration_images(NUM_CALIB)

    # Inspect before conversion
    sig = inspect_saved_model()

    def representative_dataset():
        for img in calib_images:
            yield [img[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("\nConverting to INT8 TFLite...")
    model = converter.convert()

    OUT_PATH.write_bytes(model)
    print(f"Saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KiB)")

    # Inspect the quantized model
    interp = tf.lite.Interpreter(model_path=str(OUT_PATH))
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    print(f"\nINT8 Model input: {in_details}")
    print(f"INT8 Model output: {out_details}")

    # Quick numerical check
    print("\n--- Quick Numerical Check (first calibration image) ---")
    test_img = calib_images[0:1].astype(np.float32)
    interp.set_tensor(in_details[0]["index"], test_img)
    interp.invoke()
    int8_output = interp.get_tensor(out_details[0]["index"])
    print(f"INT8 output shape: {int8_output.shape}, dtype: {int8_output.dtype}")
    print(f"INT8 output stats: min={int8_output.min():.4f}, max={int8_output.max():.4f}, mean={int8_output.mean():.4f}")

    # Also test the float32 TFLite for comparison
    float_tflite_path = SAVED_MODEL_DIR / "mfn_s8_v1_reconstructed_float32.tflite"
    if float_tflite_path.exists():
        print("\n--- Comparison with FP32 TFLite ---")
        f_interp = tf.lite.Interpreter(model_path=str(float_tflite_path))
        f_interp.allocate_tensors()
        f_in = f_interp.get_input_details()[0]
        f_out = f_interp.get_output_details()[0]
        f_interp.set_tensor(f_in["index"], test_img)
        f_interp.invoke()
        fp32_output = f_interp.get_tensor(f_out["index"])
        print(f"FP32 output shape: {fp32_output.shape}, dtype: {fp32_output.dtype}")

        # Dequantize INT8 output for comparison
        q_scale = out_details[0]["quantization_parameters"]["scales"][0]
        q_zp = out_details[0]["quantization_parameters"]["zero_points"][0]
        deq_int8 = (int8_output.astype(np.float32) - q_zp) * q_scale
        cosine_sim = np.dot(fp32_output.flatten(), deq_int8.flatten()) / (
            np.linalg.norm(fp32_output.flatten()) * np.linalg.norm(deq_int8.flatten())
        )
        corr = np.corrcoef(fp32_output.flatten(), deq_int8.flatten())[0, 1]
        print(f"Cosine similarity (FP32 vs INT8 dequantized): {cosine_sim:.6f}")
        print(f"Pearson correlation: {corr:.6f}")
        print(f"Output scale: {q_scale}, zero_point: {q_zp}")

    print("\nDone.")


if __name__ == "__main__":
    os.chdir(REPO_DIR)
    convert()
