#!/usr/bin/env python3
"""Convert Official InsightFace MobileFaceNet (w600k_mbf) to TFLite INT8 for Ethos-U55."""

import os
import sys
import shutil
import subprocess
import argparse
import numpy as np
from pathlib import Path

MODEL_NAME = "mobilefacenet"
INPUT_SIZE = 112
EMBEDDING_DIM = 128

CALIB_DIR = Path(__file__).parent.parent / "calibration_data" / "qat_112"
OUTPUT_DIR = Path(__file__).parent
ONNX_MODEL = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
ONNX_FIXED = OUTPUT_DIR / f"{MODEL_NAME}_fixed.onnx"
TF_DIR = OUTPUT_DIR / "saved_model"
OUTPUT_FLOAT = OUTPUT_DIR / f"{MODEL_NAME}_float32.tflite"
OUTPUT_INT8 = OUTPUT_DIR / f"{MODEL_NAME}_qat_int8.tflite"


def fix_onnx_shapes():
    print("\n[1/5] Fix ONNX Shapes (dynamic batch -> static)")
    print("-" * 50)
    import onnx
    model = onnx.load(str(ONNX_MODEL))
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                dim.dim_value = 1
    for out in model.graph.output:
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                dim.dim_value = 1
    onnx.save(model, str(ONNX_FIXED))
    print(f"  Saved: {ONNX_FIXED.name}")
    return True


def convert_to_tflite():
    print("\n[2/5] Convert ONNX -> TFLite Float32 (onnx2tf + BatchNorm fusion)")
    print("-" * 50)
    import onnx2tf
    if TF_DIR.exists():
        shutil.rmtree(TF_DIR)
    onnx2tf.convert(
        input_onnx_file_path=str(ONNX_FIXED),
        output_folder_path=str(TF_DIR),
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
    )
    float_src = TF_DIR / f"{MODEL_NAME}_fixed_float32.tflite"
    if float_src.exists():
        shutil.copy(float_src, OUTPUT_FLOAT)
        size_kb = OUTPUT_FLOAT.stat().st_size / 1024
        print(f"  Saved: {OUTPUT_FLOAT.name} ({size_kb:.1f} KB)")
        return True
    return False


def load_calibration_images(max_images=500):
    import cv2
    image_files = sorted(CALIB_DIR.glob("*.jpg"))[:max_images]
    print(f"  Loading {len(image_files)} images from qat_112/")
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = (img.astype(np.float32) - 127.5) / 127.5
        images.append(img)
    print(f"  Loaded {len(images)} images")
    return images


def quantize_int8(calib_images, num_calib=500):
    print("\n[3/5] INT8 Full Integer Quantization")
    print("-" * 50)
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for img in calib_images[:num_calib]:
            yield [np.expand_dims(img, axis=0).astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_disable_per_channel = False

    tflite_model = converter.convert()
    with open(OUTPUT_INT8, 'wb') as f:
        f.write(tflite_model)
    print(f"  Saved: {OUTPUT_INT8.name} ({len(tflite_model)/1024:.1f} KB)")
    return True


def validate_quantization(calib_images):
    print("\n[4/5] Validate INT8 vs Float32 Accuracy")
    print("-" * 50)
    import tensorflow as tf

    f32 = tf.lite.Interpreter(model_path=str(OUTPUT_FLOAT))
    f32.allocate_tensors()
    i8 = tf.lite.Interpreter(model_path=str(OUTPUT_INT8))
    i8.allocate_tensors()

    def get_embedding(interp, img):
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        if inp['dtype'] == np.int8:
            qp = inp.get('quantization_parameters', {})
            s = qp.get('scales', [1.0])[0]
            zp = qp.get('zero_points', [0])[0]
            data = np.clip(img / s + zp, -128, 127).astype(np.int8)
        else:
            data = img.astype(np.float32)
        interp.set_tensor(inp['index'], np.expand_dims(data, 0))
        interp.invoke()
        emb = interp.get_tensor(out['index'])[0]
        if out['dtype'] == np.int8:
            qp = out.get('quantization_parameters', {})
            s = qp.get('scales', [1.0])[0]
            zp = qp.get('zero_points', [0])[0]
            emb = (emb.astype(np.float32) - zp) * s
        return emb

    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    sims = []
    for img in calib_images[:100]:
        sims.append(cosine(get_embedding(f32, img), get_embedding(i8, img)))

    mean_sim = np.mean(sims)
    print(f"  Mean cosine similarity: {mean_sim:.4f} ({mean_sim*100:.2f}%)")
    print(f"  Range: [{np.min(sims):.4f}, {np.max(sims):.4f}]")
    print(f"  Status: {'PASS' if mean_sim >= 0.98 else 'FAIL'}")
    return mean_sim


def vela_compile():
    print("\n[5/5] Vela Compile (ethos-u55-64)")
    print("-" * 50)
    vela = shutil.which("vela")
    if not vela:
        print("  Vela not found. Install: pip install ethos-u-vela")
        return None

    cmd = ["vela", str(OUTPUT_INT8),
           "--accelerator-config", "ethos-u55-64",
           "--optimise", "Performance",
           "--output-dir", str(OUTPUT_DIR)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.split('\n'):
        if any(k in line for k in ['SRAM', 'CPU operator', 'NPU operator', 'Batch Inference']):
            print(f"  {line.strip()}")

    vela_out = OUTPUT_DIR / f"{MODEL_NAME}_qat_int8_vela.tflite"
    if vela_out.exists():
        print(f"  Output: {vela_out.name} ({vela_out.stat().st_size/1024:.1f} KB)")
        return vela_out
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500]}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-calib', type=int, default=500)
    parser.add_argument('--skip-vela', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("Official InsightFace MobileFaceNet -> TFLite INT8")
    print("=" * 60)
    print(f"  Input: {INPUT_SIZE}x{INPUT_SIZE}x3, [-1,1]")
    print(f"  Output: {EMBEDDING_DIM}D embedding")
    print(f"  Source: w600k_mbf (WebFace600K, ArcFace)")

    if not ONNX_MODEL.exists():
        print(f"\nERROR: {ONNX_MODEL} not found")
        return 1

    if not fix_onnx_shapes():
        return 1
    if not convert_to_tflite():
        print("TFLite conversion failed!")
        return 1

    calib_images = load_calibration_images(max_images=args.num_calib)
    if not quantize_int8(calib_images, args.num_calib):
        return 1

    sim = validate_quantization(calib_images)

    vela_out = None
    if not args.skip_vela:
        vela_out = vela_compile()

    # Cleanup
    for p in [ONNX_FIXED, TF_DIR]:
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for f in [OUTPUT_FLOAT, OUTPUT_INT8, vela_out]:
        if f and f.exists():
            print(f"  {f.name}: {f.stat().st_size/1024:.1f} KB")
    print(f"  INT8 accuracy: {sim*100:.2f}%" if sim else "  accuracy: N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
