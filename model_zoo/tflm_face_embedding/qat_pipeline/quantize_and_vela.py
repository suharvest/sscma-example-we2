#!/usr/bin/env python3
"""Quantize distilled QAT ONNX to INT8 TFLite, then Vela-compile for Ethos-U55.

Pipeline:
    QAT ONNX (FP32, QAT-tuned)
        -> onnx2tf -> TF SavedModel
        -> tf.lite.TFLiteConverter (INT8 + representative dataset)
        -> vela -> Ethos-U55 .tflite (NPU-mapped)

Usage:
    cd .../tflm_face_embedding && uv run python training/quantize_and_vela.py \\
        --onnx training/output/qat_distilled/model_qat_clean.onnx \\
        --out training/output/qat_distilled \\
        --num-calib 500
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # tflm_face_embedding/
CALIB_DIR = SCRIPT_DIR / "calibration_data" / "qat_112"


def convert_onnx_to_tf(input_onnx: Path, out_dir: Path) -> Path:
    import onnx2tf
    tf_dir = out_dir / "saved_model"
    if tf_dir.exists():
        shutil.rmtree(tf_dir)
    onnx2tf.convert(
        input_onnx_file_path=str(input_onnx),
        output_folder_path=str(tf_dir),
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
    )
    floats = sorted(tf_dir.glob("*_float32.tflite"))
    if not floats:
        raise RuntimeError(f"No float32 tflite in {tf_dir}")
    fp32 = out_dir / "model_distilled_qat.float32.tflite"
    shutil.copy(floats[0], fp32)
    print(f"FP32 TFLite: {fp32} ({fp32.stat().st_size/1024:.1f} KiB)")
    return tf_dir


def load_calib_images(limit: int) -> np.ndarray:
    imgs = []
    for p in sorted(CALIB_DIR.glob("*.jpg"))[:limit]:
        try:
            img = Image.open(p).convert("RGB")
            if img.size != (112, 112):
                img = img.resize((112, 112), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            imgs.append((arr / 127.5) - 1.0)
        except Exception:
            pass
    if not imgs:
        raise RuntimeError(f"No calibration images in {CALIB_DIR}")
    print(f"Loaded {len(imgs)} calibration images")
    return np.asarray(imgs, dtype=np.float32)


def export_int8(tf_dir: Path, calib: np.ndarray, out: Path) -> Path:
    def rep():
        for img in calib:
            yield [img[np.newaxis, ...].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    model = conv.convert()
    out.write_bytes(model)
    print(f"INT8 TFLite: {out} ({out.stat().st_size/1024:.1f} KiB)")
    return out


def run_vela(int8_path: Path, out_dir: Path) -> Path:
    vela = shutil.which("vela")
    if not vela:
        raise RuntimeError("vela not found in PATH")
    result = subprocess.run(
        [vela, str(int8_path),
         "--accelerator-config", "ethos-u55-64",
         "--optimise", "Performance",
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, check=False,
    )
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError(f"vela failed: {result.returncode}")
    vela_out = out_dir / (int8_path.stem + "_vela.tflite")
    if vela_out.exists():
        print(f"Vela: {vela_out} ({vela_out.stat().st_size/1024:.1f} KiB)")
    return vela_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-calib", type=int, default=500)
    p.add_argument("--skip-vela", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tf_dir = convert_onnx_to_tf(args.onnx, args.out)
    int8 = export_int8(tf_dir, load_calib_images(args.num_calib),
                       args.out / "model_distilled_qat.int8.tflite")
    if not args.skip_vela:
        run_vela(int8, args.out)


if __name__ == "__main__":
    main()
