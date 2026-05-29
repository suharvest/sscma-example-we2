#!/usr/bin/env python3
"""Export official w600k MobileFaceNet with a fixed 512D -> 128D projection.

This keeps the official w600k backbone intact and appends the trained projection
from outputs/w600k_projection_128d.npz. It is a minimal-inheritance experiment:
accuracy should track the official model more closely than a separately trained
student, but intermediate-tensor SRAM may remain close to the original w600k.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVED_MODEL = SCRIPT_DIR / "official_mobilefacenet" / "saved_model_512d"
DEFAULT_PROJECTION = SCRIPT_DIR / "outputs" / "w600k_projection_128d.npz"
DEFAULT_OUT_DIR = SCRIPT_DIR / "official_mobilefacenet" / "w600k_projected_128d"
CALIB_DIR = SCRIPT_DIR / "calibration_data" / "qat_112"


class ProjectedW600K(tf.Module):
    def __init__(self, saved_model_dir, projection_path):
        super().__init__()
        self.base = tf.saved_model.load(str(saved_model_dir))
        projection_data = np.load(projection_path)
        self.mean = tf.constant(projection_data["mean"].astype(np.float32), dtype=tf.float32)
        self.projection = tf.constant(projection_data["projection"].astype(np.float32), dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([1, 112, 112, 3], tf.float32, name="input_1")])
    def serve(self, input_1):
        embedding512 = self.base.signatures["serving_default"](input_1=input_1)["output_0"]
        embedding128 = tf.matmul(embedding512 - self.mean, self.projection)
        return {"embedding": embedding128}


def load_calibration_images(limit):
    images = []
    for path in sorted(CALIB_DIR.glob("*.jpg"))[:limit]:
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (112, 112):
                img = img.resize((112, 112), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            images.append((arr / 127.5) - 1.0)
        except Exception:
            pass
    if not images:
        raise RuntimeError(f"No calibration images found in {CALIB_DIR}")
    return np.asarray(images, dtype=np.float32)


def export_float(saved_model_dir, projection_path, out_dir):
    wrapped_dir = out_dir / "saved_model"
    if wrapped_dir.exists():
        shutil.rmtree(wrapped_dir)
    module = ProjectedW600K(saved_model_dir, projection_path)
    tf.saved_model.save(module, str(wrapped_dir), signatures={"serving_default": module.serve})

    converter = tf.lite.TFLiteConverter.from_saved_model(str(wrapped_dir))
    model = converter.convert()
    out_path = out_dir / "w600k_projected_128d.float32.tflite"
    out_path.write_bytes(model)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")
    return wrapped_dir, out_path


def export_int8(wrapped_dir, calib_images, out_dir):
    def representative_dataset():
        for img in calib_images:
            yield [img[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(wrapped_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model = converter.convert()
    out_path = out_dir / "w600k_projected_128d.int8.tflite"
    out_path.write_bytes(model)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")
    return out_path


def run_vela(tflite_path, out_dir):
    vela = shutil.which("vela")
    if not vela:
        print("Vela not found; skipping")
        return
    result = subprocess.run(
        [
            vela,
            str(tflite_path),
            "--accelerator-config",
            "ethos-u55-64",
            "--optimise",
            "Performance",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError(f"Vela failed: {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-model", type=Path, default=DEFAULT_SAVED_MODEL)
    parser.add_argument("--projection", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--skip-vela", action="store_true")
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib_images = load_calibration_images(args.num_calib)
    wrapped_dir, _ = export_float(args.saved_model, args.projection, args.out_dir)
    int8_path = export_int8(wrapped_dir, calib_images, args.out_dir)
    if not args.skip_vela:
        run_vela(int8_path, args.out_dir)


if __name__ == "__main__":
    main()
