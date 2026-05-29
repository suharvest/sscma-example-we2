#!/usr/bin/env python3
"""Convert MFN singlepath+ReLU SavedModel to INT8 TFLite for Ethos-U55."""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
SAVED_MODEL_DIR = SCRIPT_DIR / "mfn_sp_relu_saved_model"
CALIB_DIR = REPO_DIR / "calibration_data" / "qat_112"
OUT_PATH = SCRIPT_DIR / "mfn_sp_relu_int8.tflite"
NUM_CALIB = 500


def load_calibration_images(limit=NUM_CALIB):
    images = []
    for path in sorted(CALIB_DIR.glob("*.jpg"))[:limit]:
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (112, 112):
                img = img.resize((112, 112), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.float32))
        except Exception as exc:
            print(f"  Skipping {path.name}: {exc}")
    if not images:
        raise RuntimeError(f"No calibration images found in {CALIB_DIR}")
    print(f"Loaded {len(images)} calibration images, shape={images[0].shape}")
    return np.stack(images, axis=0)


def convert():
    calib_images = load_calibration_images()

    # Inspect saved model
    loaded = tf.saved_model.load(str(SAVED_MODEL_DIR))
    sig = loaded.signatures["serving_default"]
    for key, spec in sig.structured_input_signature[1].items():
        print(f"Input  '{key}': shape={spec.shape}, dtype={spec.dtype}")
    for key, spec in sig.structured_outputs.items():
        print(f"Output '{key}': shape={spec.shape}, dtype={spec.dtype}")

    def representative_dataset():
        for img in calib_images:
            yield [img[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting to INT8 TFLite...")
    model = converter.convert()
    OUT_PATH.write_bytes(model)
    print(f"Saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KiB)")

    # Quick check
    interp = tf.lite.Interpreter(model_path=str(OUT_PATH))
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    print(f"INT8 input:  q={in_details[0]['quantization']}")
    print(f"INT8 output: q={out_details[0]['quantization']}")
    print("Done.")


if __name__ == "__main__":
    os.chdir(REPO_DIR)
    convert()
