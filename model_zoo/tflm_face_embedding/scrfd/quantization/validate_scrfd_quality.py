#!/usr/bin/env python3
"""
Validate SCRFD TFLite score heads after quantization.

This is a lightweight gate for conversion / QAT experiments. It catches the
failure mode where one score branch is quantized to an unusable range or all
scores in a branch collapse to zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


INPUT_SIZE = 160
STRIDE_BY_COUNT = {800: 8, 200: 16, 50: 32}


def load_images(image_dir: Path, limit: int) -> list[np.ndarray]:
    image_paths = [
        path for path in sorted(image_dir.glob("*.jpg"))
        if not path.name.startswith("._")
    ][:limit]
    images = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image.astype(np.float32) / 255.0)

    return images


def quantize_input(image: np.ndarray, input_detail: dict) -> np.ndarray:
    batch = image[np.newaxis, ...].astype(np.float32)
    if input_detail["dtype"] != np.int8:
        return batch

    scale, zero_point = input_detail["quantization"]
    return np.clip(
        np.round(batch / scale + zero_point),
        -128,
        127,
    ).astype(np.int8)


def collect_score_outputs(output_details: list[dict]) -> dict[int, dict]:
    score_outputs = {}

    for output_detail in output_details:
        shape = output_detail["shape"]
        if len(shape) == 3:
            count = int(shape[1])
            channels = int(shape[2])
        elif len(shape) == 2:
            count = int(shape[0])
            channels = int(shape[1])
        else:
            continue

        stride = STRIDE_BY_COUNT.get(count)
        if stride is None or channels != 1:
            continue

        score_outputs[stride] = output_detail

    return score_outputs


def dequantize_score(raw: np.ndarray, output_detail: dict) -> np.ndarray:
    score = raw.astype(np.float32)
    scale, zero_point = output_detail["quantization"]
    if output_detail["dtype"] == np.int8 and scale != 0:
        score = (score - zero_point) * scale
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SCRFD TFLite score quality")
    parser.add_argument("--model", required=True, help="TFLite model path")
    parser.add_argument(
        "--image-dir",
        default="../../calibration_data/qat_160",
        help="Calibration/test image directory",
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument(
        "--min-score-scale",
        type=float,
        default=1e-6,
        help="Minimum acceptable quantization scale for score outputs",
    )
    parser.add_argument(
        "--min-branch-max",
        type=float,
        default=1e-4,
        help="Minimum observed max score per branch",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    image_dir = Path(args.image_dir)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    images = load_images(image_dir, args.num_samples)
    if not images:
        raise RuntimeError(f"No images loaded from {image_dir}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    score_outputs = collect_score_outputs(output_details)

    missing = [stride for stride in (8, 16, 32) if stride not in score_outputs]
    if missing:
        raise RuntimeError(f"Missing score outputs for strides: {missing}")

    max_scores = {stride: [] for stride in (8, 16, 32)}
    failures = []

    for stride, output_detail in score_outputs.items():
        scale, zero_point = output_detail["quantization"]
        print(
            f"stride {stride}: shape={output_detail['shape'].tolist()} "
            f"dtype={output_detail['dtype']} scale={scale:.10g} zp={zero_point}"
        )
        if output_detail["dtype"] == np.int8 and scale < args.min_score_scale:
            failures.append(
                f"stride {stride} score scale {scale:.10g} < {args.min_score_scale}"
            )

    for image in images:
        interpreter.set_tensor(input_detail["index"], quantize_input(image, input_detail))
        interpreter.invoke()

        for stride, output_detail in score_outputs.items():
            raw = interpreter.get_tensor(output_detail["index"])
            score = dequantize_score(raw, output_detail)
            max_scores[stride].append(float(np.max(score)))

    print("\nScore max distribution:")
    for stride in (8, 16, 32):
        values = np.asarray(max_scores[stride], dtype=np.float32)
        print(
            f"  s{stride}: min={values.min():.6f} mean={values.mean():.6f} "
            f"p50={np.quantile(values, 0.5):.6f} p90={np.quantile(values, 0.9):.6f} "
            f"max={values.max():.6f}"
        )
        if values.max() < args.min_branch_max:
            failures.append(
                f"stride {stride} max score {values.max():.6f} < {args.min_branch_max}"
            )

    best_any_stride = np.max(
        np.stack([np.asarray(max_scores[stride]) for stride in (8, 16, 32)]),
        axis=0,
    )
    print(
        f"\nBest stride max: mean={best_any_stride.mean():.6f} "
        f"p10={np.quantile(best_any_stride, 0.1):.6f} "
        f"p50={np.quantile(best_any_stride, 0.5):.6f}"
    )

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
