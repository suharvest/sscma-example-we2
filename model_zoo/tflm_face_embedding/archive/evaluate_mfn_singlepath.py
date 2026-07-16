#!/usr/bin/env python3
"""Evaluate MFN singlepath (PReLU vs ReLU) on LFW view2.

Uses float32 TFLite models with in-graph normalization (Sub/Mul).
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
LFW_DIR = SCRIPT_DIR / "datasets" / "lfw"
PAIRS_CSV = SCRIPT_DIR / "calibration_data" / "pairs.csv"

MODELS = {
    "MFN-SP-PReLU": SCRIPT_DIR / "official_mobilefacenet" / "mfn_sp_saved_model" / "mfn_s8_v1_singlepath_float32.tflite",
    "MFN-SP-ReLU":  SCRIPT_DIR / "official_mobilefacenet" / "mfn_sp_relu_saved_model" / "mfn_s8_v1_singlepath_relu_float32.tflite",
}

IMG_SIZE = 112


def load_pairs(path: Path):
    """Load LFW view2 pairs. Returns (same_pairs, diff_pairs) as list of (path1, path2)."""
    same_pairs = []
    diff_pairs = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            row = [c.strip() for c in row]
            if len(row) < 3:
                continue
            # Same pair: name, img1, img2
            # Diff pair: name1, img1, name2, img2
            if row[2].isdigit():
                # Same pair: name, img1, img2
                name = row[0]
                img1 = LFW_DIR / name / f"{name}_{int(row[1]):04d}.jpg"
                img2 = LFW_DIR / name / f"{name}_{int(row[2]):04d}.jpg"
                if img1.exists() and img2.exists():
                    same_pairs.append((str(img1), str(img2)))
            else:
                # Different pair: name1, img1, name2, img2
                name1 = row[0]
                img_num1 = int(row[1])
                name2 = row[2]
                img_num2 = int(row[3]) if len(row) > 3 and row[3] else 1
                img1 = LFW_DIR / name1 / f"{name1}_{img_num1:04d}.jpg"
                img2 = LFW_DIR / name2 / f"{name2}_{img_num2:04d}.jpg"
                if img1.exists() and img2.exists():
                    diff_pairs.append((str(img1), str(img2)))

    print(f"Loaded {len(same_pairs)} same pairs, {len(diff_pairs)} diff pairs")
    return same_pairs, diff_pairs


def preprocess_image(path: str) -> np.ndarray:
    """Load image, center crop to square, resize to 112x112, return as float32 [0,255].

    The model has in-graph normalization (Sub(127.5) * Mul(1/127.5)),
    so input should be raw pixel values in [0, 255].
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Center crop to square
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)  # [0, 255]
    return arr


def load_model(tflite_path: str):
    """Load TFLite model and return interpreter + input/output details."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    return interp, in_detail, out_detail


def compute_embeddings(interp, in_detail, out_detail, paths):
    """Compute embeddings for all unique paths."""
    unique = sorted(set(paths))
    cache = {}
    for i, path in enumerate(unique):
        try:
            arr = preprocess_image(path)
            # Model expects NHWC: [1, 112, 112, 3]
            inp = arr[np.newaxis, ...].astype(np.float32)
            interp.set_tensor(in_detail["index"], inp)
            interp.invoke()
            emb = interp.get_tensor(out_detail["index"]).flatten().astype(np.float64)
            cache[path] = emb
        except Exception as e:
            print(f"  FAIL: {path}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(unique)}")
    print(f"  Computed {len(cache)}/{len(unique)} embeddings")
    return cache


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def evaluate_model(name, tflite_path, same_pairs, diff_pairs):
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  Model: {tflite_path}")

    interp, in_detail, out_detail = load_model(str(tflite_path))

    # Collect all unique image paths
    all_paths = []
    for a, b in same_pairs:
        all_paths.extend([a, b])
    for a, b in diff_pairs:
        all_paths.extend([a, b])

    print(f"  Computing embeddings for {len(set(all_paths))} unique images...")
    start = time.time()
    cache = compute_embeddings(interp, in_detail, out_detail, all_paths)
    embed_time = time.time() - start
    print(f"  Embedding time: {embed_time:.1f}s")

    # Compute similarities
    same_sims = []
    for a, b in same_pairs:
        if a in cache and b in cache:
            same_sims.append(cosine_similarity(cache[a], cache[b]))

    diff_sims = []
    for a, b in diff_pairs:
        if a in cache and b in cache:
            diff_sims.append(cosine_similarity(cache[a], cache[b]))

    ss = np.array(same_sims)
    ds = np.array(diff_sims)

    # Accuracy at best threshold
    thresholds = np.linspace(-1, 1, 400)
    best_acc = max(
        (np.sum(ss >= t) + np.sum(ds < t)) / (len(ss) + len(ds)) for t in thresholds
    )

    # EER
    all_thresholds = np.unique(np.concatenate([ss, ds, np.linspace(-1, 1, 1000)]))
    fars = np.array([np.mean(ds >= t) for t in all_thresholds])
    frrs = np.array([np.mean(ss < t) for t in all_thresholds])
    eer_idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[eer_idx] + frrs[eer_idx]) / 2.0)

    # TAR@FAR=1%
    target_far = 0.01
    valid = np.where(fars <= target_far)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmin(frrs[valid])]
        tar_far1 = float(1.0 - frrs[best_idx])
    else:
        tar_far1 = 0.0

    # Separation
    separation = float(ss.mean() - ds.mean())

    results = {
        "same_mean": float(ss.mean()),
        "diff_mean": float(ds.mean()),
        "same_std": float(ss.std()),
        "diff_std": float(ds.std()),
        "separation": separation,
        "accuracy": float(best_acc),
        "eer": eer,
        "tar@far1%": tar_far1,
        "n_same": len(same_sims),
        "n_diff": len(diff_sims),
    }

    print(f"\n  --- Results: {name} ---")
    print(f"  Same pairs:     {results['n_same']}")
    print(f"  Diff pairs:     {results['n_diff']}")
    print(f"  Same μ±σ:       {results['same_mean']:.4f} ± {results['same_std']:.4f}")
    print(f"  Diff μ±σ:       {results['diff_mean']:.4f} ± {results['diff_std']:.4f}")
    print(f"  Separation:     {results['separation']:.4f}")
    print(f"  Accuracy:       {results['accuracy']:.4%}")
    print(f"  EER:            {results['eer']:.4%}")
    print(f"  TAR@FAR=1%:     {results['tar@far1%']:.4%}")

    return results


def main():
    os.chdir(SCRIPT_DIR)

    same_pairs, diff_pairs = load_pairs(PAIRS_CSV)

    all_results = {}
    for name, model_path in MODELS.items():
        if model_path.exists():
            all_results[name] = evaluate_model(name, model_path, same_pairs, diff_pairs)
        else:
            print(f"\nSKIP {name}: model not found at {model_path}")

    # Comparison
    if len(all_results) >= 2:
        print(f"\n{'='*60}")
        print("COMPARISON: PReLU vs ReLU")
        print(f"{'Metric':<20} {'PReLU':>12} {'ReLU':>12} {'Delta':>12}")
        print("-" * 56)
        names = list(all_results.keys())
        keys = ["separation", "accuracy", "eer", "tar@far1%"]
        fmt = {"separation": ".4f", "accuracy": ".4%", "eer": ".4%", "tar@far1%": ".4%"}
        for k in keys:
            v1 = all_results[names[0]][k]
            v2 = all_results[names[1]][k]
            delta = v2 - v1
            print(f"  {k:<18} {v1:>12{fmt[k]}} {v2:>12{fmt[k]}} {delta:>+12{fmt[k]}}")

    print("\nDone.")


if __name__ == "__main__":
    main()
