#!/usr/bin/env python3
"""Evaluate multiple pre-Vela embedding TFLite models on local LFW/CFP data."""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import compute_embedding

# CFP-FP images are already face-cropped. Keep this consistent with
# run_cfp_only.py so profile/frontal pairs are not rejected as oversized.
compute_embedding.MAX_FACE_RATIO = 1.0

from compute_embedding import FaceEmbeddingPipeline, cosine_similarity  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
SCRFD_MODEL = "scrfd/models/scrfd_500m_kps_int8.tflite"

DEFAULT_MODELS = {
    "w600k-512d": "official_mobilefacenet/w600k_mbf_int8.tflite",
    "mfn-lrelu-128d": "official_mobilefacenet/mfn_lrelu_int8.tflite",
    "mfn-relu-128d": "official_mobilefacenet/mfn_relu_clean_int8.tflite",
    "mfn-w0.5-norm": "official_mobilefacenet/mfn_w0.5_norm_int8.tflite",
    "mfn-w0.6": "official_mobilefacenet/mfn_w0.6_int8.tflite",
}


def get_output_dim(model_path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=str(model_path))
    out_shape = interp.get_output_details()[0]["shape"]
    return int(np.prod(out_shape[1:]))


def load_lfw_pairs(max_pairs):
    lfw = Path("datasets/lfw")
    random.seed(42)
    people = sorted([d for d in lfw.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) >= 2])
    test_people = random.sample(people, min(max_pairs * 2, len(people)))

    same_pairs = []
    for person in test_people:
        imgs = sorted(person.glob("*.jpg"))[:2]
        if len(imgs) == 2:
            same_pairs.append((str(imgs[0]), str(imgs[1])))

    diff_pairs = []
    for i in range(0, len(test_people) - 1, 2):
        a_imgs = sorted(test_people[i].glob("*.jpg"))
        b_imgs = sorted(test_people[i + 1].glob("*.jpg"))
        if a_imgs and b_imgs:
            diff_pairs.append((str(a_imgs[0]), str(b_imgs[0])))

    return same_pairs[:max_pairs], diff_pairs[: max_pairs // 2]


def load_cfp_pairs(max_pairs):
    cfp = Path("datasets/cfp/cfp-dataset")

    def load_list(path):
        mapping = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mapping[int(parts[0])] = parts[1]
        return mapping

    def load_pairs(path):
        pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and "," in line:
                    f_idx, p_idx = line.split(",")
                    pairs.append((int(f_idx), int(p_idx)))
        return pairs

    frontal_map = load_list(cfp / "Protocol/Pair_list_F.txt")
    profile_map = load_list(cfp / "Protocol/Pair_list_P.txt")
    same_raw = load_pairs(cfp / "Protocol/Split/FP/01/same.txt")
    diff_raw = load_pairs(cfp / "Protocol/Split/FP/01/diff.txt")

    def idx_to_path(idx, mapping):
        img_path = mapping.get(idx, "")
        full_path = (cfp / "Protocol" / img_path).resolve()
        return str(full_path) if full_path.exists() else None

    same_pairs = []
    for f_idx, p_idx in same_raw[:max_pairs]:
        f_path = idx_to_path(f_idx, frontal_map)
        p_path = idx_to_path(p_idx, profile_map)
        if f_path and p_path:
            same_pairs.append((f_path, p_path))

    diff_pairs = []
    for f_idx, p_idx in diff_raw[:max_pairs]:
        f_path = idx_to_path(f_idx, frontal_map)
        p_path = idx_to_path(p_idx, profile_map)
        if f_path and p_path:
            diff_pairs.append((f_path, p_path))

    return same_pairs[:max_pairs], diff_pairs[: max_pairs // 2]


def compute_embeddings(pipeline, paths):
    cache = {}
    failures = 0
    for path in sorted(set(paths)):
        try:
            result = pipeline.compute(Image.open(path), debug=False)
            cache[path] = result["embedding"].astype(np.float32)
        except Exception:
            failures += 1
    return cache, failures


def summarize_pairs(pairs, cache):
    same_pairs, diff_pairs = pairs
    same_sims = []
    diff_sims = []

    for a, b in same_pairs:
        if a in cache and b in cache:
            same_sims.append(cosine_similarity(cache[a], cache[b]))

    for a, b in diff_pairs:
        if a in cache and b in cache:
            diff_sims.append(cosine_similarity(cache[a], cache[b]))

    if not same_sims or not diff_sims:
        return None

    ss = np.asarray(same_sims)
    ds = np.asarray(diff_sims)
    thresholds = np.linspace(-1, 1, 400)
    best_acc = max((np.sum(ss >= t) + np.sum(ds < t)) / (len(ss) + len(ds)) for t in thresholds)
    return {
        "sep": float(ss.mean() - ds.mean()),
        "acc": float(best_acc),
        "same": float(ss.mean()),
        "diff": float(ds.mean()),
        "s_min": float(ss.min()),
        "d_max": float(ds.max()),
        "n": f"{len(ss)}s/{len(ds)}d",
    }


def print_table(rows, dataset_name):
    print(f"\n# {dataset_name}")
    print(f"{'Model':<18} {'Dim':>5} {'Sep':>8} {'Acc':>8} {'Same':>8} {'Diff':>8} {'S_min':>8} {'D_max':>8} {'N':>10}")
    print("-" * 94)
    for row in rows:
        stats = row["stats"].get(dataset_name)
        if not stats:
            print(f"{row['name']:<18} {row['dim']:>5} FAILED")
            continue
        print(
            f"{row['name']:<18} {row['dim']:>5} {stats['sep']:>8.4f} {stats['acc']:>8.1%} "
            f"{stats['same']:>8.4f} {stats['diff']:>8.4f} {stats['s_min']:>8.4f} "
            f"{stats['d_max']:>8.4f} {stats['n']:>10}"
        )


def parse_model_arg(values):
    models = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        models[name] = path
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model as name=path or path. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        print(f"Changing cwd to {SCRIPT_DIR}")
        os.chdir(SCRIPT_DIR)

    models = parse_model_arg(args.model) if args.model else DEFAULT_MODELS
    datasets = {
        "LFW": load_lfw_pairs(args.max_pairs),
        "CFP-FP": load_cfp_pairs(args.max_pairs),
    }
    all_paths = []
    for same_pairs, diff_pairs in datasets.values():
        for a, b in same_pairs + diff_pairs:
            all_paths.extend([a, b])

    start = time.time()
    rows = []
    for name, model in models.items():
        model_path = Path(model)
        dim = get_output_dim(model_path)
        compute_embedding.EMB_OUTPUT_DIM = dim
        print(f"\nEvaluating {name}: {model} ({dim}D)")
        pipeline = FaceEmbeddingPipeline(SCRFD_MODEL, model, backend="tflite")
        cache, failures = compute_embeddings(pipeline, all_paths)
        print(f"Computed {len(cache)} embeddings (failures={failures})")

        rows.append(
            {
                "name": name,
                "dim": dim,
                "stats": {dataset_name: summarize_pairs(pairs, cache) for dataset_name, pairs in datasets.items()},
            }
        )

    for dataset_name in datasets:
        print_table(rows, dataset_name)

    print(f"\nDone in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
