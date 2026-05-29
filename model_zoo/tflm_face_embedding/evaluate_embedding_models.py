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
compute_embedding.ALLOW_CENTER_CROP_FALLBACK = True

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
    def image_paths(person):
        return sorted(
            p
            for p in person.glob("*.jpg")
            if not p.name.startswith(".") and not p.name.startswith("._")
        )

    people = sorted([d for d in lfw.iterdir() if d.is_dir() and len(image_paths(d)) >= 2])
    test_people = random.sample(people, min(max_pairs * 2, len(people)))

    same_pairs = []
    for person in test_people:
        imgs = image_paths(person)[:2]
        if len(imgs) == 2:
            same_pairs.append((str(imgs[0]), str(imgs[1])))

    diff_pairs = []
    for i in range(0, len(test_people) - 1, 2):
        a_imgs = image_paths(test_people[i])
        b_imgs = image_paths(test_people[i + 1])
        if a_imgs and b_imgs:
            diff_pairs.append((str(a_imgs[0]), str(b_imgs[0])))

    return same_pairs[:max_pairs], diff_pairs[: max_pairs // 2]


def parse_split_ids(value):
    splits = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            splits.extend(range(int(start), int(end) + 1))
        else:
            splits.append(int(item))
    return sorted(set(splits))


def load_cfp_pairs(max_pairs, split_ids=(1,)):
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
    def idx_to_path(idx, mapping):
        img_path = mapping.get(idx, "")
        full_path = (cfp / "Protocol" / img_path).resolve()
        return str(full_path) if full_path.exists() else None

    same_pairs = []
    diff_pairs = []

    for split_id in split_ids:
        same_raw = load_pairs(cfp / f"Protocol/Split/FP/{split_id:02d}/same.txt")
        diff_raw = load_pairs(cfp / f"Protocol/Split/FP/{split_id:02d}/diff.txt")

        split_same = []
        for f_idx, p_idx in same_raw[:max_pairs]:
            f_path = idx_to_path(f_idx, frontal_map)
            p_path = idx_to_path(p_idx, profile_map)
            if f_path and p_path:
                split_same.append((f_path, p_path))

        split_diff = []
        for f_idx, p_idx in diff_raw[:max_pairs]:
            f_path = idx_to_path(f_idx, frontal_map)
            p_path = idx_to_path(p_idx, profile_map)
            if f_path and p_path:
                split_diff.append((f_path, p_path))

        same_pairs.extend(split_same[:max_pairs])
        diff_pairs.extend(split_diff[: max_pairs // 2])

    return same_pairs, diff_pairs


def compute_embeddings(pipeline, paths):
    cache = {}
    failures = 0
    fallbacks = 0
    for path in sorted(set(paths)):
        try:
            result = pipeline.compute(Image.open(path), debug=False)
            cache[path] = result["embedding"].astype(np.float32)
            if result.get("fallback"):
                fallbacks += 1
        except Exception:
            failures += 1
    return cache, failures, fallbacks


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
        "same_sims": ss,
        "diff_sims": ds,
    }


def error_rates_at_threshold(stats, threshold):
    same = stats["same_sims"]
    diff = stats["diff_sims"]
    frr = float(np.mean(same < threshold))
    far = float(np.mean(diff >= threshold))
    tar = 1.0 - frr
    acc = float((np.sum(same >= threshold) + np.sum(diff < threshold)) / (len(same) + len(diff)))
    return {"far": far, "frr": frr, "tar": tar, "acc": acc}


def verification_metrics(stats):
    same = stats["same_sims"]
    diff = stats["diff_sims"]
    thresholds = np.unique(np.concatenate([same, diff, np.linspace(-1, 1, 1000)]))

    fars = np.asarray([np.mean(diff >= t) for t in thresholds])
    frrs = np.asarray([np.mean(same < t) for t in thresholds])
    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    result = {"eer": eer, "eer_threshold": eer_threshold}
    for target_far in (0.10, 0.05, 0.01):
        valid = np.where(fars <= target_far)[0]
        if len(valid) == 0:
            result[f"tar@far{target_far:g}"] = 0.0
            result[f"thr@far{target_far:g}"] = float(thresholds[-1])
            result[f"far@far{target_far:g}"] = float(fars[-1])
        else:
            best_idx = valid[np.argmin(frrs[valid])]
            result[f"tar@far{target_far:g}"] = float(1.0 - frrs[best_idx])
            result[f"thr@far{target_far:g}"] = float(thresholds[best_idx])
            result[f"far@far{target_far:g}"] = float(fars[best_idx])
    return result


def accuracy_at_threshold(stats, threshold):
    same = stats["same_sims"]
    diff = stats["diff_sims"]
    return float((np.sum(same >= threshold) + np.sum(diff < threshold)) / (len(same) + len(diff)))


def balanced_summary(row, dataset_names):
    stats_by_dataset = row["stats"]
    if any(not stats_by_dataset.get(name) for name in dataset_names):
        return None

    best = None
    for threshold in np.linspace(-1, 1, 800):
        accs = [accuracy_at_threshold(stats_by_dataset[name], threshold) for name in dataset_names]
        macro = float(np.mean(accs))
        floor = float(np.min(accs))
        gap = float(np.max(accs) - np.min(accs))
        hmean = float(len(accs) / np.sum([1.0 / max(acc, 1e-6) for acc in accs]))
        score = hmean - 0.25 * gap
        candidate = {
            "threshold": float(threshold),
            "accs": accs,
            "macro": macro,
            "floor": floor,
            "gap": gap,
            "hmean": hmean,
            "score": score,
        }
        key = (score, floor, macro, -gap)
        if best is None or key > best[0]:
            best = (key, candidate)
    return best[1]


def print_balanced_table(rows, dataset_names):
    print("\n# Balanced single-threshold selection")
    print("Score = harmonic_mean(LFW_acc, CFP_acc) - 0.25 * abs_gap; higher is better.")
    print(
        f"{'Model':<18} {'Dim':>5} {'Thr':>8} "
        f"{dataset_names[0] + '@Thr':>10} {dataset_names[1] + '@Thr':>12} "
        f"{'Floor':>8} {'Gap':>8} {'Score':>8}"
    )
    print("-" * 88)
    ranked = []
    for row in rows:
        summary = balanced_summary(row, dataset_names)
        if summary:
            ranked.append((summary["score"], row, summary))
        else:
            print(f"{row['name']:<18} {row['dim']:>5} FAILED")

    for _, row, summary in sorted(ranked, key=lambda item: item[0], reverse=True):
        print(
            f"{row['name']:<18} {row['dim']:>5} {summary['threshold']:>8.4f} "
            f"{summary['accs'][0]:>10.1%} {summary['accs'][1]:>12.1%} "
            f"{summary['floor']:>8.1%} {summary['gap']:>8.1%} {summary['score']:>8.3f}"
        )


def print_operating_table(rows, dataset_names):
    print("\n# Shared-threshold operating point")
    print("FAR = false accept rate; FRR = false reject rate at the balanced shared threshold.")
    print(
        f"{'Model':<18} {'Thr':>8} "
        f"{dataset_names[0] + ' FAR':>10} {dataset_names[0] + ' FRR':>10} "
        f"{dataset_names[1] + ' FAR':>12} {dataset_names[1] + ' FRR':>12}"
    )
    print("-" * 76)
    ranked = []
    for row in rows:
        summary = balanced_summary(row, dataset_names)
        if summary:
            ranked.append((summary["score"], row, summary))
    for _, row, summary in sorted(ranked, key=lambda item: item[0], reverse=True):
        threshold = summary["threshold"]
        rates = [error_rates_at_threshold(row["stats"][name], threshold) for name in dataset_names]
        print(
            f"{row['name']:<18} {threshold:>8.4f} "
            f"{rates[0]['far']:>10.1%} {rates[0]['frr']:>10.1%} "
            f"{rates[1]['far']:>12.1%} {rates[1]['frr']:>12.1%}"
        )


def print_verification_table(rows, dataset_name):
    print(f"\n# {dataset_name} verification metrics")
    print(
        f"{'Model':<18} {'EER':>8} {'EER Thr':>8} "
        f"{'TAR@FAR10':>10} {'TAR@FAR5':>10} {'TAR@FAR1':>10}"
    )
    print("-" * 76)
    for row in rows:
        stats = row["stats"].get(dataset_name)
        if not stats:
            print(f"{row['name']:<18} FAILED")
            continue
        metrics = verification_metrics(stats)
        print(
            f"{row['name']:<18} {metrics['eer']:>8.1%} {metrics['eer_threshold']:>8.4f} "
            f"{metrics['tar@far0.1']:>10.1%} {metrics['tar@far0.05']:>10.1%} "
            f"{metrics['tar@far0.01']:>10.1%}"
        )


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
    parser.add_argument("--cfp-splits", default="1")
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
        "CFP-FP": load_cfp_pairs(args.max_pairs, parse_split_ids(args.cfp_splits)),
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
        cache, failures, fallbacks = compute_embeddings(pipeline, all_paths)
        print(f"Computed {len(cache)} embeddings (failures={failures}, fallbacks={fallbacks})")

        rows.append(
            {
                "name": name,
                "dim": dim,
                "stats": {dataset_name: summarize_pairs(pairs, cache) for dataset_name, pairs in datasets.items()},
            }
        )

    for dataset_name in datasets:
        print_table(rows, dataset_name)
    for dataset_name in datasets:
        print_verification_table(rows, dataset_name)
    print_balanced_table(rows, list(datasets.keys()))
    print_operating_table(rows, list(datasets.keys()))

    print(f"\nDone in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
