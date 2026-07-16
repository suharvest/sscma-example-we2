#!/usr/bin/env python3
"""Evaluate 512D -> 128D post-embedding compression for w600k_mbf.

This is a PC-side experiment. It does not reduce Ethos-U SRAM by itself; it
checks whether compressing the exported embedding preserves discriminability.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import compute_embedding

compute_embedding.EMB_OUTPUT_DIM = 512
# CFP-FP images are already face-cropped. Keep this consistent with
# run_cfp_only.py so the detector does not reject large valid faces.
compute_embedding.MAX_FACE_RATIO = 1.0

from compute_embedding import FaceEmbeddingPipeline, cosine_similarity, l2_normalize  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
SCRFD_MODEL = "scrfd/models/scrfd_500m_kps_int8.tflite"
W600K_MODEL = "official_mobilefacenet/w600k_mbf_int8.tflite"


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


def fit_pca(pipeline, num_calib):
    calib_paths = sorted(Path("calibration_data/qat_112").glob("*.jpg"))[:num_calib]
    embs = []
    failures = 0
    for path in calib_paths:
        try:
            result = pipeline.compute(Image.open(path), debug=False)
            embs.append(result["embedding"].astype(np.float32))
        except Exception:
            failures += 1

    if len(embs) < 128:
        raise RuntimeError(f"Need at least 128 calibration embeddings, got {len(embs)}")

    x = np.stack(embs)
    mean = x.mean(axis=0)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:128].astype(np.float32)
    return mean.astype(np.float32), components, len(embs), failures


def make_projectors(pca_mean, pca_components, trained_projection_path=None):
    rng = np.random.default_rng(42)
    random_proj = rng.normal(size=(128, 512)).astype(np.float32) / np.sqrt(128.0)
    trained_projection = None
    trained_mean = None
    if trained_projection_path:
        data = np.load(trained_projection_path)
        trained_projection = data["projection"].astype(np.float32)
        trained_mean = data["mean"].astype(np.float32)

    def identity512(x):
        return l2_normalize(x)

    def truncate128(x):
        return l2_normalize(x[:128])

    def random128(x):
        return l2_normalize(random_proj @ x)

    def pca128(x):
        return l2_normalize(pca_components @ (x - pca_mean))

    projectors = {
        "512D-baseline": identity512,
        "128D-truncate": truncate128,
        "128D-random-proj": random128,
        "128D-pca": pca128,
    }

    if trained_projection is not None:
        def trained128(x):
            return l2_normalize((x - trained_mean) @ trained_projection)

        projectors["128D-trained"] = trained128

    return projectors


def evaluate_pairs(name, pairs, cache, projectors):
    print(f"\n# {name}: {len(pairs[0])} same, {len(pairs[1])} diff")
    print(f"{'Method':<18} {'Sep':>8} {'Acc':>8} {'Same':>8} {'Diff':>8} {'S_min':>8} {'D_max':>8} {'N':>10}")
    print("-" * 86)

    same_pairs, diff_pairs = pairs
    for method, project in projectors.items():
        same_sims = []
        diff_sims = []

        for a, b in same_pairs:
            if a in cache and b in cache:
                same_sims.append(cosine_similarity(project(cache[a]), project(cache[b])))

        for a, b in diff_pairs:
            if a in cache and b in cache:
                diff_sims.append(cosine_similarity(project(cache[a]), project(cache[b])))

        if not same_sims or not diff_sims:
            print(f"{method:<18} FAILED")
            continue

        ss = np.asarray(same_sims)
        ds = np.asarray(diff_sims)
        thresholds = np.linspace(-1, 1, 400)
        best_acc = max((np.sum(ss >= t) + np.sum(ds < t)) / (len(ss) + len(ds)) for t in thresholds)
        print(
            f"{method:<18} {ss.mean() - ds.mean():>8.4f} {best_acc:>8.1%} "
            f"{ss.mean():>8.4f} {ds.mean():>8.4f} {ss.min():>8.4f} {ds.max():>8.4f} "
            f"{len(ss)}s/{len(ds)}d"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--model", default=W600K_MODEL)
    parser.add_argument("--projection", help="Optional .npz produced by train_w600k_projection_128d.py")
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        print(f"Changing cwd to {SCRIPT_DIR}")
        import os

        os.chdir(SCRIPT_DIR)

    start = time.time()
    print(f"Embedding model: {args.model}")
    print("Loading pipeline...")
    pipeline = FaceEmbeddingPipeline(SCRFD_MODEL, args.model, backend="tflite")

    print(f"Fitting PCA from {args.num_calib} calibration crops...")
    pca_mean, pca_components, n_calib, n_calib_fail = fit_pca(pipeline, args.num_calib)
    print(f"PCA fitted from {n_calib} embeddings (failures={n_calib_fail})")
    projectors = make_projectors(pca_mean, pca_components, args.projection)

    datasets = {
        "LFW": load_lfw_pairs(args.max_pairs),
        "CFP-FP": load_cfp_pairs(args.max_pairs),
    }

    all_paths = []
    for same_pairs, diff_pairs in datasets.values():
        for a, b in same_pairs + diff_pairs:
            all_paths.extend([a, b])

    print(f"Computing {len(set(all_paths))} evaluation embeddings...")
    cache, failures = compute_embeddings(pipeline, all_paths)
    print(f"Computed {len(cache)} embeddings (failures={failures})")

    for name, pairs in datasets.items():
        evaluate_pairs(name, pairs, cache, projectors)

    print(f"\nDone in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
