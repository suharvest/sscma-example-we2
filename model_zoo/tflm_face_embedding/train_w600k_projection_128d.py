#!/usr/bin/env python3
"""Train a 512D -> 128D projection for w600k embeddings.

This is an output-compression experiment. It preserves the w600k backbone and
learns a small projection that keeps pairwise cosine relationships close to the
512D teacher embeddings. It does not reduce Ethos-U intermediate tensor SRAM.
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

import compute_embedding

compute_embedding.EMB_OUTPUT_DIM = 512
compute_embedding.MAX_FACE_RATIO = 1.0

from compute_embedding import FaceEmbeddingPipeline, cosine_similarity, l2_normalize  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
SCRFD_MODEL = "scrfd/models/scrfd_500m_kps_int8.tflite"
W600K_MODEL = "official_mobilefacenet/w600k_mbf_int8.tflite"
DEFAULT_CACHE = SCRIPT_DIR / "outputs" / "w600k_calib_embeddings_512d.npz"
DEFAULT_OUT = SCRIPT_DIR / "outputs" / "w600k_projection_128d.npz"


def collect_image_paths(limit):
    paths = sorted(
        p for p in (SCRIPT_DIR / "calibration_data" / "qat_112").glob("*.jpg") if not p.name.startswith("._")
    )
    return paths[:limit]


def compute_or_load_embeddings(limit, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        embeddings = data["embeddings"].astype(np.float32)
        paths = data["paths"].astype(str)
        if len(embeddings) >= limit:
            print(f"Loaded {limit} cached embeddings from {cache_path}")
            return paths[:limit], embeddings[:limit]

    pipeline = FaceEmbeddingPipeline(SCRFD_MODEL, W600K_MODEL, backend="tflite")
    embeddings = []
    kept_paths = []
    failures = 0
    for path in collect_image_paths(limit):
        try:
            result = pipeline.compute(Image.open(path), debug=False)
            embeddings.append(result["embedding"].astype(np.float32))
            kept_paths.append(str(path))
        except Exception:
            failures += 1

    if len(embeddings) < 128:
        raise RuntimeError(f"Need at least 128 embeddings, got {len(embeddings)}")

    embeddings = np.stack(embeddings).astype(np.float32)
    kept_paths = np.asarray(kept_paths)
    np.savez(cache_path, paths=kept_paths, embeddings=embeddings)
    print(f"Cached {len(embeddings)} embeddings to {cache_path} (failures={failures})")
    return kept_paths, embeddings


def fit_pca_init(embeddings, output_dim):
    mean = embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(embeddings - mean, full_matrices=False)
    return vt[:output_dim].T.astype(np.float32), mean.squeeze(0).astype(np.float32)


def train_projection(embeddings, output_dim, steps, batch_size, lr, seed):
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    pca_w, mean = fit_pca_init(embeddings, output_dim)
    w = tf.Variable(pca_w)
    x_all = tf.constant(embeddings, dtype=tf.float32)
    mean_tf = tf.constant(mean, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    n = embeddings.shape[0]
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch_size)
        x = tf.gather(x_all, idx)
        teacher = tf.linalg.matmul(x, x, transpose_b=True)

        with tf.GradientTape() as tape:
            y = tf.linalg.matmul(x - mean_tf, w)
            y = tf.math.l2_normalize(y, axis=-1)
            student = tf.linalg.matmul(y, y, transpose_b=True)
            pair_loss = tf.reduce_mean(tf.square(student - teacher))
            col_reg = tf.reduce_mean(tf.square(tf.linalg.matmul(w, w, transpose_a=True) - tf.eye(output_dim)))
            loss = pair_loss + 0.001 * col_reg

        grads = tape.gradient(loss, [w])
        opt.apply_gradients(zip(grads, [w]))

        if step == 1 or step % 100 == 0 or step == steps:
            print(f"step {step:5d}/{steps} loss={loss.numpy():.6f} pair={pair_loss.numpy():.6f}")

    return mean, w.numpy().astype(np.float32)


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


def compute_eval_cache(max_pairs):
    pipeline = FaceEmbeddingPipeline(SCRFD_MODEL, W600K_MODEL, backend="tflite")
    datasets = {
        "LFW": load_lfw_pairs(max_pairs),
        "CFP-FP": load_cfp_pairs(max_pairs),
    }
    all_paths = []
    for same_pairs, diff_pairs in datasets.values():
        for a, b in same_pairs + diff_pairs:
            all_paths.extend([a, b])

    cache = {}
    failures = 0
    for path in sorted(set(all_paths)):
        try:
            result = pipeline.compute(Image.open(path), debug=False)
            cache[path] = result["embedding"].astype(np.float32)
        except Exception:
            failures += 1
    print(f"Computed {len(cache)} eval embeddings (failures={failures})")
    return datasets, cache


def evaluate_projection(datasets, cache, mean, projection):
    def project(x):
        return l2_normalize((x - mean) @ projection)

    print(f"\n{'Dataset':<8} {'Method':<16} {'Sep':>8} {'Acc':>8} {'Same':>8} {'Diff':>8} {'N':>10}")
    print("-" * 76)
    for dataset_name, (same_pairs, diff_pairs) in datasets.items():
        for method in ("512D-baseline", "128D-trained"):
            same_sims = []
            diff_sims = []
            for a, b in same_pairs:
                if a in cache and b in cache:
                    ea, eb = cache[a], cache[b]
                    same_sims.append(cosine_similarity(ea, eb) if method == "512D-baseline" else cosine_similarity(project(ea), project(eb)))
            for a, b in diff_pairs:
                if a in cache and b in cache:
                    ea, eb = cache[a], cache[b]
                    diff_sims.append(cosine_similarity(ea, eb) if method == "512D-baseline" else cosine_similarity(project(ea), project(eb)))

            ss = np.asarray(same_sims)
            ds = np.asarray(diff_sims)
            thresholds = np.linspace(-1, 1, 400)
            acc = max((np.sum(ss >= t) + np.sum(ds < t)) / (len(ss) + len(ds)) for t in thresholds)
            print(
                f"{dataset_name:<8} {method:<16} {ss.mean() - ds.mean():>8.4f} {acc:>8.1%} "
                f"{ss.mean():>8.4f} {ds.mean():>8.4f} {len(ss)}s/{len(ds)}d"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=1200)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-pairs", type=int, default=80)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        print(f"Changing cwd to {SCRIPT_DIR}")
        os.chdir(SCRIPT_DIR)

    start = time.time()
    _, embeddings = compute_or_load_embeddings(args.num_train, args.cache)
    mean, projection = train_projection(
        embeddings=embeddings,
        output_dim=args.output_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    projected = np.asarray([(emb - mean) @ projection for emb in embeddings], dtype=np.float32)
    scale = np.max(np.abs(projection)) / 127.0
    projection_i8 = np.round(projection / scale).clip(-128, 127).astype(np.int8)
    np.savez(
        args.output,
        mean=mean,
        projection=projection,
        projection_i8=projection_i8,
        projection_i8_scale=np.asarray(scale, dtype=np.float32),
        train_projected_mean=projected.mean(axis=0).astype(np.float32),
        train_projected_std=projected.std(axis=0).astype(np.float32),
    )
    print(f"\nSaved projection: {args.output}")
    print(f"  float32 matrix: {projection.nbytes / 1024:.1f} KiB")
    print(f"  int8 matrix: {projection_i8.nbytes / 1024:.1f} KiB, scale={scale:.8f}")

    datasets, cache = compute_eval_cache(args.eval_pairs)
    evaluate_projection(datasets, cache, mean, projection)
    print(f"\nDone in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
