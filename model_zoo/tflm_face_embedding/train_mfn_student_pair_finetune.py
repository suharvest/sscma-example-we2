#!/usr/bin/env python3
"""Fine-tune a compact 128D student with labeled LFW pairs.

This is a follow-up to train_mfn_student_distill.py. The distillation-only S2
student meets the SRAM target but keeps too much different-person similarity on
LFW. This script keeps the same width/output architecture and adds supervised
pair losses from LFW DevTrain match/mismatch pairs.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

import compute_embedding
from compute_embedding import FaceEmbeddingPipeline
from train_mfn_student_distill import (
    PROJECTION_NPZ,
    TEACHER_MODEL,
    build_student,
    compute_teacher_embeddings,
    export_tflite,
    projected_targets,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LFW_DIR = SCRIPT_DIR / "datasets" / "lfw"
CFP_DIR = SCRIPT_DIR / "datasets" / "cfp" / "cfp-dataset"
MATCH_CSV = SCRIPT_DIR / "calibration_data" / "matchpairsDevTrain.csv"
MISMATCH_CSV = SCRIPT_DIR / "calibration_data" / "mismatchpairsDevTrain.csv"
SCRFD_MODEL = SCRIPT_DIR / "scrfd" / "models" / "scrfd_500m_kps_int8.tflite"
OUT_DIR = SCRIPT_DIR / "official_mobilefacenet" / "student_distill_w1_pairft"


def lfw_image_path(name, index):
    return LFW_DIR / name / f"{name}_{int(index):04d}.jpg"


def parse_split_ids(value):
    if not value:
        return []
    splits = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            splits.extend(range(int(start), int(end) + 1))
        else:
            splits.append(int(item))
    return sorted(set(splits))


def load_cfp_list(path):
    mapping = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    mapping[int(parts[0])] = parts[1]
    return mapping


def load_cfp_split_pairs(split_id, max_pairs_per_split):
    frontal_map = load_cfp_list(CFP_DIR / "Protocol" / "Pair_list_F.txt")
    profile_map = load_cfp_list(CFP_DIR / "Protocol" / "Pair_list_P.txt")

    def idx_to_path(idx, mapping):
        img_path = mapping.get(idx, "")
        full_path = (CFP_DIR / "Protocol" / img_path).resolve()
        return full_path if full_path.exists() else None

    def read_pairs(path, label):
        result = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                f_idx, p_idx = line.split(",", 1)
                f_path = idx_to_path(int(f_idx), frontal_map)
                p_path = idx_to_path(int(p_idx), profile_map)
                if f_path and p_path:
                    result.append((f_path, p_path, label))
                if max_pairs_per_split and len(result) >= max_pairs_per_split:
                    break
        return result

    split_dir = CFP_DIR / "Protocol" / "Split" / "FP" / f"{split_id:02d}"
    return read_pairs(split_dir / "same.txt", 1.0) + read_pairs(split_dir / "diff.txt", 0.0)


def load_pairs(max_pairs=None, cfp_splits="", cfp_max_pairs_per_split=0):
    paths = []
    path_to_idx = {}
    pairs = []

    def add_path(path):
        key = str(path)
        idx = path_to_idx.get(key)
        if idx is None:
            idx = len(paths)
            path_to_idx[key] = idx
            paths.append(path)
        return idx

    with MATCH_CSV.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            a = add_path(lfw_image_path(row[0], row[1]))
            b = add_path(lfw_image_path(row[0], row[2]))
            pairs.append((a, b, 1.0))
            if max_pairs and len(pairs) >= max_pairs:
                break

    pos_count = len(pairs)
    with MISMATCH_CSV.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            a = add_path(lfw_image_path(row[0], row[1]))
            b = add_path(lfw_image_path(row[2], row[3]))
            pairs.append((a, b, 0.0))
            if max_pairs and len(pairs) >= pos_count + max_pairs:
                break

    for split_id in parse_split_ids(cfp_splits):
        before = len(pairs)
        for a_path, b_path, label in load_cfp_split_pairs(split_id, cfp_max_pairs_per_split):
            a = add_path(a_path)
            b = add_path(b_path)
            pairs.append((a, b, label))
        print(f"Loaded CFP-FP split {split_id:02d}: {len(pairs) - before} pairs")

    return paths, np.asarray(pairs, dtype=np.float32)


def align_images(paths, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        cached_paths = [str(p) for p in data["paths"]]
        print(f"Loaded aligned cache: {cache_path}")
        return data["images"].astype(np.uint8), np.asarray(cached_paths)

    compute_embedding.MAX_FACE_RATIO = 1.0
    compute_embedding.ALLOW_CENTER_CROP_FALLBACK = True
    pipeline = FaceEmbeddingPipeline(str(SCRFD_MODEL), str(TEACHER_MODEL), backend="tflite")

    images = []
    kept_paths = []
    kept_index = {}
    for i, path in enumerate(paths):
        try:
            result = pipeline.compute(Image.open(path).convert("RGB"), debug=False)
            kept_index[i] = len(images)
            images.append(result["aligned_face"].astype(np.uint8))
            kept_paths.append(str(path))
        except Exception as exc:
            print(f"skip {path}: {exc}")
        if (i + 1) % 250 == 0:
            print(f"  aligned {i + 1}/{len(paths)}", flush=True)

    if not images:
        raise RuntimeError("No LFW images aligned")

    np.savez(cache_path, images=np.stack(images), paths=np.asarray(kept_paths))
    print(f"Saved aligned cache: {cache_path}")

    return np.stack(images), np.asarray(kept_paths)


def remap_pairs(paths, kept_paths, pairs):
    kept_lookup = {p: i for i, p in enumerate(kept_paths)}
    remapped = []
    for a, b, label in pairs:
        a_path = str(paths[int(a)])
        b_path = str(paths[int(b)])
        if a_path in kept_lookup and b_path in kept_lookup:
            remapped.append((kept_lookup[a_path], kept_lookup[b_path], label))
    return np.asarray(remapped, dtype=np.float32)


def fine_tune(
    model,
    images,
    target128,
    pairs,
    epochs,
    batch_size,
    lr,
    checkpoint_path,
    checkpoint_every,
    distill_weight,
    positive_weight,
    negative_weight,
    negative_margin,
    threshold_weight,
    threshold,
    threshold_margin,
):
    x = (images.astype(np.float32) / 127.5) - 1.0
    pair_ds = tf.data.Dataset.from_tensor_slices(pairs)
    pair_ds = pair_ds.shuffle(len(pairs), reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    x_tf = tf.constant(x, dtype=tf.float32)
    target_tf = tf.constant(target128, dtype=tf.float32)

    @tf.function
    def step(batch_pairs):
        idx_a = tf.cast(batch_pairs[:, 0], tf.int32)
        idx_b = tf.cast(batch_pairs[:, 1], tf.int32)
        labels = batch_pairs[:, 2]
        with tf.GradientTape() as tape:
            xa = tf.gather(x_tf, idx_a)
            xb = tf.gather(x_tf, idx_b)
            ta = tf.gather(target_tf, idx_a)
            tb = tf.gather(target_tf, idx_b)

            pa = tf.math.l2_normalize(model(xa, training=True), axis=-1)
            pb = tf.math.l2_normalize(model(xb, training=True), axis=-1)
            ta = tf.math.l2_normalize(ta, axis=-1)
            tb = tf.math.l2_normalize(tb, axis=-1)

            sim = tf.reduce_sum(pa * pb, axis=-1)
            distill_loss = 0.5 * tf.reduce_mean(
                1.0 - tf.reduce_sum(pa * ta, axis=-1)
                + 1.0 - tf.reduce_sum(pb * tb, axis=-1)
            )

            pos_mask = labels
            neg_mask = 1.0 - labels
            pos_loss = tf.reduce_sum((1.0 - sim) * pos_mask) / (tf.reduce_sum(pos_mask) + 1e-6)
            neg_loss = (
                tf.reduce_sum(tf.square(tf.nn.relu(sim - negative_margin)) * neg_mask)
                / (tf.reduce_sum(neg_mask) + 1e-6)
            )
            pos_threshold = threshold + threshold_margin
            neg_threshold = threshold - threshold_margin
            threshold_loss = (
                tf.reduce_sum(tf.square(tf.nn.relu(pos_threshold - sim)) * pos_mask)
                / (tf.reduce_sum(pos_mask) + 1e-6)
                + tf.reduce_sum(tf.square(tf.nn.relu(sim - neg_threshold)) * neg_mask)
                / (tf.reduce_sum(neg_mask) + 1e-6)
            )
            loss = (
                distill_weight * distill_loss
                + positive_weight * pos_loss
                + negative_weight * neg_loss
                + threshold_weight * threshold_loss
            )

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, distill_loss, pos_loss, neg_loss, threshold_loss

    for epoch in range(1, epochs + 1):
        losses = []
        distill_losses = []
        pos_losses = []
        neg_losses = []
        threshold_losses = []
        for batch_pairs in pair_ds:
            loss, distill_loss, pos_loss, neg_loss, threshold_loss = step(batch_pairs)
            losses.append(float(loss))
            distill_losses.append(float(distill_loss))
            pos_losses.append(float(pos_loss))
            neg_losses.append(float(neg_loss))
            threshold_losses.append(float(threshold_loss))

        print(
            f"epoch {epoch:03d}/{epochs} "
            f"loss={np.mean(losses):.5f} "
            f"distill={np.mean(distill_losses):.5f} "
            f"pos={np.mean(pos_losses):.5f} "
            f"neg={np.mean(neg_losses):.5f} "
            f"thr={np.mean(threshold_losses):.5f}",
            flush=True,
        )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            model.save_weights(checkpoint_path)
            print(f"checkpoint: {checkpoint_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--projection", type=Path, default=PROJECTION_NPZ)
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--distill-weight", type=float, default=0.3)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--negative-weight", type=float, default=8.0)
    parser.add_argument("--negative-margin", type=float, default=0.05)
    parser.add_argument("--threshold-weight", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--threshold-margin", type=float, default=0.04)
    parser.add_argument("--cfp-splits", default="", help="Optional CFP-FP train splits, e.g. 2-10. Split 01 is reserved for evaluation.")
    parser.add_argument("--cfp-max-pairs-per-split", type=int, default=0)
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    start = time.time()
    name = f"mfn_w{args.width:g}_pairft_{args.embedding_dim}d"
    out_prefix = args.out_dir / name
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths, pairs = load_pairs(args.max_pairs, args.cfp_splits, args.cfp_max_pairs_per_split)
    print(f"Loaded {len(paths)} unique LFW paths and {len(pairs)} pairs")
    images, kept_paths = align_images(paths, args.out_dir / f"{name}_aligned_lfw.npz")
    pairs = remap_pairs(paths, kept_paths, pairs)
    print(f"Kept {len(images)} aligned images and {len(pairs)} valid pairs")

    teacher512 = compute_teacher_embeddings(images, args.out_dir / f"{name}_teacher512_{len(images)}.npz")
    target128 = projected_targets(teacher512, args.projection)

    model = build_student(width=args.width, embedding_dim=args.embedding_dim)
    model.load_weights(args.init_weights)
    print(f"Loaded initial weights: {args.init_weights}")
    print(f"Model params: {model.count_params():,}")

    weights_path = args.out_dir / f"{name}.weights.h5"
    fine_tune(
        model,
        images,
        target128,
        pairs,
        args.epochs,
        args.batch_size,
        args.lr,
        weights_path,
        args.checkpoint_every,
        args.distill_weight,
        args.positive_weight,
        args.negative_weight,
        args.negative_margin,
        args.threshold_weight,
        args.threshold,
        args.threshold_margin,
    )
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")
    export_tflite(model, images, out_prefix, args.num_calib)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
