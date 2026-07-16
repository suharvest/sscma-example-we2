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
    l2_np,
    load_teacher,
    projected_targets,
    teacher_input_from_uint8,
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


def load_identity_paths(identity_dirs, max_identity_images=0):
    if not identity_dirs:
        return []

    paths = []
    for value in identity_dirs.split(","):
        root = Path(value.strip())
        if not root:
            continue
        if not root.is_absolute():
            root = SCRIPT_DIR / root
        if not root.exists():
            raise FileNotFoundError(f"Identity dataset not found: {root}")

        found = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if max_identity_images and len(found) > max_identity_images:
            rng = np.random.default_rng(42)
            found = [found[i] for i in rng.permutation(len(found))[:max_identity_images]]
        paths.extend(found)
        print(f"Loaded identity dataset paths: {root} images={len(found)}")
    return paths


def identity_roots(identity_dirs):
    roots = []
    if not identity_dirs:
        return roots
    for value in identity_dirs.split(","):
        root = Path(value.strip())
        if not root:
            continue
        if not root.is_absolute():
            root = SCRIPT_DIR / root
        roots.append(root.resolve())
    return roots


def is_under(path, roots):
    resolved = Path(path).resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            pass
    return False


def load_prealigned_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (112, 112):
        img = img.resize((112, 112), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def align_images(paths, cache_path, prealigned_roots=None):
    prealigned_roots = prealigned_roots or []
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
    prealigned_count = 0
    for i, path in enumerate(paths):
        try:
            if is_under(path, prealigned_roots):
                aligned_face = load_prealigned_image(path)
                prealigned_count += 1
            else:
                result = pipeline.compute(Image.open(path).convert("RGB"), debug=False)
                aligned_face = result["aligned_face"].astype(np.uint8)
            kept_index[i] = len(images)
            images.append(aligned_face)
            kept_paths.append(str(path))
        except Exception as exc:
            print(f"skip {path}: {exc}")
        if (i + 1) % 250 == 0:
            print(f"  aligned {i + 1}/{len(paths)}", flush=True)

    if not images:
        raise RuntimeError("No LFW images aligned")

    np.savez(cache_path, images=np.stack(images), paths=np.asarray(kept_paths))
    print(f"Saved aligned cache: {cache_path} (prealigned={prealigned_count})")

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


def identity_from_path(path):
    parts = Path(str(path)).parts
    if "lfw" in parts:
        idx = parts.index("lfw")
        if idx + 1 < len(parts):
            return f"lfw:{parts[idx + 1]}"
    if "Images" in parts:
        idx = parts.index("Images")
        if idx + 1 < len(parts):
            return f"cfp:{parts[idx + 1]}"
    for idx, part in enumerate(parts):
        if part.startswith("glint360k_") and idx + 1 < len(parts):
            return f"glint:{parts[idx + 1]}"
    return str(path)


def identity_labels_for_paths(paths, class_to_label):
    labels = [class_to_label.get(identity_from_path(path), -1) for path in paths]
    return np.asarray(labels, dtype=np.int32)


def build_identity_label_sets(kept_paths, identity_paths, min_images):
    all_paths = list(kept_paths) + [str(path) for path in identity_paths]
    identities = [identity_from_path(path) for path in all_paths]
    counts = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1

    class_names = sorted(identity for identity, count in counts.items() if count >= min_images)
    class_to_label = {identity: i for i, identity in enumerate(class_names)}
    pair_labels = identity_labels_for_paths(kept_paths, class_to_label)
    identity_labels = identity_labels_for_paths(identity_paths, class_to_label)
    valid_images = int(np.sum(pair_labels >= 0) + np.sum(identity_labels >= 0))
    print(
        f"ArcFace identities: classes={len(class_names)} "
        f"valid_images={valid_images}/{len(all_paths)} min_images={min_images} "
        f"pair_valid={int(np.sum(pair_labels >= 0))}/{len(pair_labels)} "
        f"stream_valid={int(np.sum(identity_labels >= 0))}/{len(identity_labels)}",
        flush=True,
    )
    valid_stream = identity_labels >= 0
    return (
        pair_labels,
        np.asarray([str(path) for path in identity_paths], dtype=str)[valid_stream],
        identity_labels[valid_stream],
        class_names,
    )


def compute_identity_projected_targets(paths, projection_path, cache_path):
    paths = [str(path) for path in paths]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        cached_paths = [str(path) for path in data["paths"]]
        if cached_paths == paths:
            print(f"Loaded identity target cache: {cache_path}", flush=True)
            return data["target128"].astype(np.float32)
        print(f"Ignoring stale identity target cache: {cache_path}", flush=True)

    projection_data = np.load(projection_path)
    mean = projection_data["mean"].astype(np.float32)
    projection = projection_data["projection"].astype(np.float32)
    interp, in_idx, out_idx, out_scale, out_zp = load_teacher()

    targets = []
    for i, path in enumerate(paths):
        image = load_prealigned_image(path)
        teacher_input = teacher_input_from_uint8(image[np.newaxis, ...])
        interp.set_tensor(in_idx, teacher_input)
        interp.invoke()
        emb = interp.get_tensor(out_idx).reshape(-1).astype(np.float32)
        emb = l2_np(((emb - out_zp) * out_scale)[np.newaxis, :])[0]
        target = l2_np(((emb - mean) @ projection)[np.newaxis, :])[0]
        targets.append(target.astype(np.float32))
        if (i + 1) % 5000 == 0:
            print(f"  identity teacher {i + 1}/{len(paths)}", flush=True)

    target128 = (
        np.stack(targets).astype(np.float32)
        if targets
        else np.empty((0, projection.shape[1]), dtype=np.float32)
    )
    np.savez(cache_path, paths=np.asarray(paths), target128=target128)
    print(f"Saved identity target cache: {cache_path}", flush=True)
    return target128


def mine_hard_negatives(model, images, kept_paths, max_pairs, batch_size):
    if max_pairs <= 0:
        return np.empty((0, 3), dtype=np.float32)

    x = (images.astype(np.float32) / 127.5) - 1.0
    embeddings = []
    for start in range(0, len(x), batch_size):
        pred = model(x[start:start + batch_size], training=False)
        pred = tf.math.l2_normalize(pred, axis=-1).numpy().astype(np.float32)
        embeddings.append(pred)
    embeddings = np.concatenate(embeddings, axis=0)

    identities = np.asarray([identity_from_path(path) for path in kept_paths])
    sim = embeddings @ embeddings.T
    same_identity = identities[:, None] == identities[None, :]
    upper = np.triu(np.ones(sim.shape, dtype=bool), k=1)
    valid = upper & ~same_identity
    scores = sim[valid]
    if len(scores) == 0:
        return np.empty((0, 3), dtype=np.float32)

    take = min(max_pairs, len(scores))
    candidate_indices = np.argpartition(scores, -take)[-take:]
    row_idx, col_idx = np.where(valid)
    selected = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    pairs = np.stack(
        [
            row_idx[selected].astype(np.float32),
            col_idx[selected].astype(np.float32),
            np.zeros(len(selected), dtype=np.float32),
        ],
        axis=1,
    )
    print(
        f"Mined {len(pairs)} hard negatives: "
        f"sim_max={scores[selected[0]]:.4f} sim_min={scores[selected[-1]]:.4f}",
        flush=True,
    )
    return pairs.astype(np.float32)


def compute_model_embeddings(model, images, batch_size):
    x = (images.astype(np.float32) / 127.5) - 1.0
    embeddings = []
    for start in range(0, len(x), batch_size):
        pred = model(x[start:start + batch_size], training=False)
        pred = tf.math.l2_normalize(pred, axis=-1).numpy().astype(np.float32)
        embeddings.append(pred)
    return np.concatenate(embeddings, axis=0)


def mine_hard_positives(model, images, pairs, max_pairs, batch_size):
    if max_pairs <= 0:
        return np.empty((0, 3), dtype=np.float32)

    positive_pairs = pairs[pairs[:, 2] > 0.5]
    if len(positive_pairs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    embeddings = compute_model_embeddings(model, images, batch_size)
    idx_a = positive_pairs[:, 0].astype(np.int32)
    idx_b = positive_pairs[:, 1].astype(np.int32)
    scores = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=-1)

    take = min(max_pairs, len(scores))
    selected = np.argpartition(scores, take - 1)[:take]
    selected = selected[np.argsort(scores[selected])]
    mined = positive_pairs[selected].astype(np.float32)
    print(
        f"Mined {len(mined)} hard positives: "
        f"sim_min={scores[selected[0]]:.4f} sim_max={scores[selected[-1]]:.4f}",
        flush=True,
    )
    return mined


def mine_teacher_hard_negatives(teacher512, kept_paths, max_pairs):
    if max_pairs <= 0:
        return np.empty((0, 3), dtype=np.float32)

    embeddings = l2_np(teacher512.astype(np.float32))
    identities = np.asarray([identity_from_path(path) for path in kept_paths])
    sim = embeddings @ embeddings.T
    same_identity = identities[:, None] == identities[None, :]
    upper = np.triu(np.ones(sim.shape, dtype=bool), k=1)
    valid = upper & ~same_identity
    scores = sim[valid]
    if len(scores) == 0:
        return np.empty((0, 3), dtype=np.float32)

    take = min(max_pairs, len(scores))
    candidate_indices = np.argpartition(scores, -take)[-take:]
    row_idx, col_idx = np.where(valid)
    selected = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    pairs = np.stack(
        [
            row_idx[selected].astype(np.float32),
            col_idx[selected].astype(np.float32),
            np.zeros(len(selected), dtype=np.float32),
        ],
        axis=1,
    )
    print(
        f"Mined {len(pairs)} teacher hard negatives: "
        f"sim_max={scores[selected[0]]:.4f} sim_min={scores[selected[-1]]:.4f}",
        flush=True,
    )
    return pairs.astype(np.float32)


def load_arcface_head(cache_path, class_names, embedding_dim):
    if cache_path is None or not cache_path.exists():
        return None

    data = np.load(cache_path, allow_pickle=False)
    cached_classes = [str(name) for name in data["class_names"]]
    weights = data["weights"].astype(np.float32)
    if cached_classes != list(class_names):
        print(f"Ignoring arcface head with different class order: {cache_path}", flush=True)
        return None
    if weights.shape != (len(class_names), embedding_dim):
        print(f"Ignoring arcface head with shape {weights.shape}: {cache_path}", flush=True)
        return None
    print(f"Loaded arcface head: {cache_path}", flush=True)
    return weights


def save_arcface_head(cache_path, class_names, arcface_weights):
    if cache_path is None or arcface_weights is None:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        class_names=np.asarray(class_names, dtype=str),
        weights=arcface_weights.numpy().astype(np.float32),
    )
    print(f"Saved arcface head: {cache_path}", flush=True)


def fine_tune(
    model,
    images,
    teacher512,
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
    teacher_pair_weight,
    hard_positive_fraction,
    hard_negative_fraction,
    identity_label_values,
    identity_stream_paths,
    identity_stream_labels,
    arcface_weight,
    arcface_scale,
    arcface_margin,
    arcface_num_classes,
    arcface_steps_per_epoch,
    identity_distill_weight,
    identity_stream_targets,
    class_names,
    arcface_head_in,
    arcface_head_out,
):
    x = (images.astype(np.float32) / 127.5) - 1.0
    pair_ds = tf.data.Dataset.from_tensor_slices(pairs)
    pair_ds = pair_ds.shuffle(len(pairs), reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    x_tf = tf.constant(x, dtype=tf.float32)
    teacher_tf = tf.constant(teacher512, dtype=tf.float32)
    target_tf = tf.constant(target128, dtype=tf.float32)
    labels_tf = tf.constant(identity_label_values, dtype=tf.int32)
    embedding_dim = int(model.output_shape[-1])
    arcface_weights = None
    if arcface_weight > 0 and arcface_num_classes > 1:
        initial_arcface_weights = load_arcface_head(arcface_head_in, class_names, embedding_dim)
        init = tf.keras.initializers.GlorotUniform()
        arcface_weights = tf.Variable(
            initial_arcface_weights
            if initial_arcface_weights is not None
            else init(shape=(arcface_num_classes, embedding_dim), dtype=tf.float32),
            trainable=True,
            name="arcface_weights",
        )
    identity_ds = None
    if (arcface_weight > 0 or identity_distill_weight > 0) and len(identity_stream_paths):
        def load_identity_sample(path, label):
            image = tf.io.read_file(path)
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, [112, 112], method=tf.image.ResizeMethod.BILINEAR)
            image = tf.cast(image, tf.float32)
            image = (image / 127.5) - 1.0
            image.set_shape([112, 112, 3])
            return image, label

        def load_identity_sample_with_target(path, label, target):
            image, label = load_identity_sample(path, label)
            return image, label, target

        identity_ds = (
            tf.data.Dataset.from_tensor_slices((identity_stream_paths, identity_stream_labels, identity_stream_targets))
            .shuffle(len(identity_stream_paths), reshuffle_each_iteration=True)
            .repeat()
            .map(load_identity_sample_with_target, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    def arcface_loss_for_embeddings(embeddings, labels):
        if arcface_weights is None:
            return tf.constant(0.0, dtype=tf.float32)

        valid = labels >= 0
        valid_embeddings = tf.boolean_mask(embeddings, valid)
        valid_labels = tf.boolean_mask(labels, valid)

        def compute_loss():
            emb = tf.math.l2_normalize(valid_embeddings, axis=-1)
            weights = tf.math.l2_normalize(arcface_weights, axis=-1)
            cosine = tf.matmul(emb, weights, transpose_b=True)
            cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

            target_cosine = tf.gather(cosine, valid_labels, axis=1, batch_dims=1)
            sine = tf.sqrt(tf.maximum(1.0 - tf.square(target_cosine), 0.0))
            margin_cos = tf.cos(tf.constant(arcface_margin, dtype=tf.float32))
            margin_sin = tf.sin(tf.constant(arcface_margin, dtype=tf.float32))
            threshold_cos = tf.cos(tf.constant(np.pi, dtype=tf.float32) - arcface_margin)
            mm = tf.sin(tf.constant(np.pi, dtype=tf.float32) - arcface_margin) * arcface_margin
            phi = target_cosine * margin_cos - sine * margin_sin
            phi = tf.where(target_cosine > threshold_cos, phi, target_cosine - mm)

            one_hot = tf.one_hot(valid_labels, arcface_num_classes, dtype=tf.float32)
            logits = (one_hot * tf.expand_dims(phi, axis=1) + (1.0 - one_hot) * cosine) * arcface_scale
            return tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(valid_labels, logits, from_logits=True)
            )

        return tf.cond(tf.size(valid_labels) > 0, compute_loss, lambda: tf.constant(0.0, dtype=tf.float32))

    def masked_mean(values, mask):
        return tf.reduce_sum(values * mask) / (tf.reduce_sum(mask) + 1e-6)

    def masked_hard_mean(values, mask, fraction):
        selected = tf.boolean_mask(values, mask > 0)

        def top_mean():
            count = tf.size(selected)
            k_float = tf.cast(count, tf.float32) * tf.constant(fraction, dtype=tf.float32)
            k = tf.maximum(1, tf.cast(tf.math.ceil(k_float), tf.int32))
            top_values, _ = tf.math.top_k(selected, k=k, sorted=False)
            return tf.reduce_mean(top_values)

        return tf.cond(
            tf.logical_and(tf.size(selected) > 0, tf.constant(fraction, dtype=tf.float32) > 0.0),
            top_mean,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    @tf.function
    def step(batch_pairs):
        idx_a = tf.cast(batch_pairs[:, 0], tf.int32)
        idx_b = tf.cast(batch_pairs[:, 1], tf.int32)
        labels = batch_pairs[:, 2]
        with tf.GradientTape() as tape:
            xa = tf.gather(x_tf, idx_a)
            xb = tf.gather(x_tf, idx_b)
            teacher_a = tf.gather(teacher_tf, idx_a)
            teacher_b = tf.gather(teacher_tf, idx_b)
            ta = tf.gather(target_tf, idx_a)
            tb = tf.gather(target_tf, idx_b)
            label_a = tf.gather(labels_tf, idx_a)
            label_b = tf.gather(labels_tf, idx_b)

            pa = tf.math.l2_normalize(model(xa, training=True), axis=-1)
            pb = tf.math.l2_normalize(model(xb, training=True), axis=-1)
            teacher_a = tf.math.l2_normalize(teacher_a, axis=-1)
            teacher_b = tf.math.l2_normalize(teacher_b, axis=-1)
            ta = tf.math.l2_normalize(ta, axis=-1)
            tb = tf.math.l2_normalize(tb, axis=-1)

            sim = tf.reduce_sum(pa * pb, axis=-1)
            teacher_sim = tf.reduce_sum(teacher_a * teacher_b, axis=-1)
            distill_loss = 0.5 * tf.reduce_mean(
                1.0 - tf.reduce_sum(pa * ta, axis=-1)
                + 1.0 - tf.reduce_sum(pb * tb, axis=-1)
            )
            teacher_pair_loss = tf.reduce_mean(tf.square(sim - teacher_sim))

            pos_mask = labels
            neg_mask = 1.0 - labels
            pos_values = 1.0 - sim
            neg_values = tf.square(tf.nn.relu(sim - negative_margin))
            if hard_positive_fraction > 0:
                pos_loss = masked_hard_mean(pos_values, pos_mask, hard_positive_fraction)
            else:
                pos_loss = masked_mean(pos_values, pos_mask)
            if hard_negative_fraction > 0:
                neg_loss = masked_hard_mean(neg_values, neg_mask, hard_negative_fraction)
            else:
                neg_loss = masked_mean(neg_values, neg_mask)
            pos_threshold = threshold + threshold_margin
            neg_threshold = threshold - threshold_margin
            pos_threshold_values = tf.square(tf.nn.relu(pos_threshold - sim))
            neg_threshold_values = tf.square(tf.nn.relu(sim - neg_threshold))
            threshold_loss = (
                (
                    masked_hard_mean(pos_threshold_values, pos_mask, hard_positive_fraction)
                    if hard_positive_fraction > 0
                    else masked_mean(pos_threshold_values, pos_mask)
                )
                + (
                    masked_hard_mean(neg_threshold_values, neg_mask, hard_negative_fraction)
                    if hard_negative_fraction > 0
                    else masked_mean(neg_threshold_values, neg_mask)
                )
            )
            arcface_loss = arcface_loss_for_embeddings(tf.concat([pa, pb], axis=0), tf.concat([label_a, label_b], axis=0))
            loss = (
                distill_weight * distill_loss
                + positive_weight * pos_loss
                + negative_weight * neg_loss
                + threshold_weight * threshold_loss
                + teacher_pair_weight * teacher_pair_loss
                + arcface_weight * arcface_loss
            )

        trainable_variables = model.trainable_variables
        if arcface_weights is not None:
            trainable_variables = trainable_variables + [arcface_weights]
        grads = tape.gradient(loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads, trainable_variables))
        return loss, distill_loss, pos_loss, neg_loss, threshold_loss, teacher_pair_loss, arcface_loss

    @tf.function
    def arcface_step(batch_x, labels, target128_batch):
        with tf.GradientTape() as tape:
            pred = model(batch_x, training=True)
            pred_n = tf.math.l2_normalize(pred, axis=-1)
            arcface_loss = arcface_loss_for_embeddings(pred_n, labels)
            target_n = tf.math.l2_normalize(target128_batch, axis=-1)
            identity_distill_loss = tf.reduce_mean(1.0 - tf.reduce_sum(pred_n * target_n, axis=-1))
            loss = arcface_weight * arcface_loss + identity_distill_weight * identity_distill_loss

        trainable_variables = model.trainable_variables
        if arcface_weights is not None:
            trainable_variables = trainable_variables + [arcface_weights]
        grads = tape.gradient(loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads, trainable_variables))
        return loss, arcface_loss, identity_distill_loss

    for epoch in range(1, epochs + 1):
        losses = []
        distill_losses = []
        pos_losses = []
        neg_losses = []
        threshold_losses = []
        teacher_pair_losses = []
        arcface_losses = []
        identity_arcface_losses = []
        identity_distill_losses = []
        for batch_pairs in pair_ds:
            loss, distill_loss, pos_loss, neg_loss, threshold_loss, teacher_pair_loss, arcface_loss = step(batch_pairs)
            losses.append(float(loss))
            distill_losses.append(float(distill_loss))
            pos_losses.append(float(pos_loss))
            neg_losses.append(float(neg_loss))
            threshold_losses.append(float(threshold_loss))
            teacher_pair_losses.append(float(teacher_pair_loss))
            arcface_losses.append(float(arcface_loss))

        if identity_ds is not None and arcface_steps_per_epoch > 0:
            identity_iter = iter(identity_ds)
            for _ in range(arcface_steps_per_epoch):
                batch_x, batch_labels, batch_targets = next(identity_iter)
                _, identity_arcface_loss, identity_distill_loss = arcface_step(batch_x, batch_labels, batch_targets)
                identity_arcface_losses.append(float(identity_arcface_loss))
                identity_distill_losses.append(float(identity_distill_loss))

        print(
            f"epoch {epoch:03d}/{epochs} "
            f"loss={np.mean(losses):.5f} "
            f"distill={np.mean(distill_losses):.5f} "
            f"pos={np.mean(pos_losses):.5f} "
            f"neg={np.mean(neg_losses):.5f} "
            f"thr={np.mean(threshold_losses):.5f} "
            f"tpair={np.mean(teacher_pair_losses):.5f} "
            f"arc={np.mean(arcface_losses):.5f} "
            f"id_distill={np.mean(identity_distill_losses) if identity_distill_losses else 0.0:.5f} "
            f"id_arc={np.mean(identity_arcface_losses) if identity_arcface_losses else 0.0:.5f}",
            flush=True,
        )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            model.save_weights(checkpoint_path)
            save_arcface_head(arcface_head_out, class_names, arcface_weights)
            print(f"checkpoint: {checkpoint_path}", flush=True)

    save_arcface_head(arcface_head_out, class_names, arcface_weights)


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
    parser.add_argument("--teacher-pair-weight", type=float, default=0.0)
    parser.add_argument("--hard-positive-fraction", type=float, default=0.0)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.0)
    parser.add_argument("--mine-hard-positives", type=int, default=0)
    parser.add_argument("--mine-hard-negatives", type=int, default=0)
    parser.add_argument("--mine-teacher-hard-negatives", type=int, default=0)
    parser.add_argument("--arcface-weight", type=float, default=0.0)
    parser.add_argument("--arcface-scale", type=float, default=32.0)
    parser.add_argument("--arcface-margin", type=float, default=0.35)
    parser.add_argument("--arcface-min-images", type=int, default=2)
    parser.add_argument("--arcface-steps-per-epoch", type=int, default=0)
    parser.add_argument("--identity-distill-weight", type=float, default=0.0)
    parser.add_argument("--identity-target-cache", type=Path)
    parser.add_argument("--arcface-head-in", type=Path)
    parser.add_argument("--arcface-head-out", type=Path)
    parser.add_argument("--aligned-cache", type=Path)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--identity-dirs", default="", help="Optional comma-separated identity image roots, e.g. datasets/glint360k_subset_112.")
    parser.add_argument("--max-identity-images", type=int, default=0)
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
    identity_paths = load_identity_paths(args.identity_dirs, args.max_identity_images)
    print(
        f"Loaded {len(paths)} pair image paths, {len(identity_paths)} streaming identity paths, "
        f"and {len(pairs)} pairs"
    )
    aligned_cache = args.aligned_cache or args.out_dir / f"{name}_aligned_lfw.npz"
    images, kept_paths = align_images(paths, aligned_cache)
    pairs = remap_pairs(paths, kept_paths, pairs)
    print(f"Kept {len(images)} aligned images and {len(pairs)} valid pairs")

    teacher_cache = args.teacher_cache or args.out_dir / f"{name}_teacher512_{len(images)}.npz"
    teacher512 = compute_teacher_embeddings(images, teacher_cache)
    target128 = projected_targets(teacher512, args.projection)

    model = build_student(width=args.width, embedding_dim=args.embedding_dim)
    model.load_weights(args.init_weights)
    print(f"Loaded initial weights: {args.init_weights}")
    print(f"Model params: {model.count_params():,}")
    mined_positive_pairs = mine_hard_positives(
        model,
        images,
        pairs,
        args.mine_hard_positives,
        args.batch_size,
    )
    if len(mined_positive_pairs):
        pairs = np.concatenate([pairs, mined_positive_pairs], axis=0)
        print(f"Training pairs after hard-positive mining: {len(pairs)}")
    mined_pairs = mine_hard_negatives(model, images, kept_paths, args.mine_hard_negatives, args.batch_size)
    if len(mined_pairs):
        pairs = np.concatenate([pairs, mined_pairs], axis=0)
        print(f"Training pairs after student hard-negative mining: {len(pairs)}")
    teacher_mined_pairs = mine_teacher_hard_negatives(
        teacher512,
        kept_paths,
        args.mine_teacher_hard_negatives,
    )
    if len(teacher_mined_pairs):
        pairs = np.concatenate([pairs, teacher_mined_pairs], axis=0)
        print(f"Training pairs after teacher hard-negative mining: {len(pairs)}")

    id_labels, identity_stream_paths, identity_stream_labels, class_names = build_identity_label_sets(
        kept_paths,
        identity_paths,
        args.arcface_min_images,
    )
    identity_stream_targets = np.zeros((len(identity_stream_paths), args.embedding_dim), dtype=np.float32)
    if args.identity_distill_weight > 0 and len(identity_stream_paths):
        target_cache = args.identity_target_cache
        if target_cache is None:
            target_cache = args.out_dir / f"{name}_identity_targets_{len(identity_stream_paths)}.npz"
        identity_stream_targets = compute_identity_projected_targets(
            identity_stream_paths,
            args.projection,
            target_cache,
        )
    weights_path = args.out_dir / f"{name}.weights.h5"
    fine_tune(
        model,
        images,
        teacher512,
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
        args.teacher_pair_weight,
        args.hard_positive_fraction,
        args.hard_negative_fraction,
        id_labels,
        identity_stream_paths,
        identity_stream_labels,
        args.arcface_weight,
        args.arcface_scale,
        args.arcface_margin,
        len(class_names),
        args.arcface_steps_per_epoch,
        args.identity_distill_weight,
        identity_stream_targets,
        class_names,
        args.arcface_head_in,
        args.arcface_head_out or args.out_dir / f"{name}_arcface_head.npz",
    )
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")
    export_tflite(model, images, out_prefix, args.num_calib)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
