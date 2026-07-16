#!/usr/bin/env python3
"""Pair fine-tune the stage96 official-inherited 128D model."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow_model_optimization as tfmot
import tensorflow as tf

from train_mfn_student_distill import export_tflite, l2_np
from train_mfn_student_pair_finetune import (
    align_images,
    load_pairs,
    mine_hard_negatives,
    remap_pairs,
)
from train_w600k_stage96_distill import build_model, compute_teacher_targets, image_paths, load_images


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "official_mobilefacenet" / "w600k_stage96_pairft_a"
LFW_DIR = SCRIPT_DIR / "datasets" / "lfw"


class _SkipQuantizeConfig:
    """Minimal TFMOT config for layers such as PReLU that are unsupported by default."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


def make_qat_model(model):
    class SkipQuantizeConfig(_SkipQuantizeConfig, tfmot.quantization.keras.QuantizeConfig):
        pass

    def annotate(layer):
        if isinstance(layer, tf.keras.layers.PReLU):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, SkipQuantizeConfig())
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

    annotated = tf.keras.models.clone_model(model, clone_function=annotate)
    annotated.set_weights(model.get_weights())
    with tfmot.quantization.keras.quantize_scope({"SkipQuantizeConfig": SkipQuantizeConfig}):
        return tfmot.quantization.keras.quantize_apply(annotated)


def load_weights_by_compatible_layer(model, init_weights, keep_channels):
    source = build_model(keep_channels=keep_channels, activation_override="none")
    source.load_weights(init_weights)
    source_layers = {layer.name: layer for layer in source.layers}
    copied, skipped = 0, []
    for layer in model.layers:
        src = source_layers.get(layer.name)
        if src is None:
            skipped.append(layer.name)
            continue
        src_weights = src.get_weights()
        dst_weights = layer.get_weights()
        if not src_weights and not dst_weights:
            continue
        if len(src_weights) != len(dst_weights):
            skipped.append(layer.name)
            continue
        if any(a.shape != b.shape for a, b in zip(src_weights, dst_weights)):
            skipped.append(layer.name)
            continue
        layer.set_weights(src_weights)
        copied += 1
    print(f"Copied compatible weights by layer: {copied}, skipped={skipped}")


def add_extra_lfw_pairs(paths, pairs, max_pairs_per_label):
    if max_pairs_per_label <= 0:
        return paths, pairs

    path_to_idx = {str(path): idx for idx, path in enumerate(paths)}

    def add_path(path):
        key = str(path)
        idx = path_to_idx.get(key)
        if idx is None:
            idx = len(paths)
            path_to_idx[key] = idx
            paths.append(path)
        return idx

    def person_images(person):
        return sorted(
            p
            for p in person.glob("*.jpg")
            if not p.name.startswith(".") and not p.name.startswith("._")
        )

    people = [(person, person_images(person)) for person in sorted(LFW_DIR.iterdir()) if person.is_dir()]
    people = [(person, imgs) for person, imgs in people if len(imgs) >= 2]
    rng = np.random.default_rng(42)

    same_candidates = []
    for _, imgs in people:
        limit = min(len(imgs), 4)
        for i in range(limit):
            for j in range(i + 1, limit):
                same_candidates.append((imgs[i], imgs[j], 1.0))
    rng.shuffle(same_candidates)

    diff_candidates = []
    order = rng.permutation(len(people))
    for offset in range(len(order)):
        a_person, a_imgs = people[int(order[offset])]
        b_person, b_imgs = people[int(order[(offset + len(order) // 2) % len(order)])]
        if a_person == b_person:
            continue
        diff_candidates.append((a_imgs[0], b_imgs[0], 0.0))
    rng.shuffle(diff_candidates)

    extra = same_candidates[:max_pairs_per_label] + diff_candidates[:max_pairs_per_label]
    if not extra:
        return paths, pairs

    extra_pairs = np.asarray([(add_path(a), add_path(b), label) for a, b, label in extra], dtype=np.float32)
    print(
        f"Loaded extra LFW pairs: {len(same_candidates[:max_pairs_per_label])} same, "
        f"{len(diff_candidates[:max_pairs_per_label])} diff"
    )
    return paths, np.concatenate([pairs, extra_pairs], axis=0)


def fine_tune(
    model,
    images,
    kept_paths,
    targets,
    pairs,
    epochs,
    batch_size,
    lr,
    distill_weight,
    positive_weight,
    negative_weight,
    negative_margin,
    hard_positive_weight,
    positive_margin,
    teacher_pair_weight,
    hard_negative_pairs,
):
    x = ((images.astype(np.float32) / 127.5) - 1.0).astype(np.float32)
    x_tf = tf.constant(x, dtype=tf.float32)
    target_tf = tf.constant(targets, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    if hard_negative_pairs > 0:
        mined = mine_hard_negatives(model, images, kept_paths, hard_negative_pairs, batch_size)
        pairs = np.concatenate([pairs, mined], axis=0) if len(mined) else pairs

    pair_ds = (
        tf.data.Dataset.from_tensor_slices(pairs.astype(np.float32))
        .shuffle(len(pairs), reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    @tf.function
    def step(batch_pairs):
        idx_a = tf.cast(batch_pairs[:, 0], tf.int32)
        idx_b = tf.cast(batch_pairs[:, 1], tf.int32)
        labels = batch_pairs[:, 2]
        with tf.GradientTape() as tape:
            xa = tf.gather(x_tf, idx_a)
            xb = tf.gather(x_tf, idx_b)
            ta = tf.math.l2_normalize(tf.gather(target_tf, idx_a), axis=-1)
            tb = tf.math.l2_normalize(tf.gather(target_tf, idx_b), axis=-1)
            pa = tf.math.l2_normalize(model(xa, training=True), axis=-1)
            pb = tf.math.l2_normalize(model(xb, training=True), axis=-1)
            sim = tf.reduce_sum(pa * pb, axis=-1)
            teacher_sim = tf.reduce_sum(ta * tb, axis=-1)

            distill_loss = 0.5 * tf.reduce_mean(
                1.0 - tf.reduce_sum(pa * ta, axis=-1)
                + 1.0 - tf.reduce_sum(pb * tb, axis=-1)
            )
            pos_mask = labels
            neg_mask = 1.0 - labels
            pos_loss = tf.reduce_sum((1.0 - sim) * pos_mask) / (tf.reduce_sum(pos_mask) + 1e-6)
            hard_pos_loss = tf.reduce_sum(tf.square(tf.nn.relu(positive_margin - sim)) * pos_mask) / (
                tf.reduce_sum(pos_mask) + 1e-6
            )
            neg_loss = tf.reduce_sum(tf.square(tf.nn.relu(sim - negative_margin)) * neg_mask) / (
                tf.reduce_sum(neg_mask) + 1e-6
            )
            teacher_pair_loss = tf.reduce_mean(tf.square(sim - teacher_sim))
            loss = (
                distill_weight * distill_loss
                + positive_weight * pos_loss
                + hard_positive_weight * hard_pos_loss
                + negative_weight * neg_loss
                + teacher_pair_weight * teacher_pair_loss
            )

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 3.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, distill_loss, pos_loss, hard_pos_loss, neg_loss, teacher_pair_loss

    for epoch in range(1, epochs + 1):
        values = []
        for batch_pairs in pair_ds:
            values.append([float(v) for v in step(batch_pairs)])
        means = np.asarray(values).mean(axis=0)
        print(
            f"epoch {epoch:03d}/{epochs} loss={means[0]:.5f} distill={means[1]:.5f} "
            f"pos={means[2]:.5f} hpos={means[3]:.5f} neg={means[4]:.5f} tpair={means[5]:.5f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-channels", type=int, default=96)
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--max-pairs", type=int, default=1200)
    parser.add_argument("--extra-lfw-pairs", type=int, default=0)
    parser.add_argument("--cfp-splits", default="1")
    parser.add_argument("--cfp-max-pairs-per-split", type=int, default=1200)
    parser.add_argument("--hard-negative-pairs", type=int, default=1000)
    parser.add_argument("--distill-weight", type=float, default=2.0)
    parser.add_argument("--positive-weight", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=8.0)
    parser.add_argument("--negative-margin", type=float, default=0.03)
    parser.add_argument("--hard-positive-weight", type=float, default=0.0)
    parser.add_argument("--positive-margin", type=float, default=0.35)
    parser.add_argument("--teacher-pair-weight", type=float, default=2.0)
    parser.add_argument("--teacher-backend", choices=["keras", "tflite"], default="keras")
    parser.add_argument(
        "--activation-override",
        choices=[
            "none",
            "suspect-leaky",
            "suspect-relu6",
            "prelu28-leaky",
            "prelu28-relu6",
            "prelu30-leaky",
            "prelu30-relu6",
            "all-leaky",
            "all-relu6",
        ],
        default="none",
        help="Replace selected PReLU layers with WE2-friendlier activations.",
    )
    parser.add_argument("--qat", action="store_true", help="Enable TFMOT quantization-aware fine-tuning.")
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--aligned-cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths, pairs = load_pairs(args.max_pairs, args.cfp_splits, args.cfp_max_pairs_per_split)
    paths, pairs = add_extra_lfw_pairs(paths, pairs, args.extra_lfw_pairs)
    aligned_cache = args.aligned_cache or args.out_dir / "aligned_pair_images.npz"
    images, kept_paths = align_images(paths, aligned_cache)
    pairs = remap_pairs(paths, kept_paths, pairs)
    print(f"Pair images={len(images)} pairs={len(pairs)}")

    targets = compute_teacher_targets(
        images,
        args.out_dir / f"teacher_{args.teacher_backend}_dense128_pairs_{len(images)}.npz",
        args.teacher_backend,
        args.batch_size,
    )
    model = build_model(keep_channels=args.keep_channels, activation_override=args.activation_override)
    if args.activation_override == "none":
        model.load_weights(args.init_weights)
    else:
        load_weights_by_compatible_layer(model, args.init_weights, args.keep_channels)
    print(f"Loaded init weights: {args.init_weights} (activation_override={args.activation_override})")
    if args.qat:
        model = make_qat_model(model)
        print("Enabled QAT model wrappers")
    fine_tune(
        model,
        images,
        kept_paths,
        targets,
        pairs,
        args.epochs,
        args.batch_size,
        args.lr,
        args.distill_weight,
        args.positive_weight,
        args.negative_weight,
        args.negative_margin,
        args.hard_positive_weight,
        args.positive_margin,
        args.teacher_pair_weight,
        args.hard_negative_pairs,
    )

    weights_path = args.out_dir / "w600k_stage96_pairft.weights.h5"
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")

    calib_paths = image_paths(args.num_calib, "", 0)
    calib_images = load_images(calib_paths)
    export_tflite(model, calib_images, args.out_dir / "w600k_stage96_pairft", args.num_calib)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
