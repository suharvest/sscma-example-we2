#!/usr/bin/env python3
"""Train a wider MobileFaceNet student from an existing 128D TFLite teacher."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_mfn_student_distill import build_student, export_tflite, image_paths, load_images, l2_np


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "official_mobilefacenet" / "iddistill_w115_v2lfw6_clone"
TEACHER_MODEL = SCRIPT_DIR / "official_mobilefacenet" / "iddistill_v2_lfw6" / "mfn_w1_pairft_128d.int8.tflite"


def quantize_input(images, input_detail):
    x = (images.astype(np.float32) / 127.5) - 1.0
    q = input_detail.get("quantization_parameters", {})
    scales = np.asarray(q.get("scales", []), dtype=np.float32)
    zero_points = np.asarray(q.get("zero_points", []), dtype=np.int32)
    if scales.size:
        x = np.round(x / float(scales.flat[0]) + int(zero_points.flat[0]))
    return np.clip(x, -128, 127).astype(np.int8)


def dequantize_output(values, output_detail):
    q = output_detail.get("quantization_parameters", {})
    scales = np.asarray(q.get("scales", []), dtype=np.float32)
    zero_points = np.asarray(q.get("zero_points", []), dtype=np.int32)
    if scales.size:
        return (values.astype(np.float32) - int(zero_points.flat[0])) * float(scales.flat[0])
    return values.astype(np.float32)


def compute_teacher_embeddings(images, teacher_model, cache_path):
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"Loaded teacher cache: {cache_path}")
        return data["embeddings"].astype(np.float32)

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=str(teacher_model))
    interp.allocate_tensors()
    input_detail = interp.get_input_details()[0]
    output_detail = interp.get_output_details()[0]
    input_index = input_detail["index"]
    output_index = output_detail["index"]

    inputs = quantize_input(images, input_detail)
    outputs = []
    for i, image in enumerate(inputs, 1):
        interp.set_tensor(input_index, image[np.newaxis, ...])
        interp.invoke()
        out = interp.get_tensor(output_index)[0]
        outputs.append(dequantize_output(out, output_detail))
        if i % 1000 == 0:
            print(f"  teacher {i}/{len(inputs)}", flush=True)

    embeddings = l2_np(np.stack(outputs).astype(np.float32))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    print(f"Saved teacher cache: {cache_path}")
    return embeddings


def train(model, images, targets, epochs, batch_size, lr, checkpoint_path, checkpoint_every, target_weight, pair_weight, augment):
    x = (images.astype(np.float32) / 127.5) - 1.0
    ds = tf.data.Dataset.from_tensor_slices((x, targets))
    ds = ds.shuffle(len(images), reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    def augment_batch(batch_x):
        if not augment:
            return batch_x
        batch_x = tf.image.random_flip_left_right(batch_x)
        batch_x = tf.image.random_brightness(batch_x, max_delta=0.06)
        batch_x = tf.image.random_contrast(batch_x, lower=0.9, upper=1.1)
        return tf.clip_by_value(batch_x, -1.0, 1.0)

    @tf.function
    def step(batch_x, batch_target):
        with tf.GradientTape() as tape:
            batch_x = augment_batch(batch_x)
            pred = tf.math.l2_normalize(model(batch_x, training=True), axis=-1)
            target = tf.math.l2_normalize(batch_target, axis=-1)
            target_loss = tf.reduce_mean(1.0 - tf.reduce_sum(pred * target, axis=-1))
            sim_student = tf.linalg.matmul(pred, pred, transpose_b=True)
            sim_teacher = tf.linalg.matmul(target, target, transpose_b=True)
            batch_n = tf.shape(batch_x)[0]
            offdiag = 1.0 - tf.eye(batch_n, dtype=tf.float32)
            pair_loss = tf.reduce_sum(tf.square(sim_student - sim_teacher) * offdiag) / (tf.reduce_sum(offdiag) + 1e-6)
            value_loss = 0.0001 * tf.reduce_mean(tf.square(pred))
            loss = target_weight * target_loss + pair_weight * pair_loss + value_loss
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, target_loss, pair_loss

    for epoch in range(1, epochs + 1):
        losses = []
        target_losses = []
        pair_losses = []
        for batch_x, batch_target in ds:
            loss, target_loss, pair_loss = step(batch_x, batch_target)
            losses.append(float(loss))
            target_losses.append(float(target_loss))
            pair_losses.append(float(pair_loss))
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"loss={np.mean(losses):.5f} target={np.mean(target_losses):.5f} pair={np.mean(pair_losses):.5f}",
            flush=True,
        )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            model.save_weights(checkpoint_path)
            print(f"checkpoint: {checkpoint_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=1.15)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--teacher-model", type=Path, default=TEACHER_MODEL)
    parser.add_argument("--num-train", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=4.0)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    start = time.time()
    name = f"mfn_w{args.width:g}_clone_{args.embedding_dim}d"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.out_dir / name

    paths = image_paths(args.num_train)
    images, _ = load_images(paths)
    print(f"Loaded {len(images)} images for {name}")
    cache_path = args.out_dir / f"{name}_teacher128_{len(images)}.npz"
    targets = compute_teacher_embeddings(images, args.teacher_model, cache_path)

    model = build_student(width=args.width, embedding_dim=args.embedding_dim)
    print(f"Model params: {model.count_params():,}")
    weights_path = args.out_dir / f"{name}.weights.h5"
    train(
        model,
        images,
        targets,
        args.epochs,
        args.batch_size,
        args.lr,
        weights_path,
        args.checkpoint_every,
        args.target_weight,
        args.pair_weight,
        args.augment,
    )
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")
    export_tflite(model, images, out_prefix, args.num_calib)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
