#!/usr/bin/env python3
"""Train a compact MobileFaceNet student from w600k teacher embeddings.

S2 target: width=0.75, output=128D, Vela SRAM < 1 MiB.
S3 target: width=0.50 after S2 passes.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_DIR = SCRIPT_DIR / "calibration_data" / "qat_112"
TEACHER_MODEL = SCRIPT_DIR / "official_mobilefacenet" / "w600k_mbf_int8.tflite"
PROJECTION_NPZ = SCRIPT_DIR / "outputs" / "w600k_projection_128d.npz"
OUT_DIR = SCRIPT_DIR / "official_mobilefacenet" / "student_distill"


def make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def conv_bn_relu(x, filters, kernel=3, stride=1, name="conv"):
    x = tf.keras.layers.Conv2D(filters, kernel, stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu")(x)


def dw_bn_relu(x, kernel=3, stride=1, name="dw"):
    x = tf.keras.layers.DepthwiseConv2D(kernel, stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu")(x)


def bottleneck(x, out_ch, stride, expansion, name):
    in_ch = int(x.shape[-1])
    hidden = make_divisible(in_ch * expansion)
    y = conv_bn_relu(x, hidden, kernel=1, stride=1, name=f"{name}_expand")
    y = dw_bn_relu(y, kernel=3, stride=stride, name=f"{name}_depthwise")
    y = tf.keras.layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name=f"{name}_project_conv")(y)
    y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_project_bn")(y)
    if stride == 1 and in_ch == out_ch:
        y = tf.keras.layers.Add(name=f"{name}_add")([x, y])
    return y


def build_student(width=0.75, embedding_dim=128):
    def c(ch):
        return make_divisible(ch * width)

    inp = tf.keras.Input(shape=(112, 112, 3), name="input")
    x = conv_bn_relu(inp, c(64), kernel=3, stride=2, name="stem")
    x = dw_bn_relu(x, kernel=3, stride=1, name="stem_dw")
    x = tf.keras.layers.Conv2D(c(64), 1, padding="same", use_bias=False, name="stem_proj_conv")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="stem_proj_bn")(x)

    block_id = 0
    for out_ch, stride, repeats, expansion in [
        (64, 2, 5, 2),
        (128, 2, 1, 4),
        (128, 1, 6, 2),
        (128, 2, 1, 4),
        (128, 1, 2, 2),
    ]:
        for i in range(repeats):
            x = bottleneck(
                x,
                c(out_ch),
                stride if i == 0 else 1,
                expansion,
                name=f"ir_{block_id}",
            )
            block_id += 1

    x = conv_bn_relu(x, c(512), kernel=1, stride=1, name="head_conv")
    x = tf.keras.layers.DepthwiseConv2D(7, padding="valid", use_bias=False, name="head_dw")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="head_dw_bn")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(embedding_dim, use_bias=False, name="embedding")(x)
    return tf.keras.Model(inp, x, name=f"mfn_student_w{width:g}_{embedding_dim}d")


def image_paths(limit):
    paths = sorted(p for p in CALIB_DIR.glob("*.jpg") if not p.name.startswith("._"))
    return paths[:limit]


def load_images(paths):
    images = []
    kept = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (112, 112):
                img = img.resize((112, 112), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.uint8))
            kept.append(str(path))
        except Exception:
            pass
    return np.stack(images), np.asarray(kept)


def load_teacher():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=str(TEACHER_MODEL))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    out_q = out.get("quantization_parameters", {})
    out_scale = float(np.asarray(out_q.get("scales", [1.0])).flat[0])
    out_zp = int(np.asarray(out_q.get("zero_points", [0])).flat[0])
    return interp, inp["index"], out["index"], out_scale, out_zp


def teacher_input_from_uint8(images):
    a = images.astype(np.int32)
    return np.where(a > 128, a - 128, a - 129).clip(-128, 127).astype(np.int8)


def l2_np(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def compute_teacher_embeddings(images, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = np.load(cache_path)
        if len(data["teacher512"]) == len(images):
            print(f"Loaded teacher cache: {cache_path}")
            return data["teacher512"].astype(np.float32)

    interp, in_idx, out_idx, out_scale, out_zp = load_teacher()
    teacher = []
    inputs = teacher_input_from_uint8(images)
    for i, img in enumerate(inputs):
        interp.set_tensor(in_idx, img[np.newaxis, ...])
        interp.invoke()
        emb = interp.get_tensor(out_idx).reshape(-1).astype(np.float32)
        emb = (emb - out_zp) * out_scale
        teacher.append(l2_np(emb[np.newaxis, :])[0])
        if (i + 1) % 1000 == 0:
            print(f"  teacher {i + 1}/{len(inputs)}")
    teacher = np.stack(teacher).astype(np.float32)
    np.savez(cache_path, teacher512=teacher)
    print(f"Saved teacher cache: {cache_path}")
    return teacher


def projected_targets(teacher512, projection_path):
    data = np.load(projection_path)
    mean = data["mean"].astype(np.float32)
    projection = data["projection"].astype(np.float32)
    return l2_np((teacher512 - mean) @ projection).astype(np.float32)


def train(
    model,
    images,
    teacher512,
    target128,
    epochs,
    batch_size,
    lr,
    checkpoint_path=None,
    checkpoint_every=5,
    target_weight=1.0,
    pair_weight=4.0,
    negative_weight=4.0,
    negative_margin=0.12,
    negative_teacher_threshold=0.30,
    augment=False,
):
    x = (images.astype(np.float32) / 127.5) - 1.0
    ds = tf.data.Dataset.from_tensor_slices((x, teacher512, target128))
    ds = ds.shuffle(len(images), reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    def augment_batch(batch_x):
        if not augment:
            return batch_x
        batch_x = tf.image.random_flip_left_right(batch_x)
        batch_x = tf.image.random_brightness(batch_x, max_delta=0.08)
        batch_x = tf.image.random_contrast(batch_x, lower=0.85, upper=1.15)
        return tf.clip_by_value(batch_x, -1.0, 1.0)

    @tf.function
    def step(batch_x, batch_teacher, batch_target):
        with tf.GradientTape() as tape:
            batch_x = augment_batch(batch_x)
            pred = model(batch_x, training=True)
            pred_n = tf.math.l2_normalize(pred, axis=-1)
            target_n = tf.math.l2_normalize(batch_target, axis=-1)

            target_loss = tf.reduce_mean(1.0 - tf.reduce_sum(pred_n * target_n, axis=-1))
            sim_teacher = tf.linalg.matmul(batch_teacher, batch_teacher, transpose_b=True)
            sim_student = tf.linalg.matmul(pred_n, pred_n, transpose_b=True)

            batch_n = tf.shape(batch_x)[0]
            offdiag = 1.0 - tf.eye(batch_n, dtype=tf.float32)
            offdiag_count = tf.reduce_sum(offdiag) + 1e-6
            pair_loss = tf.reduce_sum(tf.square(sim_student - sim_teacher) * offdiag) / offdiag_count

            negative_mask = tf.cast(sim_teacher < negative_teacher_threshold, tf.float32) * offdiag
            negative_count = tf.reduce_sum(negative_mask) + 1e-6
            negative_loss = (
                tf.reduce_sum(tf.square(tf.nn.relu(sim_student - negative_margin)) * negative_mask)
                / negative_count
            )

            value_loss = 0.0001 * tf.reduce_mean(tf.square(pred))
            loss = (
                target_weight * target_loss
                + pair_weight * pair_loss
                + negative_weight * negative_loss
                + value_loss
            )
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, target_loss, pair_loss, negative_loss

    for epoch in range(1, epochs + 1):
        losses = []
        target_losses = []
        pair_losses = []
        negative_losses = []
        for batch_x, batch_teacher, batch_target in ds:
            loss, target_loss, pair_loss, negative_loss = step(batch_x, batch_teacher, batch_target)
            losses.append(float(loss))
            target_losses.append(float(target_loss))
            pair_losses.append(float(pair_loss))
            negative_losses.append(float(negative_loss))
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"loss={np.mean(losses):.5f} "
            f"target={np.mean(target_losses):.5f} "
            f"pair={np.mean(pair_losses):.5f} "
            f"neg={np.mean(negative_losses):.5f}",
            flush=True,
        )
        if checkpoint_path and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            model.save_weights(checkpoint_path)
            print(f"checkpoint: {checkpoint_path}", flush=True)


def export_tflite(model, images, out_prefix, num_calib):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    float_path = Path(f"{out_prefix}.float32.tflite")
    int8_path = Path(f"{out_prefix}.int8.tflite")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = converter.convert()
    float_path.write_bytes(float_model)
    print(f"Saved {float_path} ({float_path.stat().st_size / 1024:.1f} KiB)")

    rep_images = ((images[:num_calib].astype(np.float32) / 127.5) - 1.0).astype(np.float32)

    def representative_dataset():
        for img in rep_images:
            yield [img[np.newaxis, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_model = converter.convert()
    int8_path.write_bytes(int8_model)
    print(f"Saved {int8_path} ({int8_path.stat().st_size / 1024:.1f} KiB)")
    return float_path, int8_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=0.75)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-train", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--projection", type=Path, default=PROJECTION_NPZ)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--init-weights", type=Path, help="Optional Keras weights to continue training from")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=4.0)
    parser.add_argument("--negative-weight", type=float, default=4.0)
    parser.add_argument("--negative-margin", type=float, default=0.12)
    parser.add_argument("--negative-teacher-threshold", type=float, default=0.30)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    start = time.time()
    name = f"mfn_w{args.width:g}_distill_{args.embedding_dim}d"
    out_prefix = args.out_dir / name
    paths = image_paths(args.num_train)
    images, kept = load_images(paths)
    print(f"Loaded {len(images)} images for {name}")

    teacher_cache = args.out_dir / f"{name}_teacher512_{len(images)}.npz"
    teacher512 = compute_teacher_embeddings(images, teacher_cache)
    target128 = projected_targets(teacher512, args.projection)

    model = build_student(width=args.width, embedding_dim=args.embedding_dim)
    print(f"Model params: {model.count_params():,}")
    weights_path = args.out_dir / f"{name}.weights.h5"
    if args.init_weights and args.init_weights.exists():
        model.load_weights(args.init_weights)
        print(f"Loaded initial weights: {args.init_weights}")
    train(
        model,
        images,
        teacher512,
        target128,
        args.epochs,
        args.batch_size,
        args.lr,
        checkpoint_path=weights_path,
        checkpoint_every=args.checkpoint_every,
        target_weight=args.target_weight,
        pair_weight=args.pair_weight,
        negative_weight=args.negative_weight,
        negative_margin=args.negative_margin,
        negative_teacher_threshold=args.negative_teacher_threshold,
        augment=args.augment,
    )
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")
    export_tflite(model, images, out_prefix, args.num_calib)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
