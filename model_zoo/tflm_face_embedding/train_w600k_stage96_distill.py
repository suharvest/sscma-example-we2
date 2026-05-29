#!/usr/bin/env python3
"""Fine-tune a lightly pruned official w600k stage96 model by distillation."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnx
import tensorflow as tf
from onnx import helper, numpy_helper
from PIL import Image

from export_w600k_stage96_dense128 import (
    DEFAULT_ONNX,
    DEFAULT_PROJECTION,
    attr_value,
    fold_dense128,
    node_by_name,
    select_stage_channels,
)
from train_mfn_student_distill import export_tflite, l2_np, teacher_input_from_uint8


SCRIPT_DIR = Path(__file__).resolve().parent
TEACHER_MODEL = SCRIPT_DIR / "official_mobilefacenet" / "w600k_dense128" / "w600k_dense128.int8.tflite"
OUT_DIR = SCRIPT_DIR / "official_mobilefacenet" / "w600k_stage96_distill"
CALIB_DIR = SCRIPT_DIR / "calibration_data" / "qat_112"


def load_initializers(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def selected_stage_channels(init, keep_channels):
    return select_stage_channels(init, keep_channels)


def slice_conv(weight, out_channels=None, in_channels=None):
    result = weight
    if out_channels is not None:
        result = result[out_channels]
    if in_channels is not None and result.shape[1] != 1:
        result = result[:, in_channels]
    return result


def prepare_initializers(onnx_path, projection_path, keep_channels):
    model = onnx.load(str(onnx_path))
    nodes = list(model.graph.node)
    init = load_initializers(model)
    channels = selected_stage_channels(init, keep_channels)
    replacements = {}

    stage_conv_names = {
        "Conv_0", "Conv_2", "Conv_4", "Conv_6", "Conv_8", "Conv_9",
        "Conv_11", "Conv_13", "Conv_15", "Conv_17", "Conv_19", "Conv_21",
        "Conv_23", "Conv_25", "Conv_27", "Conv_29", "Conv_31",
    }
    stage_prelu_names = {
        "PRelu_1", "PRelu_3", "PRelu_5", "PRelu_7", "PRelu_10", "PRelu_12",
        "PRelu_16", "PRelu_18", "PRelu_22", "PRelu_24", "PRelu_28", "PRelu_30",
    }

    for name in stage_conv_names:
        node = node_by_name(nodes, name)
        weight = init[node.input[1]].astype(np.float32)
        bias = init[node.input[2]].astype(np.float32)
        in_channels = None if name in {"Conv_0", "Conv_2"} else channels
        if int(attr_value(node, "group", 1)) == 128:
            in_channels = None
        replacements[node.input[1]] = slice_conv(weight, out_channels=channels, in_channels=in_channels)
        replacements[node.input[2]] = bias[channels]

    for name in stage_prelu_names:
        node = node_by_name(nodes, name)
        replacements[node.input[1]] = init[node.input[1]].astype(np.float32)[channels]

    conv33 = node_by_name(nodes, "Conv_33")
    replacements[conv33.input[1]] = init[conv33.input[1]].astype(np.float32)[:, channels]

    gemm, dense128_weight, dense128_bias = fold_dense128(init, nodes, projection_path)
    return model, nodes[: nodes.index(gemm)], init, replacements, dense128_weight, dense128_bias, channels


def keras_conv_from_onnx(x, node, init, replacements, tensor_channels):
    weight = replacements.get(node.input[1], init[node.input[1]].astype(np.float32))
    bias = replacements.get(node.input[2], init[node.input[2]].astype(np.float32))
    group = int(attr_value(node, "group", 1))
    strides = tuple(int(v) for v in attr_value(node, "strides", [1, 1]))
    pads = attr_value(node, "pads", [0, 0, 0, 0])
    padding = "same" if any(int(v) for v in pads) else "valid"

    out_ch = int(weight.shape[0])
    in_ch = int(tensor_channels[node.input[0]])
    if node.name == "Conv_2" and out_ch != 128:
        group = out_ch // 2
    elif group == 128 and out_ch != 128:
        group = out_ch

    if group == in_ch and weight.shape[1] == 1 and out_ch == in_ch:
        layer = tf.keras.layers.DepthwiseConv2D(
            tuple(int(v) for v in attr_value(node, "kernel_shape", [3, 3])),
            strides=strides,
            padding=padding,
            use_bias=True,
            name=node.name,
        )
        y = layer(x)
        kernel = np.transpose(weight[:, 0, :, :], (1, 2, 0))[:, :, :, None]
        layer.set_weights([kernel.astype(np.float32), bias.astype(np.float32)])
    else:
        layer = tf.keras.layers.Conv2D(
            out_ch,
            tuple(int(v) for v in attr_value(node, "kernel_shape", [1, 1])),
            strides=strides,
            padding=padding,
            groups=group,
            use_bias=True,
            name=node.name,
        )
        y = layer(x)
        kernel = np.transpose(weight, (2, 3, 1, 0))
        layer.set_weights([kernel.astype(np.float32), bias.astype(np.float32)])

    tensor_channels[node.output[0]] = out_ch
    return y


def activation_from_prelu(node_name, alpha, activation_override):
    if activation_override == "none":
        return None

    target, mode = activation_override.split("-", 1)
    override_targets = {
        "all": None,
        "suspect": {"PRelu_28", "PRelu_30"},
        "prelu28": {"PRelu_28"},
        "prelu30": {"PRelu_30"},
    }
    if target not in override_targets:
        raise ValueError(f"Unsupported activation override target: {target}")
    selected = override_targets[target]
    if selected is not None and node_name not in selected:
        return None

    if mode == "leaky":
        slope = float(np.mean(alpha))
        return tf.keras.layers.LeakyReLU(alpha=slope, name=f"{node_name}_leaky")
    if mode == "relu6":
        return tf.keras.layers.ReLU(max_value=6.0, name=f"{node_name}_relu6")
    raise ValueError(f"Unsupported activation override: {activation_override}")


def build_model(
    onnx_path=DEFAULT_ONNX,
    projection_path=DEFAULT_PROJECTION,
    keep_channels=96,
    activation_override="none",
):
    _, nodes, init, replacements, dense128_weight, dense128_bias, _ = prepare_initializers(
        onnx_path, projection_path, keep_channels
    )

    inp = tf.keras.Input(shape=(112, 112, 3), name="input_1")
    tensors = {"input.1": inp}
    tensor_channels = {"input.1": 3}
    for node in nodes:
        if node.op_type == "Conv":
            tensors[node.output[0]] = keras_conv_from_onnx(tensors[node.input[0]], node, init, replacements, tensor_channels)
        elif node.op_type == "PRelu":
            alpha = replacements.get(node.input[1], init[node.input[1]].astype(np.float32))
            layer = activation_from_prelu(node.name, alpha, activation_override)
            if layer is None:
                layer = tf.keras.layers.PReLU(shared_axes=[1, 2], name=node.name)
                y = layer(tensors[node.input[0]])
                layer.set_weights([alpha.reshape(1, 1, -1).astype(np.float32)])
            else:
                y = layer(tensors[node.input[0]])
            tensors[node.output[0]] = y
            tensor_channels[node.output[0]] = tensor_channels[node.input[0]]
        elif node.op_type == "Add":
            tensors[node.output[0]] = tf.keras.layers.Add(name=node.name)([tensors[node.input[0]], tensors[node.input[1]]])
            tensor_channels[node.output[0]] = tensor_channels[node.input[0]]
        elif node.op_type == "Flatten":
            x = tf.keras.layers.Permute((3, 1, 2), name="to_nchw")(tensors[node.input[0]])
            tensors[node.output[0]] = tf.keras.layers.Flatten(name=node.name)(x)
            tensor_channels[node.output[0]] = int(np.prod(tensors[node.output[0]].shape[1:]))
        else:
            raise RuntimeError(f"Unsupported node {node.op_type} {node.name}")

    dense = tf.keras.layers.Dense(128, use_bias=True, name="embedding128")
    out = dense(tensors[nodes[-1].output[0]])
    dense.set_weights([dense128_weight.T.astype(np.float32), dense128_bias.astype(np.float32)])
    return tf.keras.Model(inp, out, name=f"w600k_stage{keep_channels}_distill")


def image_paths(limit, identity_dirs="", max_identity_images=0):
    paths = sorted(p for p in CALIB_DIR.glob("*.jpg") if not p.name.startswith("._"))
    if identity_dirs:
        extra = []
        for value in identity_dirs.split(","):
            root = Path(value.strip())
            if not root:
                continue
            if not root.is_absolute():
                root = SCRIPT_DIR / root
            found = sorted(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if max_identity_images:
                found = found[:max_identity_images]
            extra.extend(found)
        paths.extend(extra)
    return paths[:limit] if limit else paths


def load_images(paths):
    images = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (112, 112):
                img = img.resize((112, 112), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.uint8))
        except Exception:
            pass
    return np.stack(images)


def load_teacher():
    interp = tf.lite.Interpreter(model_path=str(TEACHER_MODEL))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    out_q = out["quantization_parameters"]
    return interp, inp["index"], out["index"], float(out_q["scales"][0]), int(out_q["zero_points"][0])


def teacher_input(images):
    return teacher_input_from_uint8(images)


def compute_tflite_teacher_targets(images, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = np.load(cache_path)
        if len(data["targets"]) == len(images):
            print(f"Loaded teacher cache: {cache_path}")
            return data["targets"].astype(np.float32)
    interp, in_idx, out_idx, out_scale, out_zp = load_teacher()
    targets = []
    inputs = teacher_input(images)
    for i, img in enumerate(inputs):
        interp.set_tensor(in_idx, img[np.newaxis, ...])
        interp.invoke()
        emb = interp.get_tensor(out_idx).reshape(-1).astype(np.float32)
        targets.append(l2_np(((emb - out_zp) * out_scale)[np.newaxis, :])[0])
        if (i + 1) % 1000 == 0:
            print(f"  teacher {i + 1}/{len(inputs)}", flush=True)
    targets = np.stack(targets).astype(np.float32)
    np.savez(cache_path, targets=targets)
    return targets


def compute_keras_teacher_targets(images, cache_path, batch_size):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        data = np.load(cache_path)
        if len(data["targets"]) == len(images):
            print(f"Loaded teacher cache: {cache_path}")
            return data["targets"].astype(np.float32)
    teacher = build_model(keep_channels=128)
    teacher.trainable = False
    x = ((images.astype(np.float32) / 127.5) - 1.0).astype(np.float32)
    targets = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        emb = teacher(batch, training=False).numpy().astype(np.float32)
        targets.append(l2_np(emb))
        print(f"  teacher {min(start + len(batch), len(x))}/{len(x)}", flush=True)
    targets = np.concatenate(targets, axis=0).astype(np.float32)
    np.savez(cache_path, targets=targets)
    return targets


def compute_teacher_targets(images, cache_path, backend, batch_size):
    if backend == "tflite":
        return compute_tflite_teacher_targets(images, cache_path)
    return compute_keras_teacher_targets(images, cache_path, batch_size)


def train(model, images, targets, epochs, batch_size, lr, pair_weight):
    x = ((images.astype(np.float32) / 127.5) - 1.0).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, targets)).shuffle(len(x)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step(batch_x, batch_t):
        with tf.GradientTape() as tape:
            pred = tf.math.l2_normalize(model(batch_x, training=True), axis=-1)
            target = tf.math.l2_normalize(batch_t, axis=-1)
            target_loss = tf.reduce_mean(1.0 - tf.reduce_sum(pred * target, axis=-1))
            sim_p = tf.matmul(pred, pred, transpose_b=True)
            sim_t = tf.matmul(target, target, transpose_b=True)
            pair_loss = tf.reduce_mean(tf.square(sim_p - sim_t))
            loss = target_loss + pair_weight * pair_loss
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 3.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, target_loss, pair_loss

    for epoch in range(1, epochs + 1):
        losses, target_losses, pair_losses = [], [], []
        for batch_x, batch_t in ds:
            loss, target_loss, pair_loss = step(batch_x, batch_t)
            losses.append(float(loss))
            target_losses.append(float(target_loss))
            pair_losses.append(float(pair_loss))
        print(
            f"epoch {epoch:03d}/{epochs} loss={np.mean(losses):.5f} "
            f"target={np.mean(target_losses):.5f} pair={np.mean(pair_losses):.5f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-channels", type=int, default=96)
    parser.add_argument("--num-train", type=int, default=12000)
    parser.add_argument("--identity-dirs", default="")
    parser.add_argument("--max-identity-images", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pair-weight", type=float, default=2.0)
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
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(args.num_train, args.identity_dirs, args.max_identity_images)
    images = load_images(paths)
    print(f"Loaded {len(images)} images")
    targets = compute_teacher_targets(
        images,
        args.out_dir / f"teacher_{args.teacher_backend}_dense128_{len(images)}.npz",
        args.teacher_backend,
        args.batch_size,
    )
    model = build_model(keep_channels=args.keep_channels, activation_override=args.activation_override)
    print(f"Model params: {model.count_params():,}")
    train(model, images, targets, args.epochs, args.batch_size, args.lr, args.pair_weight)
    weights_path = args.out_dir / "w600k_stage96_distill.weights.h5"
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")
    export_tflite(model, images, args.out_dir / "w600k_stage96_distill", args.num_calib)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
