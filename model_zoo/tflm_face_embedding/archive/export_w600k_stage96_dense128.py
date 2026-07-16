#!/usr/bin/env python3
"""Prune the official w600k early 128-channel stage to 96 channels.

This is a small-inheritance experiment. It keeps the official graph topology,
but slices the shared 128-channel stage (nodes Conv_0 through Conv_31) to 96
channels, adjusts Conv_33 input channels, and folds the official output to a
128D Dense head. The selected channels are chosen as 48 channel-pairs because
Conv_2 uses group=64 with two channels per group.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import tensorflow as tf
from onnx import TensorProto, helper, numpy_helper
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = SCRIPT_DIR / "official_mobilefacenet" / "w600k_mbf_fixed.onnx"
DEFAULT_PROJECTION = SCRIPT_DIR / "outputs" / "w600k_projection_128d.npz"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "official_mobilefacenet"
CALIB_DIR = SCRIPT_DIR / "calibration_data" / "qat_112"


def attr_value(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def set_attr(node, name, value):
    del node.attribute[:]
    for key, old_value in value.items():
        node.attribute.append(helper.make_attribute(key, old_value))


def load_initializers(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def node_by_name(nodes, name):
    for node in nodes:
        if node.name == name:
            return node
    raise RuntimeError(f"Node not found: {name}")


def last_node(nodes, op_type):
    found = [node for node in nodes if node.op_type == op_type]
    if not found:
        raise RuntimeError(f"No {op_type} node found")
    return found[-1]


def select_stage_channels(init, keep_channels):
    if keep_channels % 2 != 0:
        raise ValueError("keep_channels must be even because Conv_2 uses channel pairs")
    conv33_weight = init["569"].astype(np.float32)  # Conv_33 [256, 128, 1, 1]
    scores = np.linalg.norm(conv33_weight.reshape(256, 64, 2), axis=(0, 2))
    selected_groups = np.argsort(scores)[-(keep_channels // 2):]
    selected_groups = np.sort(selected_groups)
    channels = []
    for group in selected_groups:
        channels.extend([int(group) * 2, int(group) * 2 + 1])
    return np.asarray(channels, dtype=np.int64)


def slice_conv_weight(weight, out_channels=None, in_channels=None):
    result = weight
    if out_channels is not None:
        result = result[out_channels]
    if in_channels is not None and result.shape[1] != 1:
        result = result[:, in_channels]
    return result


def slice_vector(vector, channels):
    return vector[channels]


def fold_dense128(init, nodes, projection_path):
    gemm = last_node(nodes, "Gemm")
    bn = last_node(nodes, "BatchNormalization")
    fc_weight = init[gemm.input[1]].astype(np.float32)
    fc_bias = init[gemm.input[2]].astype(np.float32)
    bn_scale = init[bn.input[1]].astype(np.float32)
    bn_bias = init[bn.input[2]].astype(np.float32)
    bn_mean = init[bn.input[3]].astype(np.float32)
    bn_var = init[bn.input[4]].astype(np.float32)
    bn_eps = float(attr_value(bn, "epsilon", 1e-5))
    projection_data = np.load(projection_path)
    proj_mean = projection_data["mean"].astype(np.float32)
    projection = projection_data["projection"].astype(np.float32)
    inv_std = bn_scale / np.sqrt(bn_var + bn_eps)
    folded_weight512 = fc_weight * inv_std[:, None]
    folded_bias512 = (fc_bias - bn_mean) * inv_std + bn_bias
    dense128_weight = (projection.T @ folded_weight512).astype(np.float32)
    dense128_bias = ((folded_bias512 - proj_mean) @ projection).astype(np.float32)
    return gemm, dense128_weight, dense128_bias


def rewrite_onnx(input_onnx, projection_path, output_onnx, keep_channels):
    model = onnx.load(str(input_onnx))
    original = onnx.load(str(input_onnx))
    init = load_initializers(model)
    nodes = list(model.graph.node)
    channels = select_stage_channels(init, keep_channels)
    print(f"Selected stage channels ({keep_channels}/128): {channels.tolist()}")

    replacements = {}

    # Nodes Conv_0..Conv_31 operate in the shared 128-channel stage. Keep a
    # common channel subset so residual Add tensors remain shape-compatible.
    stage_conv_names = [
        "Conv_0", "Conv_2", "Conv_4", "Conv_6", "Conv_8", "Conv_9",
        "Conv_11", "Conv_13", "Conv_15", "Conv_17", "Conv_19", "Conv_21",
        "Conv_23", "Conv_25", "Conv_27", "Conv_29", "Conv_31",
    ]
    stage_prelu_names = [
        "PRelu_1", "PRelu_3", "PRelu_5", "PRelu_7", "PRelu_10", "PRelu_12",
        "PRelu_16", "PRelu_18", "PRelu_22", "PRelu_24", "PRelu_28", "PRelu_30",
    ]

    for name in stage_conv_names:
        node = node_by_name(nodes, name)
        weight_name, bias_name = node.input[1], node.input[2]
        weight = init[weight_name].astype(np.float32)
        bias = init[bias_name].astype(np.float32)

        in_channels = None
        out_channels = channels
        if name != "Conv_0":
            in_channels = channels
        if name == "Conv_2":
            # Conv_2 is grouped in 2-channel pairs. With selected paired
            # channels, each new group still consumes 2 adjacent input channels.
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            attrs["group"] = keep_channels // 2
            set_attr(node, "group", attrs)
            in_channels = None
        elif int(attr_value(node, "group", 1)) == 128:
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            attrs["group"] = keep_channels
            set_attr(node, "group", attrs)
            in_channels = None

        replacements[weight_name] = numpy_helper.from_array(
            slice_conv_weight(weight, out_channels=out_channels, in_channels=in_channels),
            weight_name,
        )
        replacements[bias_name] = numpy_helper.from_array(slice_vector(bias, channels), bias_name)

    for name in stage_prelu_names:
        node = node_by_name(nodes, name)
        slope_name = node.input[1]
        slope = init[slope_name].astype(np.float32)
        replacements[slope_name] = numpy_helper.from_array(slice_vector(slope, channels), slope_name)

    # Conv_33 transitions from the pruned 128-channel stage into 256 channels.
    conv33 = node_by_name(nodes, "Conv_33")
    conv33_weight_name = conv33.input[1]
    conv33_weight = init[conv33_weight_name].astype(np.float32)
    replacements[conv33_weight_name] = numpy_helper.from_array(conv33_weight[:, channels], conv33_weight_name)

    gemm, dense128_weight, dense128_bias = fold_dense128(init, nodes, projection_path)
    dense128_weight_name = "fc_stage_dense128.weight"
    dense128_bias_name = "fc_stage_dense128.bias"
    dense128_output_name = f"embedding128_stage{keep_channels}"

    keep_nodes = nodes[: nodes.index(gemm)]
    keep_nodes.append(
        helper.make_node(
            "Gemm",
            inputs=[gemm.input[0], dense128_weight_name, dense128_bias_name],
            outputs=[dense128_output_name],
            name=f"Gemm_stage{keep_channels}_dense128",
            transB=1,
        )
    )

    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)

    kept_initializer_names = {inp for node in keep_nodes[:-1] for inp in node.input}
    original_initializers = {initializer.name: initializer for initializer in original.graph.initializer}
    del model.graph.initializer[:]
    for name, initializer in original_initializers.items():
        if name in kept_initializer_names:
            model.graph.initializer.append(replacements.get(name, initializer))
    model.graph.initializer.append(numpy_helper.from_array(dense128_weight, dense128_weight_name))
    model.graph.initializer.append(numpy_helper.from_array(dense128_bias, dense128_bias_name))

    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info(dense128_output_name, TensorProto.FLOAT, [1, 128]))

    onnx.checker.check_model(model)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_onnx))
    print(f"Saved {output_onnx}")
    return output_onnx


def convert_onnx_to_tflite(input_onnx, out_dir, name):
    import onnx2tf

    tf_dir = out_dir / "saved_model"
    if tf_dir.exists():
        shutil.rmtree(tf_dir)
    onnx2tf.convert(
        input_onnx_file_path=str(input_onnx),
        output_folder_path=str(tf_dir),
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
    )
    float_candidates = sorted(tf_dir.glob("*_float32.tflite"))
    if not float_candidates:
        raise RuntimeError(f"No float32 tflite produced in {tf_dir}")
    float_path = out_dir / f"{name}.float32.tflite"
    shutil.copy(float_candidates[0], float_path)
    print(f"Saved {float_path} ({float_path.stat().st_size / 1024:.1f} KiB)")
    return tf_dir


def load_calibration_images(limit):
    images = []
    for path in sorted(CALIB_DIR.glob("*.jpg"))[:limit]:
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (112, 112):
                img = img.resize((112, 112), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            images.append((arr / 127.5) - 1.0)
        except Exception:
            pass
    if not images:
        raise RuntimeError(f"No calibration images found in {CALIB_DIR}")
    return np.asarray(images, dtype=np.float32)


def export_int8(tf_dir, calib_images, out_dir, name):
    def representative_dataset():
        for img in calib_images:
            yield [img[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model = converter.convert()
    int8_path = out_dir / f"{name}.int8.tflite"
    int8_path.write_bytes(model)
    print(f"Saved {int8_path} ({int8_path.stat().st_size / 1024:.1f} KiB)")
    return int8_path


def run_vela(tflite_path, out_dir):
    vela = shutil.which("vela")
    if not vela:
        print("Vela not found; skipping")
        return
    result = subprocess.run(
        [
            vela,
            str(tflite_path),
            "--accelerator-config",
            "ethos-u55-64",
            "--optimise",
            "Performance",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError(f"Vela failed: {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--projection", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--keep-channels", type=int, default=96)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--skip-vela", action="store_true")
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    name = f"w600k_stage{args.keep_channels}_dense128"
    out_dir = args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = rewrite_onnx(args.onnx, args.projection, out_dir / f"{name}.onnx", args.keep_channels)
    tf_dir = convert_onnx_to_tflite(onnx_path, out_dir, name)
    int8_path = export_int8(tf_dir, load_calibration_images(args.num_calib), out_dir, name)
    if not args.skip_vela:
        run_vela(int8_path, out_dir)


if __name__ == "__main__":
    main()
