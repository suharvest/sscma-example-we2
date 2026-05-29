#!/usr/bin/env python3
"""Prune the official w600k tail channels and fold the output to 128D.

This keeps the official backbone up to Conv_93 intact, keeps a selected subset
of the final 64 tail channels, and folds Gemm+BN+512D->128D projection into a
new Gemm head. It is a small structural experiment to see whether trimming the
last 7x7 tail feature map lowers Vela SRAM while preserving as much of the
official embedding behavior as possible.
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


def load_initializers(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def last_node(nodes, op_type):
    found = [node for node in nodes if node.op_type == op_type]
    if not found:
        raise RuntimeError(f"No {op_type} node found")
    return found[-1]


def channel_columns(channels):
    cols = []
    for channel in channels:
        start = int(channel) * 49
        cols.extend(range(start, start + 49))
    return np.asarray(cols, dtype=np.int64)


def select_tail_channels(folded_dense128_weight, channels):
    per_channel = folded_dense128_weight.reshape(128, 64, 49)
    scores = np.linalg.norm(per_channel, axis=(0, 2))
    selected = np.argsort(scores)[-channels:]
    return np.sort(selected)


def rewrite_onnx(input_onnx, projection_path, output_onnx, tail_channels):
    if not 1 <= tail_channels <= 64:
        raise ValueError("--tail-channels must be in [1, 64]")

    model = onnx.load(str(input_onnx))
    original = onnx.load(str(input_onnx))
    init = load_initializers(model)
    nodes = list(model.graph.node)
    gemm = last_node(nodes, "Gemm")
    bn = last_node(nodes, "BatchNormalization")

    conv93 = next((node for node in nodes if node.name == "Conv_93"), None)
    prelu94 = next((node for node in nodes if node.name == "PRelu_94"), None)
    if conv93 is None or prelu94 is None:
        raise RuntimeError("Expected Conv_93 and PRelu_94 in official w600k ONNX")
    if bn.input[0] != gemm.output[0]:
        raise RuntimeError("Expected final BN to consume final Gemm")

    fc_weight = init[gemm.input[1]].astype(np.float32)
    fc_bias = init[gemm.input[2]].astype(np.float32)
    if int(attr_value(gemm, "transB", 0)) != 1:
        raise RuntimeError("Expected final Gemm transB=1")

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
    folded_dense128_weight = (projection.T @ folded_weight512).astype(np.float32)
    dense128_bias = ((folded_bias512 - proj_mean) @ projection).astype(np.float32)

    selected_channels = select_tail_channels(folded_dense128_weight, tail_channels)
    selected_cols = channel_columns(selected_channels)
    dense128_weight = folded_dense128_weight[:, selected_cols].astype(np.float32)
    print(f"Selected tail channels ({tail_channels}/64): {selected_channels.tolist()}")

    conv_weight = init[conv93.input[1]].astype(np.float32)[selected_channels]
    conv_bias = init[conv93.input[2]].astype(np.float32)[selected_channels]
    prelu_slope = init[prelu94.input[1]].astype(np.float32)[selected_channels]

    replacements = {
        conv93.input[1]: numpy_helper.from_array(conv_weight, conv93.input[1]),
        conv93.input[2]: numpy_helper.from_array(conv_bias, conv93.input[2]),
        prelu94.input[1]: numpy_helper.from_array(prelu_slope, prelu94.input[1]),
    }

    keep_nodes = nodes[: nodes.index(gemm)]
    dense128_weight_name = "fc_tail_dense128.weight"
    dense128_bias_name = "fc_tail_dense128.bias"
    dense128_output_name = f"embedding128_tail{tail_channels}"
    keep_nodes.append(
        helper.make_node(
            "Gemm",
            inputs=[gemm.input[0], dense128_weight_name, dense128_bias_name],
            outputs=[dense128_output_name],
            name=f"Gemm_tail{tail_channels}_dense128",
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
    parser.add_argument("--tail-channels", type=int, default=48)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--skip-vela", action="store_true")
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    name = f"w600k_tail{args.tail_channels}_dense128"
    out_dir = args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = rewrite_onnx(args.onnx, args.projection, out_dir / f"{name}.onnx", args.tail_channels)
    tf_dir = convert_onnx_to_tflite(onnx_path, out_dir, name)
    int8_path = export_int8(tf_dir, load_calibration_images(args.num_calib), out_dir, name)
    if not args.skip_vela:
        run_vela(int8_path, out_dir)


if __name__ == "__main__":
    main()
