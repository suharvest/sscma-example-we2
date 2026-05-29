#!/usr/bin/env python3
"""Fold official w600k final Dense+BN+projection into a 3136 -> 128 head.

The official ONNX tail is:

    Flatten(3136) -> Gemm(3136->512) -> BatchNorm(512) -> embedding512

This script folds the trained outputs/w600k_projection_128d.npz projection into
that tail and rewrites it as:

    Flatten(3136) -> Gemm(3136->128)

The backbone is otherwise unchanged, so the float output should match the
previous "official 512D output + 512->128 projection" experiment closely while
removing the 512D output head from the graph.
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
DEFAULT_OUT_DIR = SCRIPT_DIR / "official_mobilefacenet" / "w600k_dense128"
CALIB_DIR = SCRIPT_DIR / "calibration_data" / "qat_112"


def attr_value(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def find_node_by_op(nodes, op_type):
    found = [node for node in nodes if node.op_type == op_type]
    if not found:
        raise RuntimeError(f"No {op_type} node found")
    return found[-1]


def load_initializers(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def rewrite_onnx(input_onnx, projection_path, output_onnx):
    model = onnx.load(str(input_onnx))
    init = load_initializers(model)
    nodes = list(model.graph.node)
    gemm = find_node_by_op(nodes, "Gemm")
    bn = find_node_by_op(nodes, "BatchNormalization")

    if bn.input[0] != gemm.output[0]:
        raise RuntimeError("Expected final BatchNormalization to consume final Gemm output")

    fc_weight = init[gemm.input[1]].astype(np.float32)
    fc_bias = init[gemm.input[2]].astype(np.float32)
    trans_b = int(attr_value(gemm, "transB", 0))
    if trans_b != 1:
        raise RuntimeError(f"Expected Gemm transB=1 for fc.weight [512,3136], got {trans_b}")

    bn_scale = init[bn.input[1]].astype(np.float32)
    bn_bias = init[bn.input[2]].astype(np.float32)
    bn_mean = init[bn.input[3]].astype(np.float32)
    bn_var = init[bn.input[4]].astype(np.float32)
    bn_eps = float(attr_value(bn, "epsilon", 1e-5))

    projection_data = np.load(projection_path)
    proj_mean = projection_data["mean"].astype(np.float32)
    projection = projection_data["projection"].astype(np.float32)
    if proj_mean.shape != (512,) or projection.shape != (512, 128):
        raise RuntimeError(f"Unexpected projection shapes: mean={proj_mean.shape} projection={projection.shape}")

    inv_std = bn_scale / np.sqrt(bn_var + bn_eps)
    folded_weight512 = fc_weight * inv_std[:, None]
    folded_bias512 = (fc_bias - bn_mean) * inv_std + bn_bias

    dense128_weight = (projection.T @ folded_weight512).astype(np.float32)
    dense128_bias = ((folded_bias512 - proj_mean) @ projection).astype(np.float32)

    dense128_weight_name = "fc_dense128.weight"
    dense128_bias_name = "fc_dense128.bias"
    dense128_output_name = "embedding128"

    keep_nodes = nodes[: nodes.index(gemm)]
    new_gemm = helper.make_node(
        "Gemm",
        inputs=[gemm.input[0], dense128_weight_name, dense128_bias_name],
        outputs=[dense128_output_name],
        name="Gemm_dense128",
        transB=1,
    )
    keep_nodes.append(new_gemm)

    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)

    del model.graph.initializer[:]
    kept_initializer_names = {inp for node in keep_nodes[:-1] for inp in node.input}
    original_initializers = {initializer.name: initializer for initializer in onnx.load(str(input_onnx)).graph.initializer}
    for name, initializer in original_initializers.items():
        if name in kept_initializer_names:
            model.graph.initializer.append(initializer)
    model.graph.initializer.append(numpy_helper.from_array(dense128_weight, dense128_weight_name))
    model.graph.initializer.append(numpy_helper.from_array(dense128_bias, dense128_bias_name))

    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(dense128_output_name, TensorProto.FLOAT, [1, 128])
    )

    onnx.checker.check_model(model)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_onnx))
    print(f"Saved {output_onnx}")
    return output_onnx


def convert_onnx_to_tflite(input_onnx, out_dir):
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
    float_path = out_dir / "w600k_dense128.float32.tflite"
    shutil.copy(float_candidates[0], float_path)
    print(f"Saved {float_path} ({float_path.stat().st_size / 1024:.1f} KiB)")
    return tf_dir, float_path


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


def export_int8(tf_dir, calib_images, out_dir):
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
    int8_path = out_dir / "w600k_dense128.int8.tflite"
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-calib", type=int, default=500)
    parser.add_argument("--skip-vela", action="store_true")
    args = parser.parse_args()

    if Path.cwd().resolve() != SCRIPT_DIR:
        os.chdir(SCRIPT_DIR)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dense_onnx = args.out_dir / "w600k_dense128.onnx"
    rewrite_onnx(args.onnx, args.projection, dense_onnx)
    tf_dir, _ = convert_onnx_to_tflite(dense_onnx, args.out_dir)
    int8_path = export_int8(tf_dir, load_calibration_images(args.num_calib), args.out_dir)
    if not args.skip_vela:
        run_vela(int8_path, args.out_dir)


if __name__ == "__main__":
    main()
