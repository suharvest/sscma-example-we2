#!/usr/bin/env python3
"""
Modify MFN ONNX model: remove split/concat branches, merge to single-path.

This is a mathematically lossless transformation: concat([W1 @ x, W2 @ x]) is
equivalent to concat([W1, W2], axis=0) @ x. The split was designed for ESP32-S3
SIMD optimization; removing it saves SRAM on Ethos-U55 NPU.

Two stages are modified:
  1. dconv_45 (14x14 -> 7x7): merge sep split + proj split
  2. conv_6   (7x7 -> 1x1):  merge sep split (DW + FC unchanged)
"""

import os
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper, checker, shape_inference, version_converter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(MODEL_DIR, "mfn_s8_v1_reconstructed.onnx")
OUTPUT_PATH = os.path.join(MODEL_DIR, "mfn_s8_v1_singlepath.onnx")


def show_graph_diff(old_model, new_model, label: str = ""):
    """Print node count / op type differences between two ONNX models."""
    from collections import Counter

    old_ops = Counter(n.op_type for n in old_model.graph.node)
    new_ops = Counter(n.op_type for n in new_model.graph.node)
    old_count = len(old_model.graph.node)
    new_count = len(new_model.graph.node)

    all_ops = sorted(set(list(old_ops.keys()) + list(new_ops.keys())))
    print(f"\n  --- Graph diff {label} ---")
    print(f"  Node count: {old_count} -> {new_count}  (removed {old_count - new_count})")
    print(f"  Initializers: {len(old_model.graph.initializer)} -> {len(new_model.graph.initializer)}")
    for op in all_ops:
        o = old_ops.get(op, 0)
        n = new_ops.get(op, 0)
        if o != n:
            change = f"(-{o-n})" if o > n else f"(+{n-o})"
            print(f"    {op}: {o} -> {n} {change}")
        else:
            print(f"    {op}: {o} (unchanged)")


def get_init_as_array(model, name: str) -> np.ndarray:
    """Extract a named initializer as a numpy array."""
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"Initializer '{name}' not found")


def remove_initializers(model, names_to_remove: list) -> None:
    """Remove initializers by name from the model graph."""
    keep = [init for init in model.graph.initializer if init.name not in names_to_remove]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def find_node_by_output(model, output_name: str):
    """Find a node that produces the given output name."""
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def replace_input_reference(model, old_name: str, new_name: str):
    """Replace all references to old_name in node inputs with new_name."""
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_name:
                node.input[i] = new_name


def apply_dconv45_fix(model):
    """
    Fix 1: dconv_45 stage (14x14 -> 7x7).

    Removes:
      - Conv_77  (dconv45_sep1)   -> sep_1  128->256
      - Conv_78  (dconv45_sep2)   -> sep_2  128->256
      - PRelu_79 (prelu after sep1)
      - PRelu_80 (prelu after sep2)
      - Concat_81                  -> dconv45_cat1
      - Conv_84  (dconv45_proj1)  -> proj_1 512->64
      - Conv_85  (dconv45_proj2)  -> proj_2 512->64
      - Concat_86                  -> dconv45_cat2

    Adds:
      - Conv dconv45_sep_merged: 128->512 (1x1)
      - PRelu dconv45_prelu_merged: alpha [512]
      - Conv dconv45_proj_merged: 512->128 (1x1)
    """
    # --- Step 1: Extract and merge sep weights ---
    sep1_w = get_init_as_array(model, "dconv_45_conv_sep_1.weight")  # [256, 128, 1, 1]
    sep2_w = get_init_as_array(model, "dconv_45_conv_sep_2.weight")  # [256, 128, 1, 1]
    sep1_b = get_init_as_array(model, "dconv_45_conv_sep_1.bias")    # [256]
    sep2_b = get_init_as_array(model, "dconv_45_conv_sep_2.bias")    # [256]
    sep_merged_w = np.concatenate([sep1_w, sep2_w], axis=0)  # [512, 128, 1, 1]
    sep_merged_b = np.concatenate([sep1_b, sep2_b], axis=0)  # [512]

    # --- Step 2: Extract and merge PReLU alphas ---
    prelu1_a = get_init_as_array(model, "PReLU_alpha_306")   # [1, 256, 1, 1]
    prelu2_a = get_init_as_array(model, "PReLU_alpha_307")   # [1, 256, 1, 1]
    prelu_merged_a = np.concatenate([prelu1_a, prelu2_a], axis=1)  # [1, 512, 1, 1]

    # --- Step 3: Extract and merge proj weights ---
    proj1_w = get_init_as_array(model, "dconv_45_conv_proj_1.weight")  # [64, 512, 1, 1]
    proj2_w = get_init_as_array(model, "dconv_45_conv_proj_2.weight")  # [64, 512, 1, 1]
    proj1_b = get_init_as_array(model, "dconv_45_conv_proj_1.bias")    # [64]
    proj2_b = get_init_as_array(model, "dconv_45_conv_proj_2.bias")    # [64]
    proj_merged_w = np.concatenate([proj1_w, proj2_w], axis=0)  # [128, 512, 1, 1]
    proj_merged_b = np.concatenate([proj1_b, proj2_b], axis=0)  # [128]

    # --- Step 4: Remove old initializers ---
    old_init_names = [
        "dconv_45_conv_sep_1.weight", "dconv_45_conv_sep_1.bias",
        "dconv_45_conv_sep_2.weight", "dconv_45_conv_sep_2.bias",
        "PReLU_alpha_306", "PReLU_alpha_307",
        "dconv_45_conv_proj_1.weight", "dconv_45_conv_proj_1.bias",
        "dconv_45_conv_proj_2.weight", "dconv_45_conv_proj_2.bias",
    ]
    remove_initializers(model, old_init_names)

    # --- Step 5: Add new merged initializers ---
    model.graph.initializer.extend([
        numpy_helper.from_array(sep_merged_w.astype(np.float32),
                                name="dconv_45_conv_sep_merged.weight"),
        numpy_helper.from_array(sep_merged_b.astype(np.float32),
                                name="dconv_45_conv_sep_merged.bias"),
        numpy_helper.from_array(prelu_merged_a.astype(np.float32),
                                name="PReLU_alpha_dconv45_merged"),
        numpy_helper.from_array(proj_merged_w.astype(np.float32),
                                name="dconv_45_conv_proj_merged.weight"),
        numpy_helper.from_array(proj_merged_b.astype(np.float32),
                                name="dconv_45_conv_proj_merged.bias"),
    ])

    # --- Step 6: Build replacement nodes ---

    # Node names of new outputs
    dconv45_sep_merged_out = "dconv45_sep_merged"
    dconv45_prelu_merged_out = "dconv45_prelu_merged"
    dconv45_proj_merged_out = "dconv45_proj_merged"

    # 6a: Merged sep conv (1x1, 128->512)
    new_sep = helper.make_node(
        "Conv",
        inputs=["res_4_block5_add_76", "dconv_45_conv_sep_merged.weight",
                "dconv_45_conv_sep_merged.bias"],
        outputs=[dconv45_sep_merged_out],
        name="Conv_dconv45_sep_merged",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=1,
    )

    # 6b: Merged PReLU (after sep)
    new_prelu = helper.make_node(
        "PRelu",
        inputs=[dconv45_sep_merged_out, "PReLU_alpha_dconv45_merged"],
        outputs=[dconv45_prelu_merged_out],
        name="PRelu_dconv45_sep_merged",
    )

    # 6c: Merged proj conv (1x1, 512->128) -- replaces proj split
    new_proj = helper.make_node(
        "Conv",
        inputs=["prelu_83", "dconv_45_conv_proj_merged.weight",
                "dconv_45_conv_proj_merged.bias"],
        outputs=[dconv45_proj_merged_out],
        name="Conv_dconv45_proj_merged",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=1,
    )

    # --- Step 7: Rebuild node list ---
    # Nodes to remove (by name)
    nodes_to_remove = {
        "Conv_77",   # dconv45_sep1
        "Conv_78",   # dconv45_sep2
        "PRelu_79",  # prelu after sep1
        "PRelu_80",  # prelu after sep2
        "Concat_81", # concat1
        "Conv_84",   # dconv45_proj1
        "Conv_85",   # dconv45_proj2
        "Concat_86", # concat2
    }

    new_nodes = []
    inserted_sep = False
    inserted_proj = False
    for node in model.graph.node:
        if node.name in nodes_to_remove:
            # Insert merged sep nodes BEFORE the first removed node
            if not inserted_sep:
                new_nodes.append(new_sep)
                new_nodes.append(new_prelu)
                inserted_sep = True
            # Skip the old split/concat nodes
            continue
        # Insert merged proj node after PRelu_83 (Node index 82)
        # The proj is inserted before the first downstream consumer (Conv_87)
        if node.name == "Conv_87" and not inserted_proj:
            new_nodes.append(new_proj)
            inserted_proj = True
        new_nodes.append(node)

    # Relink: DW conv (Conv_82) input should point to merged PReLU output instead of dconv45_cat1_81
    for node in new_nodes:
        for i, inp in enumerate(node.input):
            if inp == "dconv45_cat1_81":
                node.input[i] = dconv45_prelu_merged_out
            elif inp == "dconv45_cat2_86":
                node.input[i] = dconv45_proj_merged_out

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    print(f"  [dconv_45] Replaced 8 nodes with 3 merged nodes")
    print(f"  [dconv_45] dconv45_cat1_81 -> {dconv45_prelu_merged_out}")
    print(f"  [dconv_45] dconv45_cat2_86 -> {dconv45_proj_merged_out}")


def apply_conv6_fix(model):
    """
    Fix 2: conv_6 / final stage (7x7 -> 1x1).

    Removes:
      - Conv_99   (s7_sep1)   -> sep_1  128->256
      - Conv_100  (s7_sep2)   -> sep_2  128->256
      - PRelu_101 (prelu after sep1)
      - PRelu_102 (prelu after sep2)
      - Concat_103              -> s7_cat

    Adds:
      - Conv conv_6sep_merged: 128->512 (1x1)
      - PRelu conv_6_prelu_merged: alpha [512]
    """
    # --- Step 1: Extract and merge sep weights ---
    sep1_w = get_init_as_array(model, "conv_6sep_1.weight")   # [256, 128, 1, 1]
    sep2_w = get_init_as_array(model, "conv_6sep_2.weight")   # [256, 128, 1, 1]
    sep1_b = get_init_as_array(model, "conv_6sep_1.bias")     # [256]
    sep2_b = get_init_as_array(model, "conv_6sep_2.bias")     # [256]
    sep_merged_w = np.concatenate([sep1_w, sep2_w], axis=0)   # [512, 128, 1, 1]
    sep_merged_b = np.concatenate([sep1_b, sep2_b], axis=0)   # [512]

    # --- Step 2: Extract and merge PReLU alphas ---
    prelu1_a = get_init_as_array(model, "PReLU_alpha_313")    # [1, 256, 1, 1]
    prelu2_a = get_init_as_array(model, "PReLU_alpha_314")    # [1, 256, 1, 1]
    prelu_merged_a = np.concatenate([prelu1_a, prelu2_a], axis=1)  # [1, 512, 1, 1]

    # --- Step 3: Remove old initializers ---
    old_init_names = [
        "conv_6sep_1.weight", "conv_6sep_1.bias",
        "conv_6sep_2.weight", "conv_6sep_2.bias",
        "PReLU_alpha_313", "PReLU_alpha_314",
    ]
    remove_initializers(model, old_init_names)

    # --- Step 4: Add new merged initializers ---
    model.graph.initializer.extend([
        numpy_helper.from_array(sep_merged_w.astype(np.float32),
                                name="conv_6sep_merged.weight"),
        numpy_helper.from_array(sep_merged_b.astype(np.float32),
                                name="conv_6sep_merged.bias"),
        numpy_helper.from_array(prelu_merged_a.astype(np.float32),
                                name="PReLU_alpha_conv6_merged"),
    ])

    # --- Step 5: Build replacement nodes ---
    conv6_sep_merged_out = "conv6_sep_merged"
    conv6_prelu_merged_out = "conv6_prelu_merged"

    # 5a: Merged sep conv (1x1, 128->512)
    new_sep = helper.make_node(
        "Conv",
        inputs=["res_5_block1_add_98", "conv_6sep_merged.weight",
                "conv_6sep_merged.bias"],
        outputs=[conv6_sep_merged_out],
        name="Conv_conv6_sep_merged",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=1,
    )

    # 5b: Merged PReLU (after sep)
    new_prelu = helper.make_node(
        "PRelu",
        inputs=[conv6_sep_merged_out, "PReLU_alpha_conv6_merged"],
        outputs=[conv6_prelu_merged_out],
        name="PRelu_conv6_sep_merged",
    )

    # --- Step 6: Rebuild node list ---
    nodes_to_remove = {
        "Conv_99",     # s7_sep1
        "Conv_100",    # s7_sep2
        "PRelu_101",   # prelu after sep1
        "PRelu_102",   # prelu after sep2
        "Concat_103",  # concat
    }

    new_nodes = []
    inserted = False
    for node in model.graph.node:
        if node.name in nodes_to_remove:
            if not inserted:
                new_nodes.append(new_sep)
                new_nodes.append(new_prelu)
                inserted = True
            continue
        new_nodes.append(node)

    # Relink: DW conv (Conv_104) input should point to merged PReLU output instead of s7_cat_103
    for node in new_nodes:
        for i, inp in enumerate(node.input):
            if inp == "s7_cat_103":
                node.input[i] = conv6_prelu_merged_out

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    print(f"  [conv_6] Replaced 5 nodes with 2 merged nodes")
    print(f"  [conv_6] s7_cat_103 -> {conv6_prelu_merged_out}")


def verify_no_remaining_splits(model):
    """Check that no 512-channel split/concat patterns remain."""
    for node in model.graph.node:
        if node.op_type == "Concat":
            for attr in node.attribute:
                if attr.name == "axis" and attr.i == 1:
                    raise ValueError(f"Remaining NCHW Concat found: {node.name} "
                                     f"inputs={list(node.input)}")


def count_params(model):
    """Count total parameters (static weights)."""
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total += int(np.prod(arr.shape))
    return total


def run_numerical_verification(orig_path, modified_path):
    """Verify output identity between original and modified models."""
    import onnxruntime as ort

    print("\n" + "=" * 70)
    print("NUMERICAL VERIFICATION")
    print("=" * 70)

    np.random.seed(42)
    test_input = (np.random.rand(1, 3, 112, 112).astype(np.float32) * 255.0) / 127.5 - 1.0

    sess_orig = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
    sess_mod = ort.InferenceSession(modified_path, providers=["CPUExecutionProvider"])

    # Get input/output names
    orig_input_name = sess_orig.get_inputs()[0].name
    mod_input_name = sess_mod.get_inputs()[0].name

    out_orig = sess_orig.run(None, {orig_input_name: test_input})[0]
    out_mod = sess_mod.run(None, {mod_input_name: test_input})[0]

    diff = np.abs(out_orig - out_mod)
    max_diff = diff.max()
    mean_diff = diff.mean()
    cos_sim = np.dot(out_orig.flatten(), out_mod.flatten()) / (
        np.linalg.norm(out_orig) * np.linalg.norm(out_mod)
    )

    print(f"  Original output shape: {out_orig.shape}")
    print(f"  Modified output shape: {out_mod.shape}")
    print(f"  Max absolute diff:     {max_diff:.10f}")
    print(f"  Mean absolute diff:    {mean_diff:.10f}")
    print(f"  Cosine similarity:     {cos_sim:.10f}")
    print(f"  Original output range: [{out_orig.min():.6f}, {out_orig.max():.6f}]")
    print(f"  Modified output range: [{out_mod.min():.6f}, {out_mod.max():.6f}]")

    if max_diff < 1e-5 and cos_sim > 0.9999999:
        print("\n  VERDICT: PASS - Outputs are identical within float32 tolerance.")
    else:
        print(f"\n  VERDICT: WARNING - max_diff={max_diff} exceeds tolerance")

    return max_diff, cos_sim


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("MFN ONNX: Split/Concat -> Single-Path Transformation")
    print("=" * 70)

    # --- Load original model ---
    print(f"\n[1/5] Loading original model: {INPUT_PATH}")
    original = onnx.load(INPUT_PATH)
    checker.check_model(original)
    print(f"  Nodes: {len(original.graph.node)}")
    print(f"  Initializers: {len(original.graph.initializer)}")
    print(f"  Params: {count_params(original):,}")

    # --- Deep copy for modification ---
    model = copy.deepcopy(original)

    # --- Stage 2: Check for Concat nodes ---
    print(f"\n[2/5] Analyzing graph structure...")
    concat_nodes = [n for n in model.graph.node if n.op_type == "Concat"]
    print(f"  Found {len(concat_nodes)} Concat nodes")
    for cn in concat_nodes:
        print(f"    {cn.name}: inputs={list(cn.input)} outputs={list(cn.output)}")

    # --- Apply Fix 1: dconv_45 ---
    print(f"\n[3/5] Applying Fix 1: dconv_45 split -> single-path")
    apply_dconv45_fix(model)

    # --- Apply Fix 2: conv_6 ---
    print(f"\n[4/5] Applying Fix 2: conv_6 split -> single-path")
    apply_conv6_fix(model)

    # --- Verify no remaining splits ---
    verify_no_remaining_splits(model)
    print(f"\n  No remaining NCHW Concat nodes - cleanup complete.")

    # --- Validate, infer shapes, save ---
    print(f"\n[5/5] Validating, shape-inferring, and saving...")
    checker.check_model(model)
    print(f"  ONNX checker: PASS")

    try:
        model = shape_inference.infer_shapes(model)
        print(f"  Shape inference: PASS")
    except Exception as e:
        print(f"  Shape inference: skipped ({e})")

    # Ensure opset 11 (same as original)
    model.opset_import[0].version = 11

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    onnx.save(model, OUTPUT_PATH)

    # --- Stats ---
    original_size = os.path.getsize(INPUT_PATH)
    new_size = os.path.getsize(OUTPUT_PATH)
    orig_params = count_params(original)
    new_params = count_params(model)

    print(f"\n  Output: {OUTPUT_PATH}")

    # --- Show diff ---
    show_graph_diff(original, model)

    # --- Numerical verification ---
    max_diff, cos_sim = run_numerical_verification(INPUT_PATH, OUTPUT_PATH)

    # ============================================================
    # EVIDENCE
    # ============================================================
    print("\n" + "=" * 70)
    print("EVIDENCE")
    print("=" * 70)

    print(f"\n  Graph differences:")
    print(f"    Nodes:     {len(original.graph.node)} -> {len(model.graph.node)}  "
          f"(removed {len(original.graph.node) - len(model.graph.node)})")
    print(f"    Inits:     {len(original.graph.initializer)} -> "
          f"{len(model.graph.initializer)}  "
          f"(removed {len(original.graph.initializer) - len(model.graph.initializer)})")
    print(f"    Params:    {orig_params:,} -> {new_params:,}  "
          f"(diff: {new_params - orig_params:+,})")

    conv_removed = len([n for n in original.graph.node if n.op_type == "Conv"]) - \
                   len([n for n in model.graph.node if n.op_type == "Conv"])
    concat_removed = len([n for n in original.graph.node if n.op_type == "Concat"]) - \
                     len([n for n in model.graph.node if n.op_type == "Concat"])
    prelu_removed = len([n for n in original.graph.node if n.op_type == "PRelu"]) - \
                    len([n for n in model.graph.node if n.op_type == "PRelu"])
    print(f"    Conv removed:   {conv_removed}")
    print(f"    Concat removed: {concat_removed} (all 3)")
    print(f"    PRelu removed:  {prelu_removed}")

    print(f"\n  File sizes:")
    print(f"    Original: {original_size:,} bytes ({original_size / 1024 / 1024:.2f} MB)")
    print(f"    Modified: {new_size:,} bytes ({new_size / 1024 / 1024:.2f} MB)")
    print(f"    Delta:    {new_size - original_size:+,} bytes")

    print(f"\n  Numerical verification:")
    print(f"    Max absolute diff:   {max_diff:.10f}")
    print(f"    Cosine similarity:   {cos_sim:.12f}")
    print(f"    Verdict:             {'PASS' if max_diff < 1e-5 else 'FAIL'}")

    print(f"\n  Cleanup: No remaining 512ch split/concat patterns.")

    print("\nDone.")


if __name__ == "__main__":
    main()
