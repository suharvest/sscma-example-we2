#!/usr/bin/env python3
"""Convert all PReLU ops to ReLU in the MFN singlepath ONNX model.

The model has 33 PReLU ops with per-channel alpha parameters (most near zero).
Replacing with ReLU enables Vela conv+activation fusion, reducing SRAM usage.

PReLU alpha stats: mean=0.019, median=0.039, 97% abs < 0.5
"""

import onnx
from onnx import TensorProto, helper
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_ONNX = SCRIPT_DIR / "mfn_s8_v1_singlepath.onnx"
OUTPUT_ONNX = SCRIPT_DIR / "mfn_s8_v1_singlepath_relu.onnx"


def convert_prelu_to_relu():
    model = onnx.load(str(INPUT_ONNX))
    graph = model.graph

    # Find all PReLU nodes and their alpha initializer names
    prelu_nodes = [n for n in graph.node if n.op_type == "PRelu"]
    alpha_names = set()
    for n in prelu_nodes:
        # PReLU: inputs = [X, slope]
        alpha_names.add(n.input[1])

    print(f"Found {len(prelu_nodes)} PReLU nodes, {len(alpha_names)} alpha initializers")

    # Replace PReLU with ReLU (ReLU takes only 1 input)
    for node in graph.node:
        if node.op_type == "PRelu":
            node.op_type = "Relu"
            # Remove the alpha/slope input (keep only input[0])
            del node.input[1:]

    # Remove alpha initializers from the graph
    new_initializers = [i for i in graph.initializer if i.name not in alpha_names]
    removed = len(graph.initializer) - len(new_initializers)
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    print(f"Removed {removed} alpha initializers, {len(new_initializers)} remaining")

    # Also clean up orphaned alpha inputs in node inputs (belt and suspenders)
    for node in graph.node:
        node.input[:] = [inp for inp in node.input if inp not in alpha_names]

    # Verify
    relu_count = sum(1 for n in graph.node if n.op_type == "Relu")
    prelu_count = sum(1 for n in graph.node if n.op_type == "PRelu")
    print(f"After conversion: {relu_count} ReLU, {prelu_count} PReLU")

    onnx.checker.check_model(model)
    onnx.save(model, str(OUTPUT_ONNX))
    print(f"Saved {OUTPUT_ONNX}")


if __name__ == "__main__":
    convert_prelu_to_relu()
