#!/usr/bin/env python3
"""office4 impostor test on pre-aligned firmware_4dof crops (4 people, 6 pairs)."""
import itertools, sys
import numpy as np, cv2
from ai_edge_litert.interpreter import Interpreter

CROPS = "/home/harvest/project/face-eval/office4/crops"
NAMES = ["1", "2", "3", "4"]
ALIGN = "firmware_4dof"  # matches device alignment


def embed(itp, rgb_u8):
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    s, zp = inp["quantization"]
    x = (rgb_u8.astype(np.float32) - 127.5) / 127.5
    q = np.clip(np.round(x / s).astype(np.int16) + int(zp), -128, 127).astype(np.int8)
    itp.set_tensor(inp["index"], q[None, ...]); itp.invoke()
    raw = itp.get_tensor(out["index"]).astype(np.float32)
    o_s, o_z = out["quantization"]
    if o_s != 0: raw = (raw - o_z) * o_s
    v = raw.reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def run(model_path, label):
    itp = Interpreter(model_path=model_path, num_threads=4); itp.allocate_tensors()
    crops = {n: cv2.cvtColor(cv2.imread(f"{CROPS}/{n}__{ALIGN}.png"), cv2.COLOR_BGR2RGB) for n in NAMES}
    E = {n: embed(itp, crops[n]) for n in NAMES}
    pairs = [(a, b, float(E[a] @ E[b])) for a, b in itertools.combinations(NAMES, 2)]
    v = np.array([p[2] for p in pairs])
    print(f"\n### {label}  ({ALIGN})")
    for a, b, c in pairs:
        print(f"    {a} vs {b} : {c:+.4f}")
    print(f"    mean={v.mean():+.4f}  max={v.max():+.4f}  min={v.min():+.4f}")
    return v


if __name__ == "__main__":
    print("=" * 60)
    print("office4 IMPOSTOR (6 pairs, 4 different people)")
    print("=" * 60)
    for mp, lab in [(sys.argv[1], "QAT distill_v2 512D int8"),
                    ("/home/harvest/project/qat-mfn/training/output/model_distilled_qat_128d.int8.tflite", "in-production 128d int8")]:
        run(mp, lab)
