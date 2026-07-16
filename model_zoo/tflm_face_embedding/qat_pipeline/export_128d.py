#!/usr/bin/env python3
"""Clamp export for end-to-end 128D QAT model -> INT8 TFLite -> Vela.

Replaces every trained FakeQuantize (including the embedding-output FQ) with a hard
torch.clamp to its learned [lo,hi]. Bounded activations + bounded 128D output => TFLite
representative-dataset min/max == tight QAT ranges => tight scale => no int8 collapse.
"""
import argparse, os, shutil, subprocess, sys, glob
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from mfn_cfg import MobileFaceNet, insert_activation_fake_quantize, freeze_batchnorm
from torch.quantization import FakeQuantize


class Clamp(nn.Module):
    def __init__(self, lo, hi):
        super().__init__(); self.lo, self.hi = float(lo), float(hi)
    def forward(self, x):
        return torch.clamp(x, self.lo, self.hi)


def build_clamped(qat_pt, act):
    m = MobileFaceNet(num_features=128, blocks=(1, 4, 6, 2), scale=1, act=act)
    m = insert_activation_fake_quantize(m); freeze_batchnorm(m)
    m.load_state_dict(torch.load(qat_pt, map_location="cpu", weights_only=False), strict=True)
    m.eval()
    n = 0
    for parent in m.modules():
        for cname, child in list(parent.named_children()):
            if isinstance(child, FakeQuantize):
                s = float(child.scale); z = int(child.zero_point)
                setattr(parent, cname, Clamp((-128 - z) * s, (127 - z) * s)); n += 1
    print(f"replaced {n} FakeQuantize -> Clamp")
    return m.eval()


def load_calib(d, n):
    ps = sorted(glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True))[:n]
    out = []
    for p in ps:
        img = Image.open(p).convert("RGB")
        if img.size != (112, 112):
            img = img.resize((112, 112), Image.BILINEAR)
        out.append((np.asarray(img, np.float32) / 127.5) - 1.0)
    return np.asarray(out, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat-pt", required=True)
    ap.add_argument("--act", default="leaky", choices=["leaky", "relu6", "relu"])
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-calib", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model = build_clamped(args.qat_pt, args.act).eval()
    with torch.no_grad():
        e = model(torch.randn(2, 3, 112, 112))
    print(f"smoke 128D xnorm={float(torch.norm(e,dim=1).mean()):.2f}")

    onnx_path = out / "model_128d.onnx"
    torch.onnx.export(model, torch.randn(1, 3, 112, 112), str(onnx_path),
                      input_names=["input"], output_names=["embedding"],
                      opset_version=13, dynamo=False)
    print(f"onnx: {onnx_path}")

    import onnx2tf, tensorflow as tf
    tf_dir = out / "saved_model"
    if tf_dir.exists(): shutil.rmtree(tf_dir)
    onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(tf_dir),
                    non_verbose=True, copy_onnx_input_output_names_to_tflite=True)
    calib = load_calib(args.calib_dir, args.num_calib)

    def rep():
        for im in calib:
            yield [im[np.newaxis, ...].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    int8 = out / "model_128d.int8.tflite"
    int8.write_bytes(conv.convert())
    print(f"INT8 tflite: {int8} ({int8.stat().st_size/1024:.1f} KiB)")

    vela = shutil.which("vela") or str(Path(sys.executable).parent / "vela")
    r = subprocess.run([vela, str(int8), "--accelerator-config", "ethos-u55-64",
                        "--optimise", "Performance", "--output-dir", str(out)],
                       capture_output=True, text=True)
    print(r.stdout[-1800:])
    if r.returncode:
        print(r.stderr[-1500:]); raise SystemExit("vela failed")
    print("DONE")


if __name__ == "__main__":
    main()
