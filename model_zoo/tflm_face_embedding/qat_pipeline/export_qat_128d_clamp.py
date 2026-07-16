#!/usr/bin/env python3
"""Clamp export: QAT .pt -> clean model with hard clamps at the QAT activation ranges
   -> PTQ INT8 TFLite -> Vela.  Robust alternative to the onnx2tf QDQ path.

Mechanism: the PTQ collapse came from TFLite recomputing per-tensor activation ranges
via min/max over the outlier-heavy LeakyReLU outputs (huge range -> signal crushed).
Here we replace each trained FakeQuantize with torch.clamp to that fake-quant's exact
learned [lo,hi] range. The clean model's activations are then bounded, so TFLite's
representative-dataset min/max == the tight QAT range -> tight scale -> the QAT-adapted
weights transfer and no collapse. Standard onnx2tf PTQ path (already validated on Vela).
"""
import argparse, os, shutil, subprocess, sys, glob
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from qat_finetune import (MobileFaceNet, insert_activation_fake_quantize,
                          freeze_batchnorm)
from torch.quantization import FakeQuantize


class Clamp(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        self.lo, self.hi = float(lo), float(hi)

    def forward(self, x):
        return torch.clamp(x, self.lo, self.hi)


def build_clamped_backbone(qat_pt):
    m = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    m = insert_activation_fake_quantize(m)
    freeze_batchnorm(m)
    sd = torch.load(qat_pt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd, strict=True)
    m.eval()
    # Replace each FakeQuantize with a hard Clamp to its learned [lo,hi]
    n = 0
    for parent in m.modules():
        for cname, child in list(parent.named_children()):
            if isinstance(child, FakeQuantize):
                s = float(child.scale); z = int(child.zero_point)
                lo = (-128 - z) * s; hi = (127 - z) * s
                setattr(parent, cname, Clamp(lo, hi))
                n += 1
    print(f"replaced {n} FakeQuantize -> Clamp")
    return m.eval()


def load_calib(calib_dir, n):
    paths = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))[:n]
    if not paths:
        paths = sorted(glob.glob(os.path.join(calib_dir, "**", "*.jpg"), recursive=True))[:n]
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size != (112, 112):
            img = img.resize((112, 112), Image.BILINEAR)
        out.append((np.asarray(img, np.float32) / 127.5) - 1.0)
    return np.asarray(out, np.float32)


class Projected128D(nn.Module):
    def __init__(self, backbone, mean512, proj):
        super().__init__()
        self.backbone = backbone
        p = nn.Linear(512, 128, bias=True)
        with torch.no_grad():
            p.weight.copy_(torch.from_numpy(proj.T))
            p.bias.copy_(-torch.from_numpy(mean512) @ torch.from_numpy(proj))
        self.proj = p

    def forward(self, x):
        return self.proj(self.backbone(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat-pt", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-pca", type=int, default=3000)
    ap.add_argument("--num-calib", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = build_clamped_backbone(args.qat_pt).to(device)

    imgs = load_calib(args.calib_dir, args.num_pca)
    with torch.no_grad():
        x = torch.from_numpy(imgs.transpose(0, 3, 1, 2))
        embs = np.concatenate([backbone(x[i:i+256].to(device)).cpu().numpy()
                               for i in range(0, len(x), 256)], 0)
    mean = embs.mean(0).astype(np.float32)
    U, S, Vt = np.linalg.svd(embs - mean, full_matrices=False)
    proj = Vt[:128].T.astype(np.float32)
    var = (S[:128] ** 2).sum() / (S ** 2).sum()
    print(f"PCA (clamped embeddings) kept variance = {var*100:.2f}%")
    np.savez(out / "pca_128.npz", mean=mean, projection=proj)

    model = Projected128D(backbone.cpu(), mean, proj).eval()
    with torch.no_grad():
        e = model(torch.randn(2, 3, 112, 112))
    print(f"smoke 128D xnorm={float(torch.norm(e,dim=1).mean()):.2f}")

    onnx_path = out / "model_qat_128d.onnx"
    torch.onnx.export(model, torch.randn(1, 3, 112, 112), str(onnx_path),
                      input_names=["input"], output_names=["embedding"],
                      opset_version=13, dynamo=False)
    print(f"onnx: {onnx_path}")

    import onnx2tf, tensorflow as tf
    tf_dir = out / "saved_model"
    if tf_dir.exists():
        shutil.rmtree(tf_dir)
    onnx2tf.convert(input_onnx_file_path=str(onnx_path),
                    output_folder_path=str(tf_dir), non_verbose=True,
                    copy_onnx_input_output_names_to_tflite=True)
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
    int8 = out / "model_qat_128d.int8.tflite"
    int8.write_bytes(conv.convert())
    print(f"INT8 tflite: {int8} ({int8.stat().st_size/1024:.1f} KiB)")

    vela = shutil.which("vela") or str(Path(sys.executable).parent / "vela")
    r = subprocess.run([vela, str(int8), "--accelerator-config", "ethos-u55-64",
                        "--optimise", "Performance", "--output-dir", str(out)],
                       capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode:
        print(r.stderr[-1500:]); raise SystemExit("vela failed")
    print("DONE")


if __name__ == "__main__":
    main()
