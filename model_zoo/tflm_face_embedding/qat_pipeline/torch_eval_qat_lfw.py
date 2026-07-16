#!/usr/bin/env python3
"""Diagnostic: eval the QAT model (fake-quant, frozen ranges) in-torch on LFW/CFP.

Isolates QAT quality from TFLite export range-mismatch. Loads the QAT state_dict
WITH the FakeQuantize wrappers (so activations are int8-simulated with the frozen
QAT ranges), runs on the .bin pairs, reports acc + impostor cosine. 512D (no PCA)."""
import argparse, pickle, sys, os
from io import BytesIO
import numpy as np
import torch, torch.nn.functional as F
import sklearn.preprocessing, sklearn.model_selection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qat_finetune import (MobileFaceNet, insert_activation_fake_quantize,
                          freeze_batchnorm, get_fake_quantize_modules)


def load_bin(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    bins, issame = data[0], data[1]
    n = len(issame)
    imgs = np.zeros((2 * n, 3, 112, 112), np.float32)
    for i in range(2 * n):
        img = Image.open(BytesIO(bins[i])).convert("RGB")
        if img.size != (112, 112):
            img = img.resize((112, 112), Image.BILINEAR)
        a = (np.asarray(img, np.float32) / 127.5) - 1.0
        imgs[i] = a.transpose(2, 0, 1)
    return imgs, np.asarray(issame, bool)


@torch.no_grad()
def embed(model, imgs, device, bs=256):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i+bs]).to(device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out, 0)


def accuracy(emb, issame, nfolds=10):
    emb = sklearn.preprocessing.normalize(emb)
    e1, e2 = emb[0::2], emb[1::2]
    d = np.sum((e1 - e2) ** 2, axis=1)
    thr = np.arange(0, 4, 0.01); accs = []
    for tr, te in sklearn.model_selection.KFold(nfolds, shuffle=False).split(d):
        ba, bt = 0, 0
        for t in thr:
            a = np.mean((d[tr] < t) == issame[tr])
            if a > ba: ba, bt = a, t
        accs.append(np.mean((d[te] < bt) == issame[te]))
    return float(np.mean(accs)), float(np.std(accs))


def imp(emb, issame):
    emb = sklearn.preprocessing.normalize(emb)
    cos = np.sum(emb[0::2] * emb[1::2], axis=1)
    return float(cos[~issame].mean()), float(cos[issame].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat-pt", required=True)
    ap.add_argument("--bins", nargs="+", required=True)
    ap.add_argument("--float", action="store_true", help="eval clean float (no fake-quant)")
    args = ap.parse_args()
    device = "cuda"
    m = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    if args.float:
        sd = torch.load(args.qat_pt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "backbone" in sd: sd = sd["backbone"]
        keys = set(m.state_dict().keys())
        sd = {k.replace(".block.", "."): v for k, v in sd.items() if k.replace(".block.", ".") in keys}
        m.load_state_dict(sd, strict=True)
        m = m.to(device).eval()
        tag = "FLOAT (no quant)"
    else:
        m = insert_activation_fake_quantize(m)
        freeze_batchnorm(m)
        sd = torch.load(args.qat_pt, map_location="cpu", weights_only=False)
        m.load_state_dict(sd, strict=True)
        m = m.to(device).eval()
        for _, fq in get_fake_quantize_modules(m):
            fq.eval()  # frozen ranges, fake-quant active
        tag = "QAT int8-sim (frozen ranges)"
    print(f"=== {tag}  {args.qat_pt} ===")
    for b in args.bins:
        imgs, issame = load_bin(b)
        e = embed(m, imgs, device)
        acc, std = accuracy(e, issame)
        im, gen = imp(e, issame)
        print(f"  {os.path.basename(b):10s} acc={acc*100:.2f}% +/-{std*100:.2f}  "
              f"impostor_mean={im:.4f} genuine_mean={gen:.4f} sep={gen-im:.4f} xnorm={np.linalg.norm(e,axis=1).mean():.2f}")


if __name__ == "__main__":
    main()
