#!/usr/bin/env python3
"""LFW/CFP INT8 TFLite eval + impostor cosine statistics.

Extends eval_tflite_lfw.py: same KFold accuracy, plus for the different-identity
(issame==False) pairs reports cosine-similarity mean / p99.99 / fraction>=0.4.
Uses single (no-flip) L2-normalized embeddings for the impostor cosine stats
(matches how the device compares a live embedding to an enrolled one).
"""
import argparse, pickle
from io import BytesIO
from pathlib import Path
import numpy as np
import sklearn.model_selection, sklearn.preprocessing
import tensorflow as tf
from PIL import Image


def load_bin(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    bins, issame = data[0], data[1]
    n = len(issame)
    imgs = np.zeros((2 * n, 112, 112, 3), np.float32)
    for i in range(2 * n):
        img = Image.open(BytesIO(bins[i])).convert("RGB")
        if img.size != (112, 112):
            img = img.resize((112, 112), Image.BILINEAR)
        imgs[i] = (np.asarray(img, np.float32) / 127.5) - 1.0
    return imgs, np.asarray(issame, bool)


def run_tflite(itp, imgs):
    ind = itp.get_input_details()[0]; outd = itp.get_output_details()[0]
    isc, izp = ind["quantization"]; osc, ozp = outd["quantization"]
    idt, odt = ind["dtype"], outd["dtype"]
    n = imgs.shape[0]
    itp.set_tensor(ind["index"], np.round(imgs[0:1] / isc + izp).astype(idt) if idt in (np.int8, np.uint8) else imgs[0:1].astype(idt))
    itp.invoke()
    D = itp.get_tensor(outd["index"]).shape[-1]
    emb = np.zeros((n, D), np.float32)
    for i in range(n):
        s = imgs[i:i+1]
        q = np.round(s / isc + izp).astype(idt) if idt in (np.int8, np.uint8) else s.astype(idt)
        itp.set_tensor(ind["index"], q); itp.invoke()
        o = itp.get_tensor(outd["index"]).astype(np.float32)
        if odt in (np.int8, np.uint8):
            o = (o - ozp) * osc
        emb[i] = o[0]
    return emb


def accuracy(emb, issame, nfolds=10):
    emb = sklearn.preprocessing.normalize(emb)
    e1, e2 = emb[0::2], emb[1::2]
    d = np.sum((e1 - e2) ** 2, axis=1)
    thr = np.arange(0, 4, 0.01)
    accs = []
    for tr, te in sklearn.model_selection.KFold(nfolds, shuffle=False).split(d):
        ba, bt = 0, 0
        for t in thr:
            a = np.mean((d[tr] < t) == issame[tr])
            if a > ba: ba, bt = a, t
        accs.append(np.mean((d[te] < bt) == issame[te]))
    return float(np.mean(accs)), float(np.std(accs))


def impostor_stats(emb, issame):
    """Cosine similarity for different-identity pairs (single, no-flip, L2-normalized)."""
    emb = sklearn.preprocessing.normalize(emb)
    e1, e2 = emb[0::2], emb[1::2]
    cos = np.sum(e1 * e2, axis=1)
    imp = cos[~issame]; gen = cos[issame]
    return {
        "impostor_mean": float(imp.mean()),
        "impostor_p99.99": float(np.percentile(imp, 99.99)),
        "impostor_frac>=0.4": float(np.mean(imp >= 0.4)),
        "genuine_mean": float(gen.mean()),
        "separation": float(gen.mean() - imp.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--bins", nargs="+", required=True)
    args = ap.parse_args()
    itp = tf.lite.Interpreter(model_path=args.tflite); itp.allocate_tensors()
    ind = itp.get_input_details()[0]; outd = itp.get_output_details()[0]
    print(f"model {args.tflite}")
    print(f"  in  {ind['shape']} {ind['dtype']} {ind['quantization']}")
    print(f"  out {outd['shape']} {outd['dtype']} {outd['quantization']}")
    for b in args.bins:
        name = Path(b).stem
        imgs, issame = load_bin(b)
        e_nf = run_tflite(itp, imgs)
        e_fl = run_tflite(itp, imgs[:, :, ::-1, :].copy())
        acc, std = accuracy(e_nf + e_fl, issame)
        st = impostor_stats(e_nf, issame)  # single pass for impostor cos
        print(f"\n=== {name}  pairs={len(issame)} ===")
        print(f"  acc(flip-fused)={acc*100:.2f}% +/-{std*100:.2f}")
        print(f"  impostor_mean={st['impostor_mean']:.4f}  p99.99={st['impostor_p99.99']:.4f}  "
              f"frac>=0.4={st['impostor_frac>=0.4']*100:.3f}%")
        print(f"  genuine_mean={st['genuine_mean']:.4f}  separation={st['separation']:.4f}")


if __name__ == "__main__":
    main()
