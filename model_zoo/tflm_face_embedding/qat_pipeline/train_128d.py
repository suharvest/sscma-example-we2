#!/usr/bin/env python3
"""End-to-end 128D QAT (drop-in for EMBEDDING_OUTPUT_DIM=128).

Technique 1: the 512->128 projection is a trainable Linear that participates in QAT
             (not post-hoc PCA), AND its 128D output is FakeQuantized -> the projection
             is quantization-aware. Warm-started from a PCA composition of the 512D
             model's final Linear+BN so we begin at PCA-float quality.
Technique 2: --act relu6 swaps LeakyReLU->ReLU6 (bounded activation, int8-friendly).
Distillation: teacher = float distill_v2 512D (LeakyReLU); target = L2-normalized
              PCA-128 projection of the teacher. Loss = (1-cos) + w*mse.

Recipe matches the proven-stable qat_finetune_if.py: calibrate observers then FREEZE
ranges, BN frozen, self/teacher-distill for N epochs.
"""
import argparse, os, sys, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfn_cfg import (MobileFaceNet, insert_activation_fake_quantize, freeze_batchnorm,
                     get_fake_quantize_modules)

# Teacher uses the original LeakyReLU 512D definition from qat_finetune.py
from qat_finetune import MobileFaceNet as MobileFaceNet512


class ImageFolderFaces(Dataset):
    def __init__(self, root, image_size=112, limit=None):
        self.paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
        if limit:
            self.paths = self.paths[:limit]
        assert self.paths, f"No images under {root}"
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


def load_512d_sd(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "backbone" in ckpt:
        sd = ckpt["backbone"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return {k.replace("module.", ""): v for k, v in sd.items()}


@torch.no_grad()
def compute_pca(teacher, loader, device, n_imgs=3000):
    teacher.eval()
    embs, seen = [], 0
    for images in loader:
        embs.append(teacher(images.to(device)).cpu().numpy())
        seen += images.size(0)
        if seen >= n_imgs:
            break
    embs = np.concatenate(embs, 0)[:n_imgs].astype(np.float64)
    mean = embs.mean(0)
    U, S, Vt = np.linalg.svd(embs - mean, full_matrices=False)
    proj = Vt[:128].T  # (512,128)
    var = (S[:128] ** 2).sum() / (S ** 2).sum()
    print(f"PCA 512->128 kept variance = {var*100:.2f}% over {len(embs)} teacher embeddings", flush=True)
    return mean.astype(np.float32), proj.astype(np.float32)


def init_projection_from_pca(student, sd512, mean_pca, proj, device):
    """Warm-start student final Linear(512->128)+BN128 to reproduce
    proj^T @ (BN(W_lin @ f) - mean_pca)."""
    W_lin = sd512["features.layers.2.weight"].cpu().numpy().astype(np.float64)      # (512,512)
    gamma = sd512["features.layers.3.weight"].cpu().numpy().astype(np.float64)
    beta = sd512["features.layers.3.bias"].cpu().numpy().astype(np.float64)
    rmean = sd512["features.layers.3.running_mean"].cpu().numpy().astype(np.float64)
    rvar = sd512["features.layers.3.running_var"].cpu().numpy().astype(np.float64)
    eps = 1e-5
    A = gamma / np.sqrt(rvar + eps)                # (512,)
    b = beta - gamma * rmean / np.sqrt(rvar + eps)  # (512,)
    P = proj.T.astype(np.float64)                  # (128,512)
    new_W = P @ (A[:, None] * W_lin)               # (128,512)
    new_bias = P @ (b - mean_pca.astype(np.float64))  # (128,)
    with torch.no_grad():
        student.features.layers[2].weight.copy_(torch.from_numpy(new_W.astype(np.float32)))
        bn = student.features.layers[3]
        bn.weight.fill_(1.0)
        bn.bias.copy_(torch.from_numpy(new_bias.astype(np.float32)))
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
    print("Initialized 128D projection from PCA composition", flush=True)


def distill_loss(s, t128, mse_w=0.1):
    cos = F.cosine_similarity(s, t128, dim=1)
    return (1 - cos).mean() + mse_w * F.mse_loss(s, t128), (1 - cos).mean().item()


@torch.no_grad()
def fidelity(student, teacher, proj_t, mean_t, loader, device, max_b=40):
    student.eval(); teacher.eval()
    for _, fq in get_fake_quantize_modules(student):
        fq.eval()
    cs = []
    for bi, images in enumerate(loader):
        if bi >= max_b:
            break
        images = images.to(device)
        t = teacher(images)
        t128 = (t - mean_t) @ proj_t  # (B,128)
        t128 = F.normalize(t128, dim=1)
        s = F.normalize(student(images), dim=1)
        cs.extend(F.cosine_similarity(s, t128, dim=1).cpu().tolist())
    c = np.array(cs)
    return float(c.mean()), float(c.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="float distill_v2 512D .pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--act", default="leaky", choices=["leaky", "relu6", "relu"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--mse-w", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}  act={args.act}", flush=True)

    sd512 = load_512d_sd(args.teacher)

    # Teacher (float LeakyReLU 512D)
    teacher = MobileFaceNet512(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    teacher.load_state_dict(sd512, strict=True)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Data
    ds = ImageFolderFaces(args.data)
    print(f"Dataset: {len(ds)} images from {args.data}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # PCA target basis from teacher
    mean_pca, proj = compute_pca(teacher, loader, device, n_imgs=3000)
    np.savez(os.path.join(args.output, "pca_basis.npz"), mean=mean_pca, projection=proj)
    proj_t = torch.from_numpy(proj).to(device)          # (512,128)
    mean_t = torch.from_numpy(mean_pca).to(device)      # (512,)

    # Student 128D
    student = MobileFaceNet(num_features=128, blocks=(1, 4, 6, 2), scale=1, act=args.act)
    # load shared backbone weights (skip final Linear/BN128 which differ in shape)
    ssd = student.state_dict()
    load = {k: v for k, v in sd512.items() if k in ssd and ssd[k].shape == v.shape}
    student.load_state_dict(load, strict=False)
    print(f"Loaded {len(load)}/{len(ssd)} shared backbone tensors", flush=True)
    init_projection_from_pca(student, sd512, mean_pca, proj, device)
    student = student.to(device)

    # sanity: float fidelity before FQ
    student.eval()
    m0, s0 = fidelity(student, teacher, proj_t, mean_t, loader, device, max_b=20)
    print(f"Float (pre-FQ) fidelity vs teacher-PCA128: {m0:.4f} +/- {s0:.4f}", flush=True)

    # Insert FQ, freeze BN
    student = insert_activation_fake_quantize(student).to(device)
    freeze_batchnorm(student)
    fqs = get_fake_quantize_modules(student)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"FQ modules: {len(fqs)}  trainable params: {trainable:,}", flush=True)

    # Calibrate observers then FREEZE ranges
    student.train()
    for _, fq in fqs:
        fq.train()
    freeze_batchnorm(student)
    with torch.no_grad():
        for bi, images in enumerate(loader):
            if bi >= 60:
                break
            student(images.to(device))
    for _, fq in fqs:
        fq.disable_observer(); fq.enable_fake_quant()
    m, s = fidelity(student, teacher, proj_t, mean_t, loader, device)
    print(f"Pre-QAT int8-sim fidelity (frozen ranges): {m:.4f} +/- {s:.4f}", flush=True)
    best = m
    torch.save(student.state_dict(), os.path.join(args.output, "model_qat_best.pt"))

    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        student.train()
        for _, fq in get_fake_quantize_modules(student):
            fq.eval()
        freeze_batchnorm(student)
        t0 = time.time(); tot = 0.0; n = 0
        for bi, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                t = teacher(images)
                t128 = F.normalize((t - mean_t) @ proj_t, dim=1)
            s_emb = F.normalize(student(images), dim=1)
            loss, lc = distill_loss(s_emb, t128, args.mse_w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * images.size(0); n += images.size(0)
            if bi % 20 == 0:
                print(f"  E{epoch} [{bi:4d}/{len(loader)}] loss={loss.item():.5f} "
                      f"cosloss={lc:.5f} avg={tot/n:.5f} {n/(time.time()-t0+1e-9):.0f} img/s", flush=True)
        sched.step()
        torch.save(student.state_dict(), os.path.join(args.output, f"model_qat_e{epoch}.pt"))
        m, s = fidelity(student, teacher, proj_t, mean_t, loader, device)
        tag = ""
        if m > best:
            best = m; tag = "  *BEST*"
            torch.save(student.state_dict(), os.path.join(args.output, "model_qat_best.pt"))
        print(f"== Epoch {epoch} done: avg_loss={tot/n:.5f}  int8-sim fidelity={m:.4f} +/- {s:.4f}"
              f"  ({time.time()-t0:.0f}s){tag}", flush=True)

    torch.save(student.state_dict(), os.path.join(args.output, "model_qat.pt"))
    print(f"\nFinal best int8-sim fidelity: {best:.4f}", flush=True)
    print("=== 128D QAT complete ===", flush=True)


if __name__ == "__main__":
    main()
