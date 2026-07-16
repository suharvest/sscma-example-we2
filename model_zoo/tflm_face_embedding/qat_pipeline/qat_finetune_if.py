#!/usr/bin/env python3
"""QAT fine-tuning (self-distillation) for 512D LeakyReLU MobileFaceNet -> INT8-robust.

Adapted from training/qat_finetune.py for the spark GB10 box:
  * Data: ImageFolder of aligned 112x112 JPGs (glint360k subset) instead of MXNet .rec
  * Teacher: FP32 copy of the SAME distilled student (self-distillation) so the
    QAT-int8 model reproduces distill_v2's float discriminability.
  * Loss: cosine-embedding (1 - cos) + small MSE term. Cosine matches face-verification
    geometry better than raw MSE.
  * FakeQuantize (per-tensor affine, int8) after every ConvBlock/LinearBlock output,
    matching Ethos-U55 per-tensor activation quantization. This is exactly what fixes
    the PTQ collapse (LeakyReLU unbounded positive range crushed by per-tensor scale).

Single-GPU. Checkpoints every epoch. Reports cosine fidelity (float-teacher vs
int8-simulated student) each epoch — the metric that predicts int8 tflite quality.
"""
import argparse, os, sys, time, glob
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the proven model def + FQ insertion from the original script
from qat_finetune import (
    MobileFaceNet, insert_activation_fake_quantize, freeze_batchnorm,
    get_fake_quantize_modules,
)


class ImageFolderFaces(Dataset):
    """Recursively load aligned 112x112 face JPGs. Labels unused (self-distill)."""
    def __init__(self, root, image_size=112):
        self.paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
        if not self.paths:
            self.paths = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
        assert self.paths, f"No images under {root}"
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [-1,1]
        arr = np.transpose(arr, (2, 0, 1))                       # CHW
        return torch.from_numpy(arr)


def distill_loss(student_emb, teacher_emb, mse_w=0.1):
    cos = F.cosine_similarity(student_emb, teacher_emb, dim=1)
    loss_cos = (1.0 - cos).mean()
    loss_mse = F.mse_loss(student_emb, teacher_emb)
    return loss_cos + mse_w * loss_mse, loss_cos.item(), loss_mse.item()


@torch.no_grad()
def cosine_fidelity(model, teacher, loader, device, max_batches=40):
    model.eval(); teacher.eval()
    for _, fq in get_fake_quantize_modules(model):
        fq.eval()  # quantize with frozen ranges, no observer update
    cosines = []
    for bi, images in enumerate(loader):
        if bi >= max_batches:
            break
        images = images.to(device)
        t = teacher(images); s = model(images)
        cosines.extend(F.cosine_similarity(t, s, dim=1).cpu().tolist())
    c = np.array(cosines)
    return float(c.mean()), float(c.std())


def calibrate(model, loader, device, num_batches=25):
    model.train()
    for _, fq in get_fake_quantize_modules(model):
        fq.train()
    with torch.no_grad():
        for bi, images in enumerate(loader):
            if bi >= num_batches:
                break
            model(images.to(device))
    print(f"  calibrated observers on {num_batches} batches", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--mse-w", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}", flush=True)

    # ---- Build + load distilled student ----
    model = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "backbone" in ckpt:
        sd = ckpt["backbone"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    print(f"Loaded student from {args.model}", flush=True)

    # ---- Frozen FP32 self-teacher ----
    teacher = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    teacher.load_state_dict(sd, strict=True)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Insert FakeQuantize + freeze BN ----
    model = insert_activation_fake_quantize(model)
    freeze_batchnorm(model)
    model = model.to(device)
    fqs = get_fake_quantize_modules(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FakeQuantize modules: {len(fqs)}  trainable params: {trainable:,}", flush=True)

    # ---- Data ----
    ds = ImageFolderFaces(args.data)
    print(f"Dataset: {len(ds)} images from {args.data}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- Baseline: calibrate observers on many batches, then FREEZE quant ranges ----
    # Fine-tune weights against FIXED int8 activation ranges (stable, matches how the
    # TFLite converter fixes ranges from calibration). Moving observers during multi-epoch
    # training create a weight<->range feedback loop that diverges.
    print("\n=== calibrate observers (pre-QAT) ===", flush=True)
    calibrate(model, loader, device, num_batches=60)
    # Freeze observers for the whole run: keep fake_quant on, stop range updates.
    for _, fq in get_fake_quantize_modules(model):
        fq.disable_observer()
        fq.enable_fake_quant()
    m, s = cosine_fidelity(model, teacher, loader, device)
    print(f"Pre-QAT int8-sim cosine fidelity (frozen ranges): {m:.4f} +/- {s:.4f}", flush=True)
    best_fid = m
    torch.save(model.state_dict(), os.path.join(args.output, "model_qat_best.pt"))

    # ---- Train (observers stay frozen; fq.eval() so ranges never move) ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _, fq in get_fake_quantize_modules(model):
            fq.eval()  # frozen ranges: fake-quant active, observer disabled
        freeze_batchnorm(model)  # keep BN frozen even in train mode
        t0 = time.time(); tot = 0.0; n = 0
        for bi, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                t_emb = teacher(images)
            s_emb = model(images)
            loss, lc, lm = distill_loss(s_emb, t_emb, args.mse_w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * images.size(0); n += images.size(0)
            if bi % 20 == 0:
                sps = n / (time.time() - t0 + 1e-9)
                print(f"  E{epoch} [{bi:4d}/{len(loader)}] loss={loss.item():.5f} "
                      f"cos={lc:.5f} mse={lm:.5f} avg={tot/n:.5f} {sps:.0f} img/s", flush=True)
        sched.step()
        ckpt_path = os.path.join(args.output, f"model_qat_e{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        m, s = cosine_fidelity(model, teacher, loader, device)
        tag = ""
        if m > best_fid:
            best_fid = m
            torch.save(model.state_dict(), os.path.join(args.output, "model_qat_best.pt"))
            tag = "  *BEST*"
        print(f"== Epoch {epoch} done: avg_loss={tot/n:.5f}  int8-sim cosine fidelity={m:.4f} +/- {s:.4f}  "
              f"saved {ckpt_path}  ({time.time()-t0:.0f}s){tag}", flush=True)

    torch.save(model.state_dict(), os.path.join(args.output, "model_qat.pt"))
    print(f"\nFinal: {os.path.join(args.output, 'model_qat.pt')}", flush=True)
    print("=== QAT fine-tuning complete ===", flush=True)


if __name__ == "__main__":
    main()
