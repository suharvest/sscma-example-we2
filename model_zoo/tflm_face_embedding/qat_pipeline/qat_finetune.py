#!/usr/bin/env python3
"""
QAT fine-tuning for 512D LeakyReLU MobileFaceNet to fix INT8 quantization collapse.

Problem: FP32 model achieves 83.5% LFW but collapse to 57% with per-tensor INT8.
Root cause: LeakyReLU activations have unbounded positive range; per-tensor quantization
scale is dominated by outliers, crushing most signal.

Solution: Insert FakeQuantize after every Conv2d output (before BatchNorm) to simulate
INT8 activation quantization during fine-tuning. Train for 1-2 epochs with distillation
loss (MSE between FP32 teacher and QAT student embeddings).

Usage (on vast.ai / GPU instance):
    source /workspace/venv/bin/activate
    export DATA_DIR=/data/ms1m-retinaface-t1
    python qat_finetune.py \
        --model /path/to/model.pt \
        --output /workspace/output/qat_finetuned \
        --epochs 2 \
        --lr 0.001 \
        --batch-size 128

Output:
    model_qat.pt          — QAT fine-tuned weights (FP32, quantization-friendly, with FQ wrappers)
    model_qat_clean.onnx  — ONNX export (clean model with QAT-tuned weights, for TFLite INT8 conversion)
"""

import argparse
import os
import sys
import time
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.quantization import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Model definition (mirrors backbones.py for standalone use)
# ---------------------------------------------------------------------------

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_c),
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
        )

    def forward(self, x):
        short_cut = x if self.residual else None
        x = self.layers(x)
        if self.residual:
            x = short_cut + x
        return x


class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = [DepthWise(c, c, True, kernel, stride, padding, groups) for _ in range(num_block)]
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(nn.Module):
    """Global Depthwise Conv -> Flatten -> Linear -> BN"""
    def __init__(self, embedding_size):
        super().__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(nn.Module):
    def __init__(self, num_features=512, blocks=(1, 4, 6, 2), scale=1):
        super().__init__()
        self.scale = scale
        self.layers = nn.ModuleList()

        # Stage 0: 112->56
        self.layers.append(
            ConvBlock(3, 64 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * scale, 64 * scale, kernel=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=64 * scale)
            )
        else:
            self.layers.append(
                Residual(64 * scale, num_block=blocks[0], groups=128, kernel=(3, 3),
                         stride=(1, 1), padding=(1, 1)),
            )

        # Stage 1-3: downsample + residual blocks
        self.layers.extend([
            DepthWise(64 * scale, 64 * scale, kernel=(3, 3), stride=(2, 2),
                      padding=(1, 1), groups=128),
            Residual(64 * scale, num_block=blocks[1], groups=128, kernel=(3, 3),
                     stride=(1, 1), padding=(1, 1)),
            DepthWise(64 * scale, 128 * scale, kernel=(3, 3), stride=(2, 2),
                      padding=(1, 1), groups=256),
            Residual(128 * scale, num_block=blocks[2], groups=256, kernel=(3, 3),
                     stride=(1, 1), padding=(1, 1)),
            DepthWise(128 * scale, 128 * scale, kernel=(3, 3), stride=(2, 2),
                      padding=(1, 1), groups=512),
            Residual(128 * scale, num_block=blocks[3], groups=256, kernel=(3, 3),
                     stride=(1, 1), padding=(1, 1)),
        ])

        self.conv_sep = ConvBlock(128 * scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(num_features)

    def forward(self, x):
        for func in self.layers:
            x = func(x)
        x = self.conv_sep(x)
        x = self.features(x)
        return x


# ---------------------------------------------------------------------------
# QAT: Insert FakeQuantize after every Conv2d (before BN)
# ---------------------------------------------------------------------------

def _make_fake_quantize():
    """Create a per-tensor affine FakeQuantize with moving-average min/max observer."""
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )


def insert_activation_fake_quantize(model: nn.Module):
    """
    Insert FakeQuantize after each ConvBlock output (post-LeakyReLU) and each
    LinearBlock output (post-BN).  This simulates INT8 per-tensor activation
    quantization at every layer boundary.

    Returns a list of (name, module) for the inserted FakeQuantize modules so
    the caller can control their training state.
    """
    replacements = []  # (parent_module, attr_name, new_child)

    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child in list(parent_module.named_children()):
            # Insert FQ after ConvBlock
            if isinstance(child, ConvBlock):
                fq = _make_fake_quantize()
                wrapper = nn.Sequential(OrderedDict([
                    ('block', child),
                    ('fq', fq),
                ]))
                replacements.append((parent_module, child_name, wrapper, f'{parent_name}.{child_name}'))

            # Insert FQ after LinearBlock
            elif isinstance(child, LinearBlock):
                fq = _make_fake_quantize()
                wrapper = nn.Sequential(OrderedDict([
                    ('block', child),
                    ('fq', fq),
                ]))
                replacements.append((parent_module, child_name, wrapper, f'{parent_name}.{child_name}'))

    for parent, attr, wrapper, name in replacements:
        setattr(parent, attr, wrapper)

    # Also wrap the main model's input with a FakeQuantize
    # (We'll do this differently: wrap the whole model's forward)

    print(f"Inserted {len(replacements)} FakeQuantize modules")
    return model


def freeze_batchnorm(model: nn.Module):
    """Put all BatchNorm layers in eval mode and freeze their stats."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def get_fake_quantize_modules(model: nn.Module):
    """Return all FakeQuantize modules in the model."""
    fqs = []
    for name, m in model.named_modules():
        if isinstance(m, FakeQuantize):
            fqs.append((name, m))
    return fqs


# ---------------------------------------------------------------------------
# Data loader using MXNet RecordIO
# ---------------------------------------------------------------------------

class MXRecordDataset(Dataset):
    """Load aligned face images from MXNet RecordIO files.

    Uses the same .rec format as insightface/arcface_torch.
    Multi-worker DataLoader works on Linux (fork mode copies MXIndexedRecordIO
    file handles to child processes).
    """

    def __init__(self, root_dir: str, image_size: int = 112, max_samples: int = None):
        import mxnet as mx

        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        assert os.path.exists(path_imgrec), f"Missing {path_imgrec}"
        assert os.path.exists(path_imgidx), f"Missing {path_imgidx}"

        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # imgrec.keys gives all valid record index keys
        all_keys = list(self.imgrec.keys)
        # Truncate: empty records cluster near end of MS1MV3 (~1.8% empty)
        # Use the first N keys to avoid empty-record overhead
        if max_samples is not None and max_samples < len(all_keys):
            self.imgidx = all_keys[:max_samples]
        else:
            # Use first 5M records (avoids empty-record tail)
            safe_count = min(len(all_keys), 5_000_000)
            self.imgidx = all_keys[:safe_count]
        self.image_size = image_size

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, idx):
        import mxnet as mx

        # Retry on empty records (rare, mostly in tail of dataset)
        for attempt in range(5):
            record_key = self.imgidx[(idx + attempt * 31337) % len(self.imgidx)]
            s = self.imgrec.read_idx(record_key)
            header, img_bytes = mx.recordio.unpack(s)
            if len(img_bytes) > 0:
                break
        # Decode MXNet-encoded image (JPEG internally)
        img = mx.image.imdecode(img_bytes).asnumpy()
        img = img.astype(np.float32)
        # Normalize to [-1, 1] (standard for insightface models)
        img = img / 127.5 - 1.0
        # Transpose from HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, teacher, dataloader, optimizer, device, epoch, args, is_main=True):
    """Distillation training: MSE between teacher and student embeddings."""
    model.train()
    teacher.eval()

    # Ensure FakeQuantize modules are in train mode (observers active)
    raw = model.module if hasattr(model, 'module') else model
    for _, fq in get_fake_quantize_modules(raw):
        fq.train()

    total_loss = 0.0
    total_samples = 0
    start = time.time()

    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)

        with torch.no_grad():
            teacher_emb = teacher(images)

        student_emb = model(images)
        loss = nn.functional.mse_loss(student_emb, teacher_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % 50 == 0 and is_main:
            elapsed = time.time() - start
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            print(f"  Epoch {epoch} [{batch_idx:5d}/{len(dataloader)}] "
                  f"loss={loss.item():.6f}  avg={total_loss/total_samples:.6f}  "
                  f"{samples_per_sec:.0f} img/s")

    avg_loss = total_loss / total_samples
    if is_main:
        print(f"Epoch {epoch} complete: avg_loss={avg_loss:.6f}  ({time.time()-start:.0f}s)")
    return avg_loss


def calibrate_observers(model, dataloader, device, num_batches=10):
    """Run a few forward passes in train mode to calibrate FakeQuantize observers."""
    model.train()
    for _, fq in get_fake_quantize_modules(model):
        fq.train()
    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    print(f"  Calibrated observers on {num_batches} batches")


def compute_cosine_fidelity(model, teacher, dataloader, device, max_batches=50):
    """Compute cosine similarity between teacher and student embeddings.

    Model is in eval mode (FakeQuantize uses calibrated ranges but no
    observer updates), so this represents the actual inference-time quantization
    effect.
    """
    model.eval()
    teacher.eval()
    # Keep FQ in eval mode so they quantize but don't update observers
    for _, fq in get_fake_quantize_modules(model):
        fq.eval()

    cosines = []
    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            t_emb = teacher(images)
            s_emb = model(images)
            # Cosine similarity per sample
            cos = nn.functional.cosine_similarity(t_emb, s_emb, dim=1)
            cosines.extend(cos.cpu().tolist())

    cos = np.array(cosines)
    return float(cos.mean()), float(cos.std())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='QAT fine-tuning for MobileFaceNet INT8')
    parser.add_argument('--model', type=str, required=True, help='Path to FP32 model.pt')
    parser.add_argument('--output', type=str, default='/workspace/output/qat_finetuned',
                        help='Output directory')
    parser.add_argument('--data', type=str, default=os.environ.get('DATA_DIR', '/data/ms1m-retinaface-t1'),
                        help='MS1MV3 data directory')
    parser.add_argument('--epochs', type=int, default=2, help='QAT fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (for debugging)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    # ---- DDP init ----
    ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if ddp:
        rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0; local_rank = 0; world_size = 1
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    def mprint(*a, **k):
        if is_main: print(*a, **k)

    if is_main:
        os.makedirs(args.output, exist_ok=True)
    mprint(f"Device: {device} (rank {rank}/{world_size})")
    mprint(f"Output: {args.output}")

    # ---- Build model and load checkpoint ----
    mprint("\n=== Building model ===")
    model = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    n_params = sum(p.numel() for p in model.parameters())
    mprint(f"Model: {n_params:,} params ({n_params/1e6:.2f}M)")

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        if 'backbone' in ckpt:
            sd = ckpt['backbone']
        elif 'state_dict_backbone' in ckpt:
            sd = ckpt['state_dict_backbone']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=True)
    mprint(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    # ---- Create teacher (FP32 reference, frozen, one per rank) ----
    teacher = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    teacher.load_state_dict(sd, strict=True)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    mprint("Teacher: frozen FP32 reference")

    # ---- Insert FakeQuantize into student model ----
    mprint("\n=== Inserting FakeQuantize modules ===")
    model = insert_activation_fake_quantize(model)
    freeze_batchnorm(model)
    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    raw_model = model.module if ddp else model
    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in raw_model.parameters())
    fq_modules = get_fake_quantize_modules(raw_model)
    mprint(f"FakeQuantize modules: {len(fq_modules)}")
    mprint(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ---- Data loading ----
    mprint(f"\n=== Loading data from {args.data} ===")
    dataset = MXRecordDataset(args.data, max_samples=args.max_samples)
    mprint(f"Dataset: {len(dataset)} samples")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True) if ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Observer calibration (rank 0 only, then broadcast not needed: DDP syncs params later) ----
    mprint("\n=== Pre-QAT observer calibration ===")
    calibrate_observers(raw_model, dataloader, device, num_batches=20)

    if is_main:
        print("\n=== Pre-QAT cosine fidelity (after observer calibration) ===")
        cos_mean, cos_std = compute_cosine_fidelity(raw_model, teacher, dataloader, device)
        print(f"Cosine similarity (student vs teacher, with per-tensor INT8 simulation): {cos_mean:.4f} +/- {cos_std:.4f}")
    if ddp:
        dist.barrier()

    # ---- Fine-tuning loop ----
    mprint(f"\n=== QAT fine-tuning: {args.epochs} epochs ===")
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss = train_one_epoch(model, teacher, dataloader, optimizer, device, epoch, args, is_main=is_main)
        scheduler.step()

        if is_main:
            ckpt_path = os.path.join(args.output, f"model_qat_e{epoch}.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")
            cos_mean, cos_std = compute_cosine_fidelity(raw_model, teacher, dataloader, device)
            print(f"  Cosine fidelity: {cos_mean:.4f} +/- {cos_std:.4f}")
        if ddp:
            dist.barrier()

    # ---- Final save (rank 0 only) ----
    if is_main:
        final_pt = os.path.join(args.output, "model_qat.pt")
        torch.save(raw_model.state_dict(), final_pt)
        print(f"\nFinal model: {final_pt}")

    # ---- ONNX Export (rank 0 only) ----
    if is_main:
        print("\n=== ONNX export (clean model with QAT-tuned weights) ===")
        model_clean = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
        model_clean = model_clean.to(device)
        model_clean.eval()

        # Transfer params AND buffers (BN running_mean/running_var) via state_dict.
        # Bug fix: using named_parameters() misses BN buffers, leaving clean model
        # with default mean=0/var=1 and crashing inference (xnorm balloons, LFW->50%).
        clean_keys = set(model_clean.state_dict().keys())
        sd_clean = {k.replace('.block.', '.'): v for k, v in raw_model.state_dict().items()
                    if k.replace('.block.', '.') in clean_keys}
        miss, unexp = model_clean.load_state_dict(sd_clean, strict=True)
        print(f"  Weight transfer: {len(sd_clean)}/{len(clean_keys)} keys matched "
              f"(missing={len(miss)} unexpected={len(unexp)})")

        onnx_path = os.path.join(args.output, "model_qat_clean.onnx")
        dummy_input = torch.randn(1, 3, 112, 112, device=device)
        torch.onnx.export(
            model_clean, dummy_input, onnx_path,
            input_names=['input'], output_names=['embedding'], opset_version=11,
            dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
        )
        print(f"  ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f} MB)")

        cos_mean, cos_std = compute_cosine_fidelity(raw_model, teacher, dataloader, device)
        print(f"\nFinal cosine fidelity (student vs teacher FP32): {cos_mean:.4f} +/- {cos_std:.4f}")
        print("\n=== QAT fine-tuning complete ===")
        print(f"Output: {args.output}")

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
