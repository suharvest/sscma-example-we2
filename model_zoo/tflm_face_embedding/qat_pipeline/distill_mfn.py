#!/usr/bin/env python3
"""
Knowledge Distillation: ResNet100 (teacher, ONNX Runtime) -> MobileFaceNet (student, PyTorch).

Teacher: glint360k_r100.onnx (InsightFace ResNet100, 65M params, 512D output, LFW 99.8%)
Student: MobileFaceNet scale=1, blocks=(1,4,6,2), 512D, LeakyReLU (~995K params)

Loss = ArcFace(student_class_logits, labels) + lambda * MSE(student_emb, teacher_emb)

The teacher ONNX model outputs a 512D L2-normalized embedding. We distill by:
  1. ArcFace margin-based classification loss on the student's class logits
  2. Embedding-level MSE distillation between teacher and student 512D vectors

Usage (on vast.ai / GPU instance):
    source /workspace/venv/bin/activate
    export DATA_DIR=/data/ms1m-retinaface-t1
    python distill_mfn.py \
        --teacher /workspace/glint360k_r100.onnx \
        --student /path/to/model.pt \
        --output /workspace/output/distill \
        --epochs 4 \
        --lr 0.01 \
        --lambda-distill 0.5 \
        --batch-size 256

Output:
    model_distill_e{N}.pt  — epoch checkpoints
    model_distill_best.pt  — best cosine fidelity checkpoint
    model_distill.onnx     — FP32 ONNX export
"""

import argparse
import os
import sys
import time
import math
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Model definition (MobileFaceNet, matches backbones.py + qat_finetune.py)
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
# ArcFace Loss (margin-based, from insightface)
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace).

    s: scale (default 64.0)
    m: margin in radians (default 0.5)
    """
    def __init__(self, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, label):
        # cosine: normalized logits [B, num_classes]
        # label: [B]
        sine = torch.sqrt(1.0 - cosine ** 2 + 1e-8)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(label, num_classes=cosine.size(1)).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.s
        return F.cross_entropy(output, label)


# ---------------------------------------------------------------------------
# Classification head (512D -> num_classes, with L2 normalization for ArcFace)
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """L2-normalized weights + bias-less Linear, designed for ArcFace input."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        # We apply L2 norm to both features and weights in forward

    def forward(self, features):
        # L2-normalize features and weights for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(features_norm, weight_norm)  # [B, num_classes]
        return cosine


# ---------------------------------------------------------------------------
# Student model with ArcFace head
# ---------------------------------------------------------------------------

class StudentWithHead(nn.Module):
    """MobileFaceNet backbone + ArcFace classification head."""
    def __init__(self, backbone, num_classes=93431):
        super().__init__()
        self.backbone = backbone
        self.head = ArcFaceHead(512, num_classes)

    def forward(self, x):
        embedding = self.backbone(x)  # [B, 512]
        logits = self.head(embedding)  # [B, num_classes]
        return embedding, logits


# ---------------------------------------------------------------------------
# Data loader using MXNet RecordIO
# ---------------------------------------------------------------------------

class MXRecordDataset(Dataset):
    """Load aligned face images from MXNet RecordIO files.

    Uses the same .rec format as insightface/arcface_torch.
    """

    def __init__(self, root_dir: str, image_size: int = 112, max_samples: int = None):
        import mxnet as mx

        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        assert os.path.exists(path_imgrec), f"Missing {path_imgrec}"
        assert os.path.exists(path_imgidx), f"Missing {path_imgidx}"

        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        all_keys = list(self.imgrec.keys)

        if max_samples is not None and max_samples < len(all_keys):
            self.imgidx = all_keys[:max_samples]
        else:
            safe_count = min(len(all_keys), 5_000_000)
            self.imgidx = all_keys[:safe_count]

        self.image_size = image_size

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, idx):
        import mxnet as mx

        for attempt in range(5):
            record_key = self.imgidx[(idx + attempt * 31337) % len(self.imgidx)]
            s = self.imgrec.read_idx(record_key)
            header, img_bytes = mx.recordio.unpack(s)
            if len(img_bytes) > 0:
                break

        img = mx.image.imdecode(img_bytes).asnumpy()
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        # Label is embedded in the header
        label = int(header.label[0]) if hasattr(header, 'label') and len(header.label) > 0 else 0
        return torch.from_numpy(img), label


class DistillationDataset(MXRecordDataset):
    """MXRecordDataset that also provides labels for ArcFace loss."""
    pass  # __getitem__ already returns (img, label)


# ---------------------------------------------------------------------------
# Teacher wrapper (ONNX Runtime)
# ---------------------------------------------------------------------------

class TeacherONNX:
    """ResNet100 ONNX teacher, frozen. Outputs 512D L2-normalized embeddings."""

    def __init__(self, onnx_path: str, device: str = 'cuda', device_id: int = 0):
        import onnxruntime as ort

        # ONNX Runtime providers: prefer CUDA on this rank's GPU
        providers = [
            ('CUDAExecutionProvider', {'device_id': device_id}),
            'CPUExecutionProvider',
        ]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = device

        # Verify model
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        print(f"Teacher ONNX input:  {input_shape}")
        print(f"Teacher ONNX output: {output_shape}")
        print(f"Teacher providers:   {self.session.get_providers()}")

    @torch.no_grad()
    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Run ONNX inference and return 512D embeddings as PyTorch tensor."""
        # images: [B, 3, 112, 112] float32 in [-1, 1]
        # ONNX model expects [-1, 1] input and outputs 512D normalized embedding
        inputs = images.cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: inputs})
        embeddings = torch.from_numpy(outputs[0]).to(self.device)
        return embeddings


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_embedding_cosine(model, teacher, dataloader, device, max_batches=50):
    """Compute mean cosine similarity between teacher and student embeddings."""
    model.eval()
    cosines = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            t_emb = teacher.get_embeddings(images)
            s_emb, _ = model(images)  # embedding only
            cos = F.cosine_similarity(t_emb, s_emb, dim=1)
            cosines.extend(cos.cpu().tolist())

    cos = np.array(cosines)
    return float(cos.mean()), float(cos.std())


def train_one_epoch(model, teacher, dataloader, optimizer, scaler, device,
                    epoch, args, arcface_loss_fn, is_main=True):
    """One epoch of distillation training."""
    model.train()
    total_arcface = 0.0
    total_distill = 0.0
    total_loss = 0.0
    total_samples = 0
    start = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Teacher inference (frozen, no grad)
        with torch.no_grad():
            teacher_emb = teacher.get_embeddings(images)

        # Student forward
        with autocast():
            student_emb, student_logits = model(images)
            arcface_loss = arcface_loss_fn(student_logits, labels)
            # Scale-invariant distillation: 1 - cos(student, teacher).
            # Teacher is L2-normalized; student backbone is not. Cosine sim handles this.
            distill_loss = 1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=1).mean()
            loss = arcface_loss + args.lambda_distill * distill_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_arcface += arcface_loss.item() * images.size(0)
        total_distill += distill_loss.item() * images.size(0)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % args.log_interval == 0 and is_main:
            elapsed = time.time() - start
            img_per_sec = total_samples / elapsed if elapsed > 0 else 0
            print(f"  Epoch {epoch} [{batch_idx:5d}/{len(dataloader)}] "
                  f"L={loss.item():.4f} AF={arcface_loss.item():.4f} "
                  f"DL={distill_loss.item():.6f}  {img_per_sec:.0f} img/s")

    avg_arcface = total_arcface / total_samples
    avg_distill = total_distill / total_samples
    avg_loss = total_loss / total_samples
    elapsed = time.time() - start
    if is_main:
        print(f"Epoch {epoch} complete: loss={avg_loss:.4f} "
              f"arcface={avg_arcface:.4f} distill={avg_distill:.6f}  "
              f"({elapsed:.0f}s, {total_samples/elapsed:.0f} img/s)")
    return avg_loss, avg_arcface, avg_distill


def save_checkpoint(model, optimizer, scaler, epoch, path):
    """Save training state."""
    state = {
        'backbone': model.backbone.state_dict(),
        'head': model.head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path} ({size_mb:.1f} MB)")


def load_checkpoint(model, checkpoint_path, device):
    """Load backbone weights from various checkpoint formats."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        # insightface arcface_torch format: {'state_dict_backbone': ..., 'head': ...}
        if 'backbone' in ckpt:
            sd = ckpt['backbone']
        elif 'state_dict_backbone' in ckpt:
            sd = ckpt['state_dict_backbone']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            # Try to use the dict directly as state_dict
            sd = {k: v for k, v in ckpt.items()}
    else:
        sd = ckpt

    # Strip 'module.' prefix from DDP training
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0 and len(missing) < 20:
        print(f"  Missing keys: {missing}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation: ResNet100 (ONNX) -> MobileFaceNet (PyTorch)')

    # Required
    parser.add_argument('--teacher', type=str, required=True,
                        help='Path to glint360k_r100.onnx (InsightFace ResNet100)')
    parser.add_argument('--output', type=str, default='/workspace/output/distill',
                        help='Output directory')

    # Optional: student initialization
    parser.add_argument('--student', type=str, default=None,
                        help='Path to initial student model.pt (optional, random init if not given)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to distillation checkpoint to resume from')

    # Data
    parser.add_argument('--data', type=str,
                        default=os.environ.get('DATA_DIR', '/data/ms1m-retinaface-t1'),
                        help='MS1MV3 data directory')

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=4, help='Distillation epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--lambda-distill', type=float, default=5.0,
                        help='Weight of distillation loss (1-cos on embeddings)')
    parser.add_argument('--eval-bins', type=str, nargs='*',
                        default=['/data/ms1m-retinaface-t1/lfw.bin',
                                 '/data/ms1m-retinaface-t1/cfp_fp.bin'],
                        help='LFW-style .bin paths for per-epoch verification (rank 0 only)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU?')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (for debugging)')

    # ArcFace
    parser.add_argument('--arcface-s', type=float, default=64.0, help='ArcFace scale')
    parser.add_argument('--arcface-m', type=float, default=0.5, help='ArcFace margin')
    parser.add_argument('--num-classes', type=int, default=93431,
                        help='Number of identities in training set')

    # Misc
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--log-interval', type=int, default=50, help='Log every N batches')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='Disable mixed precision')

    args = parser.parse_args()

    # ---- DDP init ----
    ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if ddp:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    is_main = (rank == 0)
    def mprint(*a, **k):
        if is_main: print(*a, **k)

    if is_main:
        os.makedirs(args.output, exist_ok=True)
    mprint(f"Device: {device} (rank {rank}/{world_size})")
    mprint(f"Output: {args.output}")

    # ---- Teacher (ONNX Runtime, frozen) — one per rank, pinned to local GPU ----
    mprint("\n=== Loading teacher (ResNet100 ONNX) ===")
    teacher = TeacherONNX(args.teacher, device=str(device), device_id=local_rank)

    # ---- Student (MobileFaceNet + ArcFace head) ----
    mprint("\n=== Building student (MobileFaceNet scale=1, 512D) ===")
    backbone = MobileFaceNet(num_features=512, blocks=(1, 4, 6, 2), scale=1)
    n_params = sum(p.numel() for p in backbone.parameters())
    mprint(f"Backbone: {n_params:,} params ({n_params/1e6:.2f}M)")

    model = StudentWithHead(backbone, num_classes=args.num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mprint(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    mprint(f"Trainable:    {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Load initial student weights if provided
    if args.student and os.path.exists(args.student):
        mprint(f"\n=== Loading initial student weights: {args.student} ===")
        load_checkpoint(backbone, args.student, device)
    elif args.student:
        mprint(f"WARNING: --student path does not exist: {args.student}")
        mprint("Starting from random initialization.")

    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ---- ArcFace loss ----
    arcface_loss_fn = ArcFaceLoss(s=args.arcface_s, m=args.arcface_m)

    # ---- Data loading ----
    mprint(f"\n=== Loading data from {args.data} ===")
    dataset = MXRecordDataset(args.data, max_samples=args.max_samples)
    mprint(f"Dataset: {len(dataset):,} samples")

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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Mixed precision ----
    scaler = GradScaler() if args.fp16 and device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision (AMP)")

    # Helper: get unwrapped student (strip DDP)
    raw_model = model.module if ddp else model

    # ---- Load LFW-style verification bins (rank 0 only) ----
    eval_sets = []
    if is_main:
        sys.path.insert(0, '/workspace/insightface/recognition/arcface_torch')
        try:
            from eval.verification import load_bin, test as vtest
            for bp in (args.eval_bins or []):
                if os.path.exists(bp):
                    print(f"Loading eval bin: {bp}")
                    ds = load_bin(bp, (112, 112))
                    if ds is not None:
                        eval_sets.append((os.path.basename(bp).replace('.bin',''), ds))
                else:
                    print(f"Skip missing eval bin: {bp}")
        except Exception as e:
            print(f"Eval setup failed (will skip): {e}")

    def eval_epoch(epoch):
        if not is_main or not eval_sets:
            return
        backbone_module = raw_model.backbone
        backbone_module.eval()
        for name, ds in eval_sets:
            try:
                acc1, std1, acc2, std2, _xn, _emb = vtest(ds, backbone_module, batch_size=128, nfolds=10)
                print(f"  [eval e{epoch} {name}] flip={acc2*100:.2f}% (+/-{std2*100:.2f})  no-flip={acc1*100:.2f}%")
            except Exception as e:
                print(f"  [eval e{epoch} {name}] FAIL: {e}")
        backbone_module.train()

    # ---- Resume from checkpoint ----
    start_epoch = 0
    if args.resume:
        mprint(f"\n=== Resuming from {args.resume} ===")
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        raw_model.backbone.load_state_dict(state['backbone'])
        raw_model.head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state.get('epoch', 0)
        if scaler is not None and 'scaler' in state:
            scaler.load_state_dict(state['scaler'])
        mprint(f"Resumed from epoch {start_epoch}")

    # ---- Pre-distillation cosine fidelity (rank 0 only) ----
    if is_main:
        print("\n=== Pre-distillation embedding cosine fidelity ===")
        cos_mean, cos_std = compute_embedding_cosine(raw_model, teacher, dataloader, device)
        print(f"Teacher-student cosine similarity: {cos_mean:.4f} +/- {cos_std:.4f}")
        print(f"(random init ~0.0-0.1;  good distillation >0.7)")
    if ddp:
        dist.barrier()

    # ---- Distillation loop ----
    mprint(f"\n=== Knowledge Distillation: {args.epochs} epochs ===")
    mprint(f"Loss = ArcFace + {args.lambda_distill} * MSE(emb_student, emb_teacher)")
    mprint(f"ArcFace: s={args.arcface_s}, m={args.arcface_m}")
    mprint(f"LR: {args.lr}, Batch: {args.batch_size}/GPU x{world_size}, Classes: {args.num_classes}")

    best_cosine = -1.0
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss, arc_loss, dist_loss = train_one_epoch(
            model, teacher, dataloader, optimizer, scaler,
            device, epoch, args, arcface_loss_fn, is_main=is_main,
        )
        scheduler.step()

        # Save epoch checkpoint + cosine fidelity + LFW eval (rank 0 only)
        if is_main:
            ckpt_path = os.path.join(args.output, f"model_distill_e{epoch}.pt")
            save_checkpoint(raw_model, optimizer, scaler, epoch, ckpt_path)
            cos_mean, cos_std = compute_embedding_cosine(raw_model, teacher, dataloader, device)
            print(f"  Cosine fidelity: {cos_mean:.4f} +/- {cos_std:.4f}")
            eval_epoch(epoch)
            if cos_mean > best_cosine:
                best_cosine = cos_mean
                best_path = os.path.join(args.output, "model_distill_best.pt")
                save_checkpoint(raw_model, optimizer, scaler, epoch, best_path)
                print(f"  New best cosine fidelity: {cos_mean:.4f}")
        if ddp:
            dist.barrier()

    # ---- Final checkpoint + ONNX (rank 0 only) ----
    if is_main:
        final_path = os.path.join(args.output, "model_distill_final.pt")
        save_checkpoint(raw_model, optimizer, scaler, args.epochs, final_path)

        print("\n=== ONNX export (backbone only) ===")
        backbone.eval()
        onnx_path = os.path.join(args.output, "model_distill.onnx")
        dummy_input = torch.randn(1, 3, 112, 112, device=device)
        torch.onnx.export(
            backbone, dummy_input, onnx_path,
            input_names=['input'],
            output_names=['embedding'],
            opset_version=11,
            dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
        )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

        print("\n=== Final evaluation ===")
        cos_mean, cos_std = compute_embedding_cosine(raw_model, teacher, dataloader, device)
        print(f"Final cosine fidelity (student vs teacher): {cos_mean:.4f} +/- {cos_std:.4f}")
        print(f"Best cosine fidelity: {best_cosine:.4f}")

        print(f"\n=== Distillation complete ===")
        print(f"Output directory: {args.output}")
        for f in sorted(Path(args.output).glob("*")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")

    if ddp:
        dist.destroy_process_group()
    print(f"\nNext step: QAT fine-tuning")
    print(f"  python qat_finetune.py --model {final_path} --output /workspace/output/qat")


if __name__ == '__main__':
    main()
