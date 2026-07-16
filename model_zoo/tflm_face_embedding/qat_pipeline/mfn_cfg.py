#!/usr/bin/env python3
"""Configurable MobileFaceNet (activation + embedding dim) with FakeQuantize
insertion, including an embedding-output FQ. Used by train_128d.py / export_128d.py.

Mirrors backbones.py / qat_finetune.py MobileFaceNet exactly (same layer names, so
512D checkpoints load into the shared backbone), but:
  * activation is configurable ('leaky' | 'relu6' | 'relu')
  * num_features configurable (128 for the drop-in)
  * an FQ (embq) after the final embedding so the 128D projection output is
    quantization-aware (this is what the old post-hoc PCA-128 lacked).
"""
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.quantization import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver


def make_act(kind):
    if kind == "leaky":
        return nn.LeakyReLU(0.01, inplace=True)
    if kind == "relu6":
        return nn.ReLU6(inplace=True)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(kind)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, act="leaky"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            make_act(act),
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
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, act="leaky"):
        super().__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1), act=act),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride, act=act),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
        )

    def forward(self, x):
        short_cut = x if self.residual else None
        x = self.layers(x)
        if self.residual:
            x = short_cut + x
        return x


class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="leaky"):
        super().__init__()
        modules = [DepthWise(c, c, True, kernel, stride, padding, groups, act=act) for _ in range(num_block)]
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(nn.Module):
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
    def __init__(self, num_features=128, blocks=(1, 4, 6, 2), scale=1, act="leaky"):
        super().__init__()
        self.scale = scale
        self.act = act
        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(3, 64 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), act=act))
        if blocks[0] == 1:
            self.layers.append(ConvBlock(64 * scale, 64 * scale, kernel=(3, 3), stride=(1, 1),
                                         padding=(1, 1), groups=64 * scale, act=act))
        else:
            self.layers.append(Residual(64 * scale, num_block=blocks[0], groups=128, kernel=(3, 3),
                                        stride=(1, 1), padding=(1, 1), act=act))
        self.layers.extend([
            DepthWise(64 * scale, 64 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, act=act),
            Residual(64 * scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act=act),
            DepthWise(64 * scale, 128 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, act=act),
            Residual(128 * scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act=act),
            DepthWise(128 * scale, 128 * scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, act=act),
            Residual(128 * scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act=act),
        ])
        self.conv_sep = ConvBlock(128 * scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0), act=act)
        self.features = GDC(num_features)
        # Embedding-output FQ (Identity until QAT enables it). Quantization-aware 128D output.
        self.embq = nn.Identity()

    def forward(self, x):
        for func in self.layers:
            x = func(x)
        x = self.conv_sep(x)
        x = self.features(x)
        x = self.embq(x)
        return x


def make_fake_quantize():
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8, qscheme=torch.per_tensor_affine, reduce_range=False,
    )


def insert_activation_fake_quantize(model):
    """Wrap every ConvBlock/LinearBlock output in Sequential(block, fq); also enable
    the embedding-output FQ (model.embq)."""
    replacements = []
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child in list(parent_module.named_children()):
            if isinstance(child, (ConvBlock, LinearBlock)):
                fq = make_fake_quantize()
                wrapper = nn.Sequential(OrderedDict([('block', child), ('fq', fq)]))
                replacements.append((parent_module, child_name, wrapper))
    for parent, attr, wrapper in replacements:
        setattr(parent, attr, wrapper)
    # embedding output FQ
    model.embq = make_fake_quantize()
    print(f"Inserted {len(replacements)} block-FQ + 1 embedding-FQ")
    return model


def get_fake_quantize_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, FakeQuantize)]


def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
