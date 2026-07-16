#!/usr/bin/env python3
"""
MobileFaceNet backbone with configurable scale and embedding size.
scale=1, blocks=(1,4,6,2) → ~1.2M params (matches esp-dl MFN).
scale=2, blocks=(1,4,6,2) → ~3.9M params (standard InsightFace mbf).

This mirrors insightface/recognition/arcface_torch/backbones/mobilefacenet.py
but adds scale and num_features configurability for the config system.
"""

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, Sequential, Module
import torch


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c),
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.residual = residual
        self.layers = Sequential(
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


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = [DepthWise(c, c, True, kernel, stride, padding, groups) for _ in range(num_block)]
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(Module):
    """Global Depthwise Conv → Flatten → Linear → BN"""
    def __init__(self, embedding_size):
        super().__init__()
        self.layers = Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            Flatten(),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(Module):
    """
    scale=1: ~1.2M params (matches esp-dl MFN)
    scale=2: ~3.9M params (standard InsightFace mbf)
    blocks=(1,4,6,2): standard MobileFaceNet depth
    """
    def __init__(self, fp16=False, num_features=128, blocks=(1, 4, 6, 2), scale=1):
        super().__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()

        # Stage 0: 112→56
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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            for func in self.layers:
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_mfn128(fp16, num_features=128, blocks=(1, 4, 6, 2), scale=1):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)
