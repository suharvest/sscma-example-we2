#!/usr/bin/env python3
"""Prepare a 112x112 RGB int8 MobileFaceNet input tensor from an image."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def quantize_rgb(img: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    normalized = img.astype(np.float32) / 127.5 - 1.0
    q = np.rint(normalized / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="/tmp/faceemb_input_int8.bin")
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--scale", type=float, default=1.0 / 127.5)
    parser.add_argument("--zero-point", type=int, default=-1)
    args = parser.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (args.size, args.size):
        rgb = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)

    tensor = quantize_rgb(rgb, args.scale, args.zero_point)
    out = Path(args.out)
    out.write_bytes(tensor.tobytes())
    print(f"wrote {out} bytes={tensor.size} shape={tensor.shape} scale={args.scale} zp={args.zero_point}")
    print(f"range=[{int(tensor.min())}, {int(tensor.max())}] mean={float(tensor.mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
