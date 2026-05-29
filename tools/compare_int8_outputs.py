#!/usr/bin/env python3
"""Compare int8 output tensors dumped by device, local TFLite, and FVP."""

import argparse
import re
from pathlib import Path

import numpy as np


def load_bin(path: str) -> np.ndarray:
    return np.frombuffer(Path(path).read_bytes(), dtype=np.int8)


def load_fvp_network_tester_output(path: str) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'output_begin.*?"data"\s*:\s*"(?P<data>.*?)"\s*\}\]\s*output_end', text, re.S)
    if not match:
        raise RuntimeError(f"cannot find network_tester output data in {path}")
    values = [int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]{1,2}", match.group("data"))]
    return np.asarray(values, dtype=np.uint8).view(np.int8)


def normalized_cosine(a: np.ndarray, b: np.ndarray, zp: int, scale: float) -> float:
    n = min(a.size, b.size)
    af = (a[:n].astype(np.float32) - zp) * scale
    bf = (b[:n].astype(np.float32) - zp) * scale
    an = float(np.linalg.norm(af))
    bn = float(np.linalg.norm(bf))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.dot(af / an, bf / bn))


def report(name: str, a: np.ndarray, b: np.ndarray, zp: int, scale: float) -> None:
    n = min(a.size, b.size)
    diff = a[:n].astype(np.int16) - b[:n].astype(np.int16)
    print(name)
    print(f"  raw_equal={bool(np.array_equal(a[:n], b[:n]))}")
    print(f"  compared={n} bytes")
    print(f"  diff_abs_max={int(np.max(np.abs(diff))) if n else 0}")
    print(f"  diff_abs_mean={float(np.mean(np.abs(diff))) if n else 0:.4f}")
    print(f"  diff_nonzero={int(np.count_nonzero(diff))}/{n}")
    print(f"  normalized_cosine={normalized_cosine(a, b, zp, scale):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-bin", required=True)
    parser.add_argument("--local-bin", required=True)
    parser.add_argument("--fvp-log", required=True)
    parser.add_argument("--fvp-bin")
    parser.add_argument("--output-zp", type=int, default=10)
    parser.add_argument("--output-scale", type=float, default=0.0393150896)
    args = parser.parse_args()

    device = load_bin(args.device_bin)
    local = load_bin(args.local_bin)
    fvp = load_fvp_network_tester_output(args.fvp_log)
    if args.fvp_bin:
        Path(args.fvp_bin).write_bytes(fvp.tobytes())

    print("Outputs:")
    print(f"  device={args.device_bin} bytes={device.size}")
    print(f"  local={args.local_bin} bytes={local.size}")
    print(f"  fvp={args.fvp_log} bytes={fvp.size}")
    report("FVP vs local", fvp, local, args.output_zp, args.output_scale)
    report("Device vs local", device, local, args.output_zp, args.output_scale)
    report("Device vs FVP", device, fvp, args.output_zp, args.output_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
