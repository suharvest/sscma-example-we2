#!/usr/bin/env python3
"""Capture the last face embedding input/output tensors from SSCMA face firmware."""

import argparse
import json
import sys
import time
from pathlib import Path

import serial
import serial.tools.list_ports


def find_port() -> str | None:
    for p in serial.tools.list_ports.comports():
        if "usbmodem" in p.device:
            return p.device
    return None


def read_available_lines(ser: serial.Serial, timeout: float) -> list[str]:
    end = time.time() + timeout
    buf = bytearray()
    lines: list[str] = []
    while time.time() < end:
        waiting = ser.in_waiting
        if waiting:
            buf.extend(ser.read(waiting))
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    lines.append(text)
        else:
            time.sleep(0.02)
    if buf:
        text = buf.decode("utf-8", errors="replace").strip()
        if text:
            lines.append(text)
    return lines


def send_at(ser: serial.Serial, cmd: str, wait: float = 0.5) -> list[str]:
    ser.write((cmd + "\r\n").encode("ascii"))
    return read_available_lines(ser, wait)


def find_json_line(lines: list[str], name: str | None = None) -> dict | None:
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        if name and f'"name":"{name}"' not in line and f'"name": "{name}"' not in line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--out", default="/tmp/himax_facedbg.json")
    parser.add_argument("--sensor", default="AT+SENSOR=1,1,0")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--invoke-wait", type=float, default=8.0)
    parser.add_argument("--debug-wait", type=float, default=10.0)
    parser.add_argument("--conf-milli", type=int, default=None)
    parser.add_argument(
        "--fixed-test-seed",
        type=int,
        default=None,
        help="Run FACEEMBTEST=SEED and dump tensors without camera/SCRFD",
    )
    parser.add_argument(
        "--flash-test",
        action="store_true",
        help="Run FACEEMBFLASH and dump tensors without camera/SCRFD",
    )
    parser.add_argument(
        "--flash-offset",
        default=None,
        help="Optional FACEEMBFLASH QSPI offset, for example 0x700000",
    )
    args = parser.parse_args()

    port = args.port or find_port()
    if not port:
        print("ERROR: no /dev/cu.usbmodem* serial port found", file=sys.stderr)
        return 2

    print(f"Connecting {port} @ {args.baud}", flush=True)
    with serial.Serial(port, args.baud, timeout=0.2, write_timeout=2) as ser:
        time.sleep(1.0)
        ser.reset_input_buffer()

        setup_cmds = [("AT+BREAK", 0.8)]
        if args.fixed_test_seed is None and not args.flash_test:
            setup_cmds.extend(
                [
                    (args.sensor, 1.0),
                    (f"AT+FACECFG={args.conf_milli}", 0.5) if args.conf_milli is not None else (None, 0),
                    ("AT+FACE=1", 1.5),
                ]
            )

        for cmd, wait in setup_cmds:
            if cmd is None:
                continue
            print(f"> {cmd}", flush=True)
            for line in send_at(ser, cmd, wait):
                print(f"  {line[:220]}", flush=True)

        face_json = None
        if args.flash_test:
            cmd = "AT+FACEEMBFLASH"
            if args.flash_offset:
                cmd += f"={args.flash_offset}"
            print(f"> {cmd}", flush=True)
            test_lines = send_at(ser, cmd, args.invoke_wait)
            face_json = find_json_line(test_lines, "FACEEMBFLASH")
            if not face_json or face_json.get("code") != 0:
                print("ERROR: FACEEMBFLASH failed or did not respond", file=sys.stderr)
                for line in test_lines[-20:]:
                    print(f"  {line[:300]}", file=sys.stderr)
                return 5
        elif args.fixed_test_seed is not None:
            cmd = f"AT+FACEEMBTEST={args.fixed_test_seed}"
            print(f"> {cmd}", flush=True)
            test_lines = send_at(ser, cmd, args.invoke_wait)
            face_json = find_json_line(test_lines, "FACEEMBTEST")
            if not face_json or face_json.get("code") != 0:
                print("ERROR: FACEEMBTEST failed or did not respond", file=sys.stderr)
                for line in test_lines[-20:]:
                    print(f"  {line[:300]}", file=sys.stderr)
                return 4
        else:
            for attempt in range(1, args.attempts + 1):
                print(f"> AT+INVOKE={args.frames},0,1 ({attempt}/{args.attempts})", flush=True)
                invoke_lines = send_at(ser, f"AT+INVOKE={args.frames},0,1", args.invoke_wait)
                candidates = []
                for line in invoke_lines:
                    if not line.startswith("{") or "INVOKE" not in line:
                        continue
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                face_json = candidates[-1] if candidates else find_json_line(invoke_lines, "INVOKE")
                if face_json:
                    data = face_json.get("data", {})
                    faces = data.get("faces", [])
                    print(f"  invoke faces={len(faces)}", flush=True)
                    if faces:
                        first = faces[0]
                        print(f"  first score={first.get('score')} box={first.get('box')}", flush=True)
                        break
                else:
                    print("  no INVOKE json received", flush=True)
                time.sleep(0.2)

        print("> AT+FACEDBG?", flush=True)
        ser.write(b"AT+FACEDBG?\r\n")
        debug_lines = read_available_lines(ser, args.debug_wait)
        debug_json = find_json_line(debug_lines, "FACEDBG?")
        if not debug_json:
            print("ERROR: no FACEDBG JSON received", file=sys.stderr)
            for line in debug_lines[-20:]:
                print(f"  {line[:300]}", file=sys.stderr)
            return 3

    out = Path(args.out)
    out.write_text(
        json.dumps({"invoke": face_json, "facedbg": debug_json, "fixed_test_seed": args.fixed_test_seed}, indent=2),
        encoding="utf-8",
    )
    data = debug_json.get("data", {})
    emb_input = data.get("emb_input", {})
    emb_output = data.get("emb_output", {})
    print(f"Saved {out}", flush=True)
    print(f"  valid={data.get('valid')} input_bytes={emb_input.get('bytes')} output_bytes={emb_output.get('bytes')}", flush=True)
    print(f"  input_dims={emb_input.get('dims')} output_dims={emb_output.get('dims')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
