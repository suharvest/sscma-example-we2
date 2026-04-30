#!/usr/bin/env python3
"""
Capture sscma_face debug output from Himax serial port.
Sends AT+FACE=1 and AT+INVOKE, then filters for [DBG] lines.

Usage:
    uv run --with pyserial python tools/face_debug.py [--port /dev/cu.usbmodemXXX]
"""

import argparse
import glob
import sys
import time

import serial


def find_himax_port():
    """Auto-detect Himax serial port (ending in 51)."""
    candidates = glob.glob("/dev/cu.usbmodem*51")
    if candidates:
        return candidates[0]
    # fallback: any usbmodem
    candidates = glob.glob("/dev/cu.usbmodem*")
    if candidates:
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Capture sscma_face debug output")
    parser.add_argument("--port", help="Serial port path")
    parser.add_argument("--baud", type=int, default=921600)
    args = parser.parse_args()

    port = args.port or find_himax_port()
    if not port:
        print("ERROR: No Himax serial port found. Specify with --port")
        sys.exit(1)

    print(f"Connecting to {port} @ {args.baud}...")
    ser = serial.Serial(port, args.baud, timeout=1)
    time.sleep(0.5)

    # Drain any buffered data
    ser.reset_input_buffer()

    # Step 1: Send AT+BREAK to stop any running inference
    print("\n--- Sending AT+BREAK ---")
    ser.write(b"AT+BREAK\r\n")
    time.sleep(1)
    ser.reset_input_buffer()

    # Step 2: Enable face mode
    print("--- Sending AT+FACE=1 ---")
    ser.write(b"AT+FACE=1\r\n")
    time.sleep(2)

    # Read face mode response
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"  RESP: {line[:200]}")

    # Step 3: Start inference (5 frames only)
    print("\n--- Sending AT+INVOKE=5,0,1 (5 frames) ---")
    print("--- Waiting for [DBG] output... ---\n")
    ser.write(b"AT+INVOKE=5,0,1\r\n")

    # Collect output for ~15 seconds
    start = time.time()
    dbg_lines = []
    all_lines = []

    while time.time() - start < 15:
        if ser.in_waiting:
            try:
                raw = ser.readline()
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            all_lines.append(line)

            # Filter for debug lines and important messages
            if any(tag in line for tag in [
                "[DBG",
                "[DET]",
                "[NMS]",
                "[VAL]",
                "[FACE]",
                "ERROR",
                "Face Embedding",
                "Face buffer",
                "SCRFD",
                "model invalid",
                "allocation failed",
                "invoke failed",
                "NPU",
                "Ethos",
                "num_faces",
                "max_score",
                "Face too small",
                "validation failed",
                "faces detected",
                "best_face",
                "oversized",
            ]):
                dbg_lines.append(line)
                print(f"  >>> {line}")
            elif line.startswith("{") and "faces" in line:
                # JSON response with faces - show abbreviated
                if len(line) > 300:
                    print(f"  JSON: {line[:150]}...{line[-100:]}")
                else:
                    print(f"  JSON: {line}")
            elif line.startswith("{") and "INVOKE" in line:
                print(f"  RESP: {line[:200]}")

    # Send BREAK
    ser.write(b"AT+BREAK\r\n")
    time.sleep(0.5)
    ser.close()

    # Summary
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    if dbg_lines:
        for line in dbg_lines:
            print(f"  {line}")
    else:
        print("  No [DBG] or ERROR lines captured!")
        print(f"  Total lines received: {len(all_lines)}")
        if all_lines:
            print("\n  First 10 lines (raw):")
            for line in all_lines[:10]:
                print(f"    {line[:200]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
