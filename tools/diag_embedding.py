#!/usr/bin/env python3
"""
Diagnostic tool: capture [DIAG] output from face embedding firmware.

Connects to Himax serial, enables face mode, starts inference,
and collects diagnostic lines to analyze whether embeddings change between frames.
"""

import serial
import serial.tools.list_ports
import sys
import time
import re


def find_himax_port():
    """Find the Himax USB serial port."""
    for p in serial.tools.list_ports.comports():
        if "usbmodem" in p.device:
            return p.device
    return None


def send_command(ser, cmd, wait=0.5):
    """Send AT command and collect response lines."""
    ser.write((cmd + "\r\n").encode())
    time.sleep(wait)
    lines = []
    while ser.in_waiting:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            lines.append(line)
    return lines


def main():
    port = find_himax_port()
    if not port:
        print("ERROR: No Himax USB serial port found")
        sys.exit(1)

    print(f"Connecting to {port} @ 921600...")
    ser = serial.Serial(port, 921600, timeout=1)
    time.sleep(0.5)

    # Drain any pending data
    ser.reset_input_buffer()

    # Enable face mode
    print("\n--- Enabling face mode ---")
    resp = send_command(ser, "AT+FACE=1", wait=1.0)
    for line in resp:
        print(f"  {line}")

    # Start continuous inference
    print("\n--- Starting inference (10 frames) ---")
    send_command(ser, "AT+INVOKE=-1,0,1", wait=0.5)

    # Collect diagnostic output
    diag_lines = []
    input_sums = []
    before_vals = []
    after_vals = []
    frame_count = 0
    max_frames = 10
    start_time = time.time()
    timeout = 30  # seconds

    while frame_count < max_frames and (time.time() - start_time) < timeout:
        try:
            if not ser.is_open:
                print("Serial port closed unexpectedly!")
                break
            if ser.in_waiting:
                line = ser.readline().decode(errors="replace").strip()
                if not line:
                    continue

                if "[DIAG]" in line:
                    diag_lines.append(line)
                    print(f"  {line}")

                # Parse input sum
                m = re.search(r"EMB input: sum=(-?\d+)", line)
                if m:
                    input_sums.append(int(m.group(1)))

                # Parse BEFORE cache inv
                m = re.search(r"BEFORE cache inv: \[([^\]]+)\]", line)
                if m:
                    vals = [int(x) for x in m.group(1).split(",")]
                    before_vals.append(vals)

                # Parse AFTER cache inv
                m = re.search(r"AFTER  cache inv: \[([^\]]+)\]", line)
                if m:
                    vals = [int(x) for x in m.group(1).split(",")]
                    after_vals.append(vals)
                    frame_count += 1  # AFTER is the last diag line per frame
        except OSError as e:
            print(f"Serial error: {e}")
            break
        else:
            time.sleep(0.05)

    # Stop inference
    print("\n--- Stopping inference ---")
    send_command(ser, "AT+BREAK", wait=0.5)
    ser.close()

    # Analysis
    print(f"\n{'='*60}")
    print(f"ANALYSIS ({frame_count} frames captured)")
    print(f"{'='*60}")

    if len(input_sums) >= 2:
        unique_sums = len(set(input_sums))
        print(f"\nInput tensor sums: {input_sums}")
        print(f"  Unique values: {unique_sums}/{len(input_sums)}")
        if unique_sums == 1:
            print("  ** WARNING: Input is IDENTICAL every frame! Problem is BEFORE embedding model. **")
        else:
            print("  OK: Input varies between frames.")

    if len(before_vals) >= 2 and len(after_vals) >= 2:
        before_same = all(v == before_vals[0] for v in before_vals)
        after_same = all(v == after_vals[0] for v in after_vals)

        print(f"\nBEFORE cache invalidation:")
        for i, v in enumerate(before_vals):
            print(f"  Frame {i}: {v}")
        print(f"  All identical: {before_same}")

        print(f"\nAFTER cache invalidation:")
        for i, v in enumerate(after_vals):
            print(f"  Frame {i}: {v}")
        print(f"  All identical: {after_same}")

        # Check if BEFORE and AFTER differ within any frame
        cache_matters = False
        for i in range(min(len(before_vals), len(after_vals))):
            if before_vals[i] != after_vals[i]:
                cache_matters = True
                break

        print(f"\n{'='*60}")
        print("DIAGNOSIS:")
        if cache_matters:
            print("  D-CACHE IS THE ROOT CAUSE!")
            print("  BEFORE and AFTER cache invalidation give different values.")
            print("  Without invalidation, CPU reads stale cached data.")
        elif before_same and after_same:
            print("  Output is identical every frame (even after cache inv).")
            if len(input_sums) >= 2 and len(set(input_sums)) > 1:
                print("  Input changes but output doesn't -> MODEL may not be running properly.")
            else:
                print("  Input is also identical -> problem is upstream (alignment/crop).")
        elif not before_same and not after_same:
            print("  Output changes between frames both before and after cache inv.")
            print("  D-Cache may NOT be the issue. Embeddings should be different.")
            print("  Check if the DEQUANTIZED embeddings are too similar (scale/zero_point issue?).")
        print(f"{'='*60}")
    else:
        print("\nNot enough data captured. Is a face visible to the camera?")
        if diag_lines:
            print("Raw diagnostic lines:")
            for l in diag_lines:
                print(f"  {l}")


if __name__ == "__main__":
    main()
