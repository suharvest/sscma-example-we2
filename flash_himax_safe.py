#!/usr/bin/env python3
"""
Flash Himax firmware while keeping ESP32 in reset state.

The ESP32 firmware monitors Himax and may reset it if it detects anomalies.
This script holds ESP32 in reset during Himax flashing to prevent interference.
"""

import serial
import subprocess
import sys
import time
import os

ESP32_PORT = "/dev/cu.wchusbserial58370593761"
HIMAX_PORT = "/dev/cu.usbmodem58370593761"
FIRMWARE = "we2_image_gen_local/output_case1_sec_wlcsp/output.img"

def hold_esp32_reset(port):
    """Hold ESP32 in reset by controlling DTR/RTS pins."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200
    ser.dtr = False  # EN pin - False = reset
    ser.rts = True   # GPIO0 - True = download mode
    ser.open()
    return ser

def release_esp32_reset(ser):
    """Release ESP32 from reset."""
    ser.dtr = True   # Release EN pin
    ser.rts = False  # Release GPIO0
    time.sleep(0.1)
    ser.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    firmware_path = os.path.join(script_dir, FIRMWARE)

    print("=== Himax Firmware Flasher (Safe Mode) ===")
    print(f"ESP32 Port: {ESP32_PORT}")
    print(f"Himax Port: {HIMAX_PORT}")
    print(f"Firmware: {firmware_path}")
    print()

    # Check firmware exists
    if not os.path.exists(firmware_path):
        print(f"ERROR: Firmware not found: {firmware_path}")
        sys.exit(1)

    # Step 1: Hold ESP32 in reset
    print("Step 1: Holding ESP32 in reset state...")
    try:
        esp32_ser = hold_esp32_reset(ESP32_PORT)
        print("  ESP32 is now held in reset (won't interfere with Himax)")
    except Exception as e:
        print(f"ERROR: Could not open ESP32 port: {e}")
        sys.exit(1)

    # Step 2: Flash Himax
    print()
    print("Step 2: Flashing Himax firmware...")
    print("=" * 50)
    print(">>> Press Himax RESET button when prompted! <<<")
    print("=" * 50)
    print()

    try:
        # Use xmodem_send.py for flashing
        xmodem_script = os.path.join(script_dir, "xmodem", "xmodem_send.py")
        result = subprocess.run(
            [
                sys.executable, xmodem_script,
                "--port", HIMAX_PORT,
                "--baudrate", "921600",
                "--file", firmware_path
            ],
            timeout=120
        )
        flash_success = (result.returncode == 0)
    except Exception as e:
        print(f"Flash error: {e}")
        flash_success = False

    # Step 3: Release ESP32
    print()
    print("Step 3: Releasing ESP32 from reset...")
    release_esp32_reset(esp32_ser)
    print("  ESP32 released, should be booting now")

    # Wait for devices to stabilize
    time.sleep(2)

    if flash_success:
        print()
        print("=== Flash Complete ===")
        print("Both ESP32 and Himax should be running now.")
    else:
        print()
        print("=== Flash may have failed ===")
        print("Check the output above for errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
