#!/usr/bin/env python3
"""
SenseCAP Watcher Flash Tool
===========================

Safely flash firmware and models to Himax WE2 on SenseCAP Watcher.
Automatically holds ESP32 in reset to prevent interference during flashing.

Usage:
    # Flash firmware only
    uv run python flash_watcher.py --firmware

    # Flash models only (face recognition)
    uv run python flash_watcher.py --models

    # Flash both firmware and models
    uv run python flash_watcher.py --firmware --models

    # Flash specific model
    uv run python flash_watcher.py --model-scrfd /path/to/scrfd.tflite
    uv run python flash_watcher.py --model-facenet /path/to/ghostfacenet.tflite
    uv run python flash_watcher.py --model-yolo /path/to/swift_yolo.tflite

Author: Claude
"""

import serial
import xmodem
import time
import os
import sys
import math
import argparse
import threading
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Default serial ports for SenseCAP Watcher
DEFAULT_ESP32_PORT = "/dev/cu.wchusbserial5AF91659653"
DEFAULT_HIMAX_PORT = "/dev/cu.usbmodem5AF91659651"

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_FIRMWARE = PROJECT_ROOT / "we2_image_gen_local/output_case1_sec_wlcsp/output.img"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "model_zoo/tflm_face_embedding"

# Model files - use model_zoo reference models
DEFAULT_SCRFD_MODEL = DEFAULT_MODELS_DIR / "scrfd/models/scrfd_500m_kps_int8_vela.tflite"
DEFAULT_FACENET_MODEL = DEFAULT_MODELS_DIR / "ghostfacenet/models/ghostfacenet_0.5_112_int8_vela.tflite"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "model_zoo/sscma/swift_yolo_nano_person_192_int8_vela.tflite"

# Flash addresses for models (must match common_config.h)
# Memory layout: sscma_micro scans 0x400000-0xE00000
MODEL_ADDRESSES = {
    "scrfd": 0x400000,      # Face detection model
    "facenet": 0x510000,    # Face embedding model (GhostFaceNet/MobileFaceNet)
    "yolo": 0x700000,       # Object detection model (Swift YOLO)
}

# Serial settings
BAUDRATE = 921600
TIMEOUT = 60

# =============================================================================
# Global state
# =============================================================================

ser = None
esp32_ser = None
send_bin_total_packets = 0


# =============================================================================
# Utility functions
# =============================================================================

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(step, text):
    """Print a step indicator."""
    print(f"\n[Step {step}] {text}")


def progress_callback(total_packets, success_count, error_count):
    """Progress bar callback for xmodem transfer."""
    global send_bin_total_packets
    if send_bin_total_packets == 0:
        return

    bar_total = 40
    progress = total_packets / send_bin_total_packets
    bar_cnt = int(progress * bar_total)
    space_cnt = bar_total - bar_cnt

    bar_string = f"\r  [{'█' * bar_cnt}{' ' * space_cnt}] {progress:.1%} ({total_packets}/{send_bin_total_packets})"
    if error_count > 0:
        bar_string += f" errors: {error_count}"

    print(bar_string, end="", flush=True)

    if progress >= 1:
        print()  # New line after completion


def find_serial_ports():
    """Auto-detect serial ports for SenseCAP Watcher."""
    import glob

    ports = {
        "esp32": None,
        "himax": None
    }

    # macOS patterns
    wch_ports = glob.glob("/dev/cu.wchusbserial*")
    usb_ports = glob.glob("/dev/cu.usbmodem*")

    # Linux patterns
    if not wch_ports:
        wch_ports = glob.glob("/dev/ttyUSB*")
    if not usb_ports:
        usb_ports = glob.glob("/dev/ttyACM*")

    if wch_ports:
        # Sort to get consistent ordering, prefer *53 for ESP32
        wch_ports.sort()
        for p in wch_ports:
            if p.endswith("53"):
                ports["esp32"] = p
                break
        if not ports["esp32"] and wch_ports:
            ports["esp32"] = wch_ports[-1]

    if usb_ports:
        usb_ports.sort()
        for p in usb_ports:
            if p.endswith("51"):
                ports["himax"] = p
                break
        if not ports["himax"] and usb_ports:
            ports["himax"] = usb_ports[0]

    return ports


# =============================================================================
# ESP32 Reset Control
# =============================================================================

class ESP32ResetController:
    """Context manager to hold ESP32 in reset during Himax operations."""

    def __init__(self, port, verbose=True):
        self.port = port
        self.ser = None
        self.verbose = verbose

    def __enter__(self):
        if not self.port:
            if self.verbose:
                print("  Warning: No ESP32 port specified, skipping reset control")
            return self

        try:
            self.ser = serial.Serial(self.port, 115200)
            self.ser.dtr = False  # Hold EN low (reset)
            if self.verbose:
                print(f"  ESP32 held in reset ({self.port})")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not control ESP32 reset: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ser:
            try:
                self.ser.dtr = True  # Release reset
                time.sleep(0.5)
                self.ser.close()
                if self.verbose:
                    print("  ESP32 released from reset")
            except:
                pass
        return False


# =============================================================================
# Himax Flash Functions
# =============================================================================

def open_himax_serial(port, baudrate=BAUDRATE, timeout=TIMEOUT):
    """Open serial connection to Himax."""
    global ser
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = timeout
    ser.bytesize = serial.EIGHTBITS
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = 0
    ser.rtscts = 0
    ser.parity = serial.PARITY_NONE
    ser.open()
    ser.flushInput()
    ser.flushOutput()
    print(f"  Himax serial opened ({port} @ {baudrate})")
    return ser


def send_at_command(command):
    """Send AT command to Himax."""
    ser.write(bytes(command + "\r", encoding='ascii'))


def wait_for_xmodem_mode(timeout=30):
    """Wait for Himax to enter xmodem download mode."""
    print("  Waiting for Himax to enter download mode...")
    print("  >>> Press the Himax Reset button NOW! <<<")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = ser.readline().strip()
            if response:
                text = response.decode('utf-8', errors='ignore')
                print(f"    {text}")
            send_at_command('1')

            # Must wait for the actual xmodem prompt, not just "X-Modem mode" mention
            if b'Send data using the xmodem protocol from your terminal' in response:
                time.sleep(1)
                ser.flushInput()
                send_at_command('1')
                return True
        except:
            pass

    return False


def xmodem_send_file(filepath, description="file"):
    """Send a file via xmodem protocol."""
    global send_bin_total_packets

    if not os.path.exists(filepath):
        print(f"  Error: File not found: {filepath}")
        return False

    file_size = os.path.getsize(filepath)
    packet_size = 128  # xmodem uses 128-byte packets
    send_bin_total_packets = math.ceil(file_size / packet_size)

    print(f"  Sending {description}: {filepath}")
    print(f"  Size: {file_size:,} bytes ({send_bin_total_packets} packets)")

    modem = xmodem.XMODEM(getc=lambda size, timeout=1: ser.read(size),
                          putc=lambda data, timeout=1: ser.write(data),
                          mode='xmodem')

    with open(filepath, 'rb') as f:
        result = modem.send(f, callback=progress_callback)

    if result:
        print(f"  {description} sent successfully!")
    else:
        print(f"  Error: Failed to send {description}")

    return result


def send_model_preamble(flash_address, packet_size=128):
    """Send preamble header before model data."""
    # Preamble format: [0xC0, 0x5A] + address(4) + offset(4) + [0x5A, 0xC0] + padding
    header = bytes([0xC0, 0x5A])
    header += flash_address.to_bytes(4, 'little')
    header += (0).to_bytes(4, 'little')  # offset = 0
    header += bytes([0x5A, 0xC0])
    header += bytes([0xFF] * (packet_size - 12))

    # Create temp file for preamble
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(header)
        temp_path = f.name

    try:
        result = xmodem_send_file(temp_path, f"preamble (addr=0x{flash_address:06X})")
    finally:
        os.unlink(temp_path)

    return result


def wait_for_reboot_prompt(timeout=30):
    """Wait for the 'Do you want to reboot?' prompt."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = ser.readline().strip()
            if response:
                text = response.decode('utf-8', errors='ignore')
                if 'Do you want to end file transmission' in text or 'reboot system' in text:
                    return True
        except:
            pass
    return False


def flash_model(model_path, flash_address, description):
    """Flash a model file to specified address."""
    print(f"\n  Flashing {description} to 0x{flash_address:06X}...")

    # Wait for reboot prompt, then continue
    if not wait_for_reboot_prompt(timeout=10):
        print("  Warning: Did not receive reboot prompt, continuing anyway...")

    time.sleep(1)
    ser.flushInput()
    send_at_command('n')  # Don't reboot, continue with more files

    # Send preamble
    if not send_model_preamble(flash_address):
        print(f"  Error: Failed to send preamble for {description}")
        return False

    # Wait for prompt again
    if not wait_for_reboot_prompt(timeout=10):
        print("  Warning: Did not receive reboot prompt after preamble")

    time.sleep(1)
    ser.flushInput()
    send_at_command('n')

    # Send model file
    return xmodem_send_file(model_path, description)


# =============================================================================
# Main Flash Functions
# =============================================================================

def flash_firmware(himax_port, esp32_port, firmware_path):
    """Flash firmware image to Himax."""
    print_header("Flashing Himax Firmware")

    if not os.path.exists(firmware_path):
        print(f"Error: Firmware not found: {firmware_path}")
        return False

    print(f"Firmware: {firmware_path}")
    print(f"Size: {os.path.getsize(firmware_path):,} bytes")

    with ESP32ResetController(esp32_port):
        print_step(1, "Opening Himax serial port")
        try:
            open_himax_serial(himax_port)
        except Exception as e:
            print(f"  Error: Could not open Himax port: {e}")
            return False

        print_step(2, "Entering download mode")
        if not wait_for_xmodem_mode():
            print("  Error: Timeout waiting for download mode")
            return False

        print_step(3, "Sending firmware")
        result = xmodem_send_file(firmware_path, "firmware")

        if result:
            print_step(4, "Rebooting device")
            wait_for_reboot_prompt(timeout=10)
            time.sleep(1)
            ser.flushInput()
            send_at_command('y')  # Confirm reboot

            time.sleep(2)
            try:
                for _ in range(10):
                    response = ser.readline().strip()
                    if response:
                        print(f"    {response.decode('utf-8', errors='ignore')}")
            except:
                pass

    return result


def flash_models(himax_port, esp32_port, models):
    """Flash model files to Himax."""
    print_header("Flashing Models")

    # Validate models
    valid_models = []
    for name, path, addr in models:
        if os.path.exists(path):
            print(f"  {name}: {path} -> 0x{addr:06X}")
            valid_models.append((name, path, addr))
        else:
            print(f"  {name}: NOT FOUND - {path}")

    if not valid_models:
        print("\nError: No valid model files found")
        return False

    with ESP32ResetController(esp32_port):
        print_step(1, "Opening Himax serial port")
        try:
            open_himax_serial(himax_port)
        except Exception as e:
            print(f"  Error: Could not open Himax port: {e}")
            return False

        print_step(2, "Entering download mode")
        if not wait_for_xmodem_mode():
            print("  Error: Timeout waiting for download mode")
            return False

        # Need to send a dummy file first to start the model flashing sequence
        # The device expects firmware first, then models
        # We'll use a minimal preamble to signal model-only flash
        print_step(3, "Initiating model flash sequence")

        for i, (name, path, addr) in enumerate(valid_models):
            print_step(4 + i, f"Flashing {name}")
            if not flash_model(path, addr, name):
                print(f"  Error: Failed to flash {name}")
                return False

        print_step(4 + len(valid_models), "Rebooting device")
        # Send 'y' to reboot
        wait_for_reboot_prompt(timeout=10)
        time.sleep(1)
        ser.flushInput()
        send_at_command('y')

        time.sleep(2)
        try:
            for _ in range(10):
                response = ser.readline().strip()
                if response:
                    print(f"    {response.decode('utf-8', errors='ignore')}")
        except:
            pass

    return True


def flash_firmware_and_models(himax_port, esp32_port, firmware_path, models):
    """Flash both firmware and models in one session."""
    print_header("Flashing Firmware and Models")

    if not os.path.exists(firmware_path):
        print(f"Error: Firmware not found: {firmware_path}")
        return False

    # Validate models
    valid_models = []
    for name, path, addr in models:
        if os.path.exists(path):
            valid_models.append((name, path, addr))

    print(f"Firmware: {firmware_path}")
    print(f"Models: {len(valid_models)} files")
    for name, path, addr in valid_models:
        print(f"  - {name}: 0x{addr:06X}")

    with ESP32ResetController(esp32_port):
        print_step(1, "Opening Himax serial port")
        try:
            open_himax_serial(himax_port)
        except Exception as e:
            print(f"  Error: Could not open Himax port: {e}")
            return False

        print_step(2, "Entering download mode")
        if not wait_for_xmodem_mode():
            print("  Error: Timeout waiting for download mode")
            return False

        print_step(3, "Sending firmware")
        if not xmodem_send_file(firmware_path, "firmware"):
            print("  Error: Failed to send firmware")
            return False

        # Flash models
        for i, (name, path, addr) in enumerate(valid_models):
            print_step(4 + i, f"Flashing {name}")
            if not flash_model(path, addr, name):
                print(f"  Error: Failed to flash {name}")
                return False

        print_step(4 + len(valid_models), "Rebooting device")
        wait_for_reboot_prompt(timeout=10)
        time.sleep(1)
        ser.flushInput()
        send_at_command('y')

        time.sleep(2)
        try:
            for _ in range(10):
                response = ser.readline().strip()
                if response:
                    print(f"    {response.decode('utf-8', errors='ignore')}")
        except:
            pass

    return True


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SenseCAP Watcher Flash Tool - Flash firmware and models to Himax WE2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Flash firmware only
  uv run python flash_watcher.py --firmware

  # Flash face recognition models
  uv run python flash_watcher.py --models

  # Flash both firmware and models
  uv run python flash_watcher.py --firmware --models

  # Flash custom model
  uv run python flash_watcher.py --model-scrfd /path/to/scrfd.tflite

Flash addresses:
  SCRFD (face detection):  0x400000
  FaceNet (embeddings):    0x510000
  YOLO (object detection): 0x700000
"""
    )

    # Port options
    parser.add_argument("--himax-port", type=str, default=None,
                        help="Himax serial port (auto-detected if not specified)")
    parser.add_argument("--esp32-port", type=str, default=None,
                        help="ESP32 serial port for reset control (auto-detected if not specified)")

    # Firmware options
    parser.add_argument("--firmware", action="store_true",
                        help="Flash firmware image")
    parser.add_argument("--firmware-path", type=str, default=str(DEFAULT_FIRMWARE),
                        help=f"Path to firmware image (default: {DEFAULT_FIRMWARE})")

    # Model options
    parser.add_argument("--models", action="store_true",
                        help="Flash all default face recognition models")
    parser.add_argument("--model-scrfd", type=str, default=None,
                        help="Path to SCRFD face detection model")
    parser.add_argument("--model-facenet", type=str, default=None,
                        help="Path to FaceNet/GhostFaceNet embedding model")
    parser.add_argument("--model-yolo", type=str, default=None,
                        help="Path to YOLO object detection model")

    # Other options
    parser.add_argument("--list-ports", action="store_true",
                        help="List detected serial ports and exit")
    parser.add_argument("--no-esp32-reset", action="store_true",
                        help="Don't control ESP32 reset (for standalone Grove Vision AI)")

    args = parser.parse_args()

    # Auto-detect ports
    detected_ports = find_serial_ports()
    himax_port = args.himax_port or detected_ports.get("himax") or DEFAULT_HIMAX_PORT
    esp32_port = None if args.no_esp32_reset else (args.esp32_port or detected_ports.get("esp32") or DEFAULT_ESP32_PORT)

    if args.list_ports:
        print("Detected serial ports:")
        print(f"  ESP32: {detected_ports.get('esp32', 'Not found')}")
        print(f"  Himax: {detected_ports.get('himax', 'Not found')}")
        return 0

    # Build model list
    models = []
    if args.models:
        models.append(("SCRFD", str(DEFAULT_SCRFD_MODEL), MODEL_ADDRESSES["scrfd"]))
        models.append(("GhostFaceNet", str(DEFAULT_FACENET_MODEL), MODEL_ADDRESSES["facenet"]))

    if args.model_scrfd:
        models.append(("SCRFD", args.model_scrfd, MODEL_ADDRESSES["scrfd"]))
    if args.model_facenet:
        models.append(("FaceNet", args.model_facenet, MODEL_ADDRESSES["facenet"]))
    if args.model_yolo:
        models.append(("YOLO", args.model_yolo, MODEL_ADDRESSES["yolo"]))

    # Check what to do
    if not args.firmware and not models:
        print("Error: Specify --firmware and/or --models (or specific model paths)")
        print("Use --help for usage information")
        return 1

    print_header("SenseCAP Watcher Flash Tool")
    print(f"Himax port: {himax_port}")
    print(f"ESP32 port: {esp32_port or 'disabled'}")

    # Execute flashing
    success = False
    try:
        if args.firmware and models:
            success = flash_firmware_and_models(himax_port, esp32_port, args.firmware_path, models)
        elif args.firmware:
            success = flash_firmware(himax_port, esp32_port, args.firmware_path)
        elif models:
            success = flash_models(himax_port, esp32_port, models)
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if ser and ser.is_open:
            ser.close()

    if success:
        print_header("Flash Complete!")
        print("Device should be rebooting with new firmware/models.")
        return 0
    else:
        print_header("Flash Failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
