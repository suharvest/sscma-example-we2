#!/usr/bin/env python3
"""
SenseCAP Watcher Flash Tool (Himax WE2 multi-model flasher)
===========================================================

Safely flash firmware and an ARBITRARY number of models to the Himax WE2 on a
SenseCAP Watcher, in ONE continuous XMODEM session. Automatically holds the
ESP32 in reset to prevent interference during flashing.

This tool is aligned with the OFFICIAL SenseCraft provisioning flow
(app_collaboration/provisioning_station/deployers/himax_deployer.py::
_flash_with_xmodem_multimodel) which:
  1. enters the WE2 bootloader (repeatedly sends "1"),
  2. sends the base firmware image via XMODEM,
  3. for each model: answers "n" to the reboot prompt, sends a 12-byte
     address preamble (magic C0 5A + addr[4] + offset[4] + 5A C0), answers "n"
     again, then sends the model binary via XMODEM,
  4. answers "y" to the final reboot prompt.

Model identity on the device is determined by FLASH ADDRESS ONLY.

  >>> IMPORTANT: INFO / class-name metadata is NOT written by this tool. <<<
  The official SenseCraft Himax deployer does NOT send any `AT+INFO` (or any
  other) command to write per-model class names / metadata to the WE2. It was
  audited (see himax_deployer.py, xmodem_send.py, device.schema.json,
  watcher_himax.yaml) and no such write path exists — the WE2 firmware / sscma
  layer manages class labels internally, keyed by flash address. There is
  therefore no `--write-info` behaviour here on purpose; see write_model_info().

Usage:
    # Flash firmware only
    uv run --with pyyaml python flash_watcher.py --firmware

    # Flash firmware + the full built-in 4-model layout (SCRFD / FaceNet /
    # SCRFD copy / Person-YOLO) with sha256 verification
    uv run --with pyyaml python flash_watcher.py --firmware --models

    # Drive everything from the official device YAML (one command, N models)
    uv run --with pyyaml python flash_watcher.py --firmware \
        --from-yaml ~/project/sensecraft-solutions/solutions/smart_space_assistant/devices/watcher_himax.yaml \
        --model-dir ~/project/grove_vision_2/sscma-example-we2/model_zoo

    # Flash arbitrary models by hand (repeatable): NAME:PATH:ADDR[:SHA256]
    uv run --with pyyaml python flash_watcher.py --firmware \
        --model "SCRFD:/path/scrfd.tflite:0x400000:86296f..." \
        --model "Person:/path/yolo.tflite:0x700000"

    # Back-compat single-slot flags
    uv run --with pyyaml python flash_watcher.py --firmware --model-scrfd /path/to/scrfd.tflite

Author: Claude
"""

import serial
import xmodem
import time
import os
import sys
import math
import hashlib
import argparse
import threading
from pathlib import Path
from collections import namedtuple

try:
    import yaml  # PyYAML; only needed for --from-yaml
except ImportError:
    yaml = None

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
DEFAULT_FACENET_MODEL = DEFAULT_MODELS_DIR / "qat_distill_v2_relu6_128d/model_128d.int8_vela.tflite"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "model_zoo/sscma/swift_yolo_nano_person_192_int8_vela.tflite"

# Flash addresses for models (must match common_config.h)
# Memory layout: sscma_micro scans 0x400000-0xE00000
MODEL_ADDRESSES = {
    "scrfd": 0x400000,      # Face detection model
    "facenet": 0x510000,    # Face embedding model (stage80 w600k MobileFaceNet)
    "scrfd2": 0x650000,     # Secondary SCRFD instance (moved from 0x600000: avoid overlap with FaceNet tail 0x64BFD0)
    "yolo": 0x700000,       # Object / person detection model (Swift YOLO)
}

# -----------------------------------------------------------------------------
# ModelSpec: one flash entry. `classes` is carried for logging/future use only
# and is NOT written to the device (no official INFO write path exists).
# -----------------------------------------------------------------------------
ModelSpec = namedtuple(
    "ModelSpec",
    ["id", "name", "path", "address", "offset", "sha256", "classes"],
)


def make_spec(name, path, address, *, id=None, offset=0, sha256=None, classes=None):
    return ModelSpec(
        id=id or name,
        name=name,
        path=str(path),
        address=address,
        offset=offset,
        sha256=(sha256.lower() if sha256 else None),
        classes=classes or [],
    )


# Built-in default 4-model layout, mirroring the official
# smart_space_assistant/devices/watcher_himax.yaml. sha256 values are the
# OFFICIAL published-model hashes (from sensecraft-statics). Local model_zoo
# files may legitimately differ; a mismatch is a WARNING unless --strict-checksum.
DEFAULT_MODELS4 = [
    make_spec("SCRFD (face detection)", DEFAULT_SCRFD_MODEL, MODEL_ADDRESSES["scrfd"],
              id="face_detection", classes=["face"],
              sha256="86296f513339ecf83e4f7a97e388cbb7331d478ab6b40824ebf9f50e54578e7b"),
    make_spec("MobileFaceNet (embedding)", DEFAULT_FACENET_MODEL, MODEL_ADDRESSES["facenet"],
              id="face_embedding", classes=[],
              sha256="bd1ccc83a9e8bf854a0bf47ce21d415c901338831d6105e182d78c10f0651874"),
    make_spec("SCRFD copy (face detection 2)", DEFAULT_SCRFD_MODEL, MODEL_ADDRESSES["scrfd2"],
              id="face_detection_2", classes=["face"],
              sha256="86296f513339ecf83e4f7a97e388cbb7331d478ab6b40824ebf9f50e54578e7b"),
    make_spec("Person (Swift YOLO)", DEFAULT_YOLO_MODEL, MODEL_ADDRESSES["yolo"],
              id="person_detection", classes=["person"],
              sha256="67621369cae06a0b661d4491111b2b5eec90fc4c397da9230554b003038c1049"),
]

# Serial settings
BAUDRATE = 921600
TIMEOUT = 60

# =============================================================================
# Global state
# =============================================================================

ser = None
esp32_ser = None
send_bin_total_packets = 0
last_progress_percent = -1


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


def sha256_file(path, chunk=1 << 20):
    """Compute the SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_model_checksums(models, strict=False, skip=False):
    """Verify each model's sha256 (when declared) before flashing.

    Returns True to proceed, False to abort. Always prints the computed digest
    so it can be copied into the device YAML. A declared/computed mismatch is a
    WARNING by default (local model_zoo files may differ from the official
    published hashes) and only fatal under --strict-checksum.
    """
    if skip:
        print("  (checksum verification skipped: --skip-checksum)")
        return True

    ok = True
    for m in models:
        if not os.path.exists(m.path):
            print(f"  [checksum] MISSING FILE: {m.name} -> {m.path}")
            ok = False
            continue
        digest = sha256_file(m.path)
        if not m.sha256:
            print(f"  [checksum] {m.name}: sha256={digest} (no expected value to compare)")
            continue
        if digest == m.sha256:
            print(f"  [checksum] {m.name}: OK ({digest})")
        else:
            print(f"  [checksum] {m.name}: MISMATCH")
            print(f"               expected {m.sha256}")
            print(f"               actual   {digest}")
            if strict:
                ok = False
            else:
                print("               (WARNING only; pass --strict-checksum to make fatal)")
    return ok


def load_models_from_yaml(yaml_path, model_dir=None):
    """Build a list[ModelSpec] from an official device YAML (watcher_himax.yaml).

    Maps each `models[]` entry:
      id            -> ModelSpec.id
      name          -> ModelSpec.name
      flash_address -> ModelSpec.address   (hex string or int)
      offset        -> ModelSpec.offset    (hex string or int, default 0)
      checksum.sha256 -> ModelSpec.sha256  (verified pre-flash)
      path          -> resolved to a LOCAL file (see below)

    Path resolution: YAML `path` is typically an https URL to the published
    model. This flasher needs a local file, so it resolves in order:
      1. an existing local path (absolute or relative to the YAML dir),
      2. <model_dir>/<basename-of-path> if --model-dir was given,
      3. otherwise raises with a clear message.
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML not installed. Run with: uv run --with pyyaml python flash_watcher.py ..."
        )
    yaml_path = Path(yaml_path).expanduser()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Official device YAML nests the list under `firmware.flash_config.models`.
    # Prefer that exact path, then fall back to a recursive search for the first
    # `models:` list-of-dicts so we survive schema-nesting changes.
    def _find_models(node):
        if isinstance(node, dict):
            m = node.get("models")
            if isinstance(m, list) and m and isinstance(m[0], dict):
                return m
            for v in node.values():
                found = _find_models(v)
                if found:
                    return found
        elif isinstance(node, list):
            for v in node:
                found = _find_models(v)
                if found:
                    return found
        return None

    fw_cfg = (data.get("firmware") or {}).get("flash_config") or {}
    entries = fw_cfg.get("models") or _find_models(data) or []
    if not entries:
        raise ValueError(
            f"No 'firmware.flash_config.models:' (or any 'models:') list found in {yaml_path}")

    def _to_int(v, default=0):
        if v is None:
            return default
        if isinstance(v, int):
            return v
        return int(str(v), 16) if str(v).lower().startswith("0x") else int(str(v))

    def _resolve_path(raw):
        raw = str(raw)
        # 1. existing local path (abs or relative to YAML dir)
        cand = Path(raw).expanduser()
        if cand.is_file():
            return str(cand)
        rel = (yaml_path.parent / raw)
        if rel.is_file():
            return str(rel)
        # 2. <model_dir>/<basename> (recursive)
        base = os.path.basename(raw.split("?")[0])
        if model_dir:
            hit = list(Path(model_dir).expanduser().rglob(base))
            if hit:
                return str(hit[0])
        # 3. give up with guidance (local-only: no network download)
        raise FileNotFoundError(
            f"Could not resolve a local file for model path '{raw}'. "
            f"Provide the file locally or pass --model-dir containing '{base}'."
        )

    specs = []
    for e in entries:
        specs.append(make_spec(
            name=e.get("name") or e.get("id") or "model",
            path=_resolve_path(e.get("path")),
            address=_to_int(e.get("flash_address")),
            id=e.get("id"),
            offset=_to_int(e.get("offset"), 0),
            sha256=(e.get("checksum") or {}).get("sha256"),
            classes=e.get("classes") or [],
        ))
    return specs


def write_model_info(model):
    """Placeholder for per-model INFO/class-name metadata writing.

    INTENTIONALLY A NO-OP. An audit of the official SenseCraft provisioning
    code (himax_deployer.py::_flash_with_xmodem_multimodel, xmodem_send.py,
    device.schema.json, watcher_himax.yaml) found NO official mechanism that
    writes class names / INFO to the WE2 over serial — there is no `AT+INFO`
    command and the YAML schema has no `classes` field for Himax models. Model
    identity is by flash address only. This stub exists so the intent is
    documented; it deliberately does not invent a protocol. If a real INFO
    write path is later discovered in firmware, implement it here.
    """
    return  # no official basis — do nothing


def progress_callback(total_packets, success_count, error_count):
    """Progress bar callback for xmodem transfer."""
    global send_bin_total_packets, last_progress_percent
    if send_bin_total_packets == 0:
        return

    percent = int((total_packets * 100) / send_bin_total_packets)
    if percent == last_progress_percent and total_packets < send_bin_total_packets:
        return
    last_progress_percent = percent

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

def open_himax_serial(port, baudrate=BAUDRATE, timeout=TIMEOUT, retries=5, retry_delay=1.0):
    """Open serial connection to Himax with retry on Resource busy."""
    global ser
    last_error = None
    for attempt in range(retries):
        try:
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
        except serial.SerialException as e:
            last_error = e
            if attempt < retries - 1:
                print(f"  Port busy, retrying in {retry_delay}s... ({attempt + 1}/{retries})")
                time.sleep(retry_delay)
            else:
                raise last_error


def send_at_command(command):
    """Send AT command to Himax."""
    ser.write(bytes(command + "\r", encoding='ascii'))


def wait_for_xmodem_mode(timeout=30):
    """Wait for Himax to enter xmodem download mode."""
    print("  Waiting for Himax to enter download mode...")
    print("  >>> Press the Himax Reset button NOW! <<<")

    old_timeout = ser.timeout
    ser.timeout = 0.1
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            try:
                response = ser.readline().strip()
                if response:
                    text = response.decode('utf-8', errors='ignore')
                    print(f"    {text}")

                # The bootloader menu is only visible while a key is held during
                # reset. Keep sending the menu key with a short serial timeout so
                # Grove Vision can enter download mode without manual key hold.
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
    finally:
        ser.timeout = old_timeout


def xmodem_send_file(filepath, description="file"):
    """Send a file via xmodem protocol."""
    global send_bin_total_packets, last_progress_percent

    if not os.path.exists(filepath):
        print(f"  Error: File not found: {filepath}")
        return False

    file_size = os.path.getsize(filepath)
    packet_size = 128  # xmodem uses 128-byte packets
    send_bin_total_packets = math.ceil(file_size / packet_size)
    last_progress_percent = -1

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


def send_model_preamble(flash_address, offset=0, packet_size=128):
    """Send preamble header before model data.

    Matches the official preamble
    (himax_deployer.py::_generate_preamble / xmodem_send.py):
    [0xC0, 0x5A] + address(4, little) + offset(4, little) + [0x5A, 0xC0] + 0xFF pad
    """
    header = bytes([0xC0, 0x5A])
    header += flash_address.to_bytes(4, 'little')
    header += offset.to_bytes(4, 'little')  # offset (usually 0)
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
    old_timeout = ser.timeout
    ser.timeout = 0.1
    start_time = time.time()
    try:
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
    finally:
        ser.timeout = old_timeout


def flash_model(model_path, flash_address, description, offset=0):
    """Flash a model file to specified address."""
    print(f"\n  Flashing {description} to 0x{flash_address:06X}...")

    # Wait for reboot prompt, then continue
    if not wait_for_reboot_prompt(timeout=10):
        print("  Warning: Did not receive reboot prompt, continuing anyway...")

    time.sleep(1)
    ser.flushInput()
    send_at_command('n')  # Don't reboot, continue with more files

    # Send preamble
    if not send_model_preamble(flash_address, offset):
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
    """Reject standalone model flashing.

    Himax WE2's bootloader accepts the firmware image first, then optional
    model files in the same XMODEM session. It does not expose a reliable
    model-only entry point from the initial bootloader menu, so attempting to
    send only a model will block waiting for a continuation prompt that never
    arrives.
    """
    print_header("Flashing Models")
    print("Error: model-only flashing is not supported by this bootloader flow.")
    print("Use --firmware together with --models or --model-* so firmware and")
    print("models are sent in one continuous XMODEM session.")
    print("")
    print("Example:")
    print("  uv run python -u flash_watcher.py --firmware --model-facenet /path/to/model.tflite")
    return False


def flash_firmware_and_models(himax_port, esp32_port, firmware_path, models,
                              strict_checksum=False, skip_checksum=False):
    """Flash firmware and an arbitrary number of models in one XMODEM session.

    `models` is a list[ModelSpec]. Order is preserved (matches the official
    per-model loop). sha256 verification runs BEFORE any serial activity.
    """
    print_header("Flashing Firmware and Models")

    if not os.path.exists(firmware_path):
        print(f"Error: Firmware not found: {firmware_path}")
        return False

    # Validate presence
    valid_models = [m for m in models if os.path.exists(m.path)]
    missing = [m for m in models if not os.path.exists(m.path)]
    for m in missing:
        print(f"  Warning: skipping missing model {m.name} -> {m.path}")

    if not valid_models:
        print("Error: no valid model files to flash.")
        return False

    print(f"Firmware: {firmware_path}")
    print(f"Models: {len(valid_models)} file(s)")
    for m in valid_models:
        cls = f" classes={m.classes}" if m.classes else ""
        print(f"  - {m.name}: 0x{m.address:06X} (offset 0x{m.offset:X}){cls}")

    # sha256 verification BEFORE touching the serial port
    print_step(0, "Verifying model checksums (sha256)")
    if not verify_model_checksums(valid_models, strict=strict_checksum, skip=skip_checksum):
        print("  Error: checksum verification failed (see --skip-checksum to bypass).")
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

        print_step(3, "Sending firmware")
        if not xmodem_send_file(firmware_path, "firmware"):
            print("  Error: Failed to send firmware")
            return False

        # Flash each model (address preamble + binary), preserving order.
        for i, m in enumerate(valid_models):
            print_step(4 + i, f"Flashing {m.name}")
            if not flash_model(m.path, m.address, m.name, offset=m.offset):
                print(f"  Error: Failed to flash {m.name}")
                return False
            # INFO/class metadata: intentionally a no-op (no official write path).
            write_model_info(m)

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

def parse_model_arg(value):
    """Parse a --model 'NAME:PATH:ADDR[:SHA256]' argument into a ModelSpec.

    ADDR is hex (0x...) or decimal. This simple parser splits on ':' and does
    not target Windows-style drive letters inside PATH.
    """
    parts = value.split(":")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            f"--model expects NAME:PATH:ADDR[:SHA256], got '{value}'")
    name, path, addr = parts[0], parts[1], parts[2]
    sha = parts[3] if len(parts) > 3 and parts[3] else None
    address = int(addr, 16) if addr.lower().startswith("0x") else int(addr)
    return make_spec(name=name, path=path, address=address, sha256=sha)


def main():
    parser = argparse.ArgumentParser(
        description="SenseCAP Watcher Flash Tool - Flash firmware and N models to Himax WE2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Firmware only
  uv run --with pyyaml python flash_watcher.py --firmware

  # Firmware + built-in 4-model layout (SCRFD / FaceNet / SCRFD copy / Person) + sha256
  uv run --with pyyaml python flash_watcher.py --firmware --models

  # Drive from the official device YAML (one command, any number of models)
  uv run --with pyyaml python flash_watcher.py --firmware \\
      --from-yaml .../devices/watcher_himax.yaml --model-dir .../model_zoo

  # Arbitrary models by hand (repeatable): NAME:PATH:ADDR[:SHA256]
  uv run --with pyyaml python flash_watcher.py --firmware \\
      --model "SCRFD:/p/scrfd.tflite:0x400000" --model "Person:/p/yolo.tflite:0x700000"

Built-in 4-model layout (matches smart_space_assistant/devices/watcher_himax.yaml):
  SCRFD  (face detection):    0x400000
  FaceNet (embeddings):       0x510000
  SCRFD copy (detection 2):   0x600000
  Person (Swift YOLO):        0x700000

NOTE: This tool does NOT write INFO / class-name metadata to the WE2. The
official SenseCraft Himax deployer has no such write path (audited); model
identity is by flash address only.
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
                        help="Flash the built-in 4-model layout (requires --firmware)")
    parser.add_argument("--from-yaml", type=str, default=None,
                        help="Load the model list from an official device YAML "
                             "(e.g. watcher_himax.yaml). Requires --firmware and PyYAML.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory to resolve YAML model paths (searched recursively "
                             "by basename when the YAML path is a URL). Local files only.")
    parser.add_argument("--model", action="append", default=[], metavar="NAME:PATH:ADDR[:SHA256]",
                        help="Add an arbitrary model (repeatable). Requires --firmware.")

    # Back-compat single-slot flags
    parser.add_argument("--model-scrfd", type=str, default=None,
                        help="Path to SCRFD face detection model -> 0x400000 (requires --firmware)")
    parser.add_argument("--model-facenet", type=str, default=None,
                        help="Path to FaceNet embedding model -> 0x510000 (requires --firmware)")
    parser.add_argument("--model-yolo", type=str, default=None,
                        help="Path to YOLO/person model -> 0x700000 (requires --firmware)")

    # Checksum control
    parser.add_argument("--strict-checksum", action="store_true",
                        help="Abort if any declared sha256 does not match the file.")
    parser.add_argument("--skip-checksum", action="store_true",
                        help="Skip sha256 verification entirely.")

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

    # Build model list (order: --from-yaml, then --models default, then --model,
    # then back-compat single slots). --from-yaml is authoritative when given.
    models = []
    try:
        if args.from_yaml:
            models.extend(load_models_from_yaml(args.from_yaml, args.model_dir))
        if args.models:
            models.extend(DEFAULT_MODELS4)
        for mv in args.model:
            models.append(parse_model_arg(mv))
    except Exception as e:
        print(f"Error building model list: {e}")
        return 1

    if args.model_scrfd:
        models.append(make_spec("SCRFD", args.model_scrfd, MODEL_ADDRESSES["scrfd"],
                                 id="face_detection", classes=["face"]))
    if args.model_facenet:
        models.append(make_spec("FaceNet", args.model_facenet, MODEL_ADDRESSES["facenet"],
                                 id="face_embedding"))
    if args.model_yolo:
        models.append(make_spec("Person", args.model_yolo, MODEL_ADDRESSES["yolo"],
                                 id="person_detection", classes=["person"]))

    # Check what to do
    if not args.firmware and not models:
        print("Error: Specify --firmware and/or model options (--models / --from-yaml / --model)")
        print("Use --help for usage information")
        return 1
    if models and not args.firmware:
        print("Error: model flashing requires --firmware for this Himax bootloader flow.")
        print("Use --firmware --models (or --from-yaml / --model) so files ship in one session.")
        return 1

    print_header("SenseCAP Watcher Flash Tool")
    print(f"Himax port: {himax_port}")
    print(f"ESP32 port: {esp32_port or 'disabled'}")
    if models:
        print(f"Models to flash: {len(models)}")

    # Execute flashing
    success = False
    try:
        if args.firmware and models:
            success = flash_firmware_and_models(
                himax_port, esp32_port, args.firmware_path, models,
                strict_checksum=args.strict_checksum, skip_checksum=args.skip_checksum)
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
