#!/bin/bash
# Flash Himax firmware while keeping ESP32 in reset
# This prevents ESP32 from resetting Himax during flashing

set -e

ESP32_PORT="/dev/cu.wchusbserial5AF91659653"
HIMAX_PORT="/dev/cu.usbmodem5AF91659651"
FIRMWARE="we2_image_gen_local/output_case1_sec_wlcsp/output.img"

echo "=== Himax Firmware Flasher ==="
echo "ESP32 Port: $ESP32_PORT"
echo "Himax Port: $HIMAX_PORT"
echo "Firmware: $FIRMWARE"
echo ""

# Check if ports exist
if [ ! -e "$ESP32_PORT" ]; then
    echo "ERROR: ESP32 port not found: $ESP32_PORT"
    exit 1
fi

if [ ! -e "$HIMAX_PORT" ]; then
    echo "ERROR: Himax port not found: $HIMAX_PORT"
    exit 1
fi

if [ ! -f "$FIRMWARE" ]; then
    echo "ERROR: Firmware not found: $FIRMWARE"
    exit 1
fi

echo "Step 1: Put ESP32 into download mode (hold reset)..."
# Use esptool to put ESP32 into download mode and keep it there
export IDF_PYTHON_ENV_PATH=/Users/harvest/.espressif/python_env/idf5.5_py3.14_env
source /Users/harvest/esp/esp-idf/export.sh > /dev/null 2>&1

# This command enters download mode but doesn't reset after
esptool.py --port "$ESP32_PORT" --before default_reset --after no_reset chip_id &
ESPTOOL_PID=$!
sleep 2

echo "Step 2: Flash Himax firmware..."
source /tmp/sscma_flash_env/bin/activate 2>/dev/null || {
    echo "Creating sscma environment..."
    cd /tmp
    uv venv sscma_flash_env
    source sscma_flash_env/bin/activate
    uv pip install python-sscma "numpy<2"
}

echo ""
echo ">>> Press Himax RESET button NOW! <<<"
echo ""

sscma.cli flasher -p "$HIMAX_PORT" -f "$FIRMWARE"

echo ""
echo "Step 3: Reset ESP32..."
# Kill esptool background process
kill $ESPTOOL_PID 2>/dev/null || true

# Reset ESP32
esptool.py --port "$ESP32_PORT" --after hard_reset chip_id > /dev/null 2>&1 || true

echo "Done! Both chips should be running now."
