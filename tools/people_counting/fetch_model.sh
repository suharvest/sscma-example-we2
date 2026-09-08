#!/bin/sh
# 下载 Swift-YOLO Nano person 检测模型并补 padding。
#
# 为什么要补 padding：xmodem 刷模型时稳定卡在距文件末尾约 6KB（47~48 个 128B 块）处，
# 同一文件跨运行可复现，且即使该模型不在 session 末位也照卡；固件传输从不出现。
# 补 16KB 零字节后真实数据在卡死点之前传完，模型加载按 header 里的 size 走，忽略尾部多余字节。
#
# 用法: sh tools/people_counting/fetch_model.sh
set -e
URL="https://files.seeedstudio.com/sscma/model_zoo/detection/person/swift_yolo_nano_person_192_int8_vela.tflite"
OUT_DIR="$(dirname "$0")/../../model_zoo"
RAW="$OUT_DIR/swift_yolo_nano_person_192_int8_vela.tflite"
PADDED="$OUT_DIR/swift_yolo_person_192_padded.tflite"

echo "下载 $URL"
curl -sSL -o "$RAW" "$URL"
echo "原始: $(wc -c < "$RAW") B"

cp "$RAW" "$PADDED"
dd if=/dev/zero bs=1024 count=16 >> "$PADDED" 2>/dev/null
echo "补后: $(wc -c < "$PADDED") B  ->  $PADDED"
echo
echo "烧录地址 0x700000，例："
echo '  --model="model_zoo/swift_yolo_person_192_padded.tflite 0x700000 0x00000"'
