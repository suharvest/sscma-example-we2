#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/project/grove_vision_2_remote_qat/model_zoo/tflm_face_embedding/scrfd/quantization"
exec > >(tee remote_qat_full.log) 2>&1

echo "== QAT full train/export =="
uv run python qat_scrfd_enhanced.py \
  --pth scrfd_500m_kps.pth \
  --onnx-ref scrfd_500m_kps.onnx \
  --output scrfd_qat_fixed.tflite \
  --data-dir ../../calibration_data/qat_160 \
  --num-images "${NUM_IMAGES:-12000}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-1e-4}" \
  --warmup-epochs 1 \
  --val-split 0.1 \
  --skip-vela \
  --device auto

echo "== Quality gate =="
uv run python validate_scrfd_quality.py \
  --model scrfd_qat_fixed.tflite \
  --image-dir "$HOME/project/grove_vision_2_remote_qat/model_zoo/tflm_face_embedding/calibration_data/qat_160" \
  --num-samples 1000

echo "== Optional Vela compile =="
if command -v vela >/dev/null 2>&1; then
  vela --accelerator-config ethos-u55-64 \
    --optimise Performance \
    scrfd_qat_fixed.tflite \
    --output-dir .
else
  echo "vela not found; leaving non-Vela TFLite only"
fi

ls -lh scrfd_qat_fixed*
