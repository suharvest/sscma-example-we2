#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/project/grove_vision_2_remote_qat/model_zoo/tflm_face_embedding"
LOG_FILE="$PWD/scrfd/quantization/remote_qat_smoke.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "== Environment =="
pwd
uv run python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
uv run python -c 'import tensorflow as tf; print("tf", tf.__version__)'
uv run python -c 'import onnxruntime as ort; print("ort", ort.__version__, ort.get_available_providers())'

cd scrfd/quantization

echo "== QAT smoke train/export =="
uv run python qat_scrfd_enhanced.py \
  --pth scrfd_500m_kps.pth \
  --onnx-ref scrfd_500m_kps.onnx \
  --output scrfd_qat_fixed_smoke.tflite \
  --data-dir ../../calibration_data/qat_160 \
  --num-images "${NUM_IMAGES:-512}" \
  --epochs "${EPOCHS:-1}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lr "${LR:-1e-4}" \
  --warmup-epochs 0 \
  --val-split 0.1 \
  --skip-vela \
  --device auto

echo "== Quality gate =="
uv run python validate_scrfd_quality.py \
  --model scrfd_qat_fixed_smoke.tflite \
  --image-dir "$HOME/project/grove_vision_2_remote_qat/model_zoo/tflm_face_embedding/calibration_data/qat_160" \
  --num-samples 200

ls -lh scrfd_qat_fixed_smoke.*
