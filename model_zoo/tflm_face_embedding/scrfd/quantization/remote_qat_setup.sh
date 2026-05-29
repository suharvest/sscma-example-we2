#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/project/grove_vision_2_remote_qat/model_zoo/tflm_face_embedding"
exec > >(tee scrfd/quantization/remote_qat_setup.log) 2>&1

echo "== uv sync =="
pwd
uv sync

echo "== packages =="
uv run python -m pip show torch tensorflow onnxruntime
uv run python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
