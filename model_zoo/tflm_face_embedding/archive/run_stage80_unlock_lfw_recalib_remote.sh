#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/w600k_stage80_unlock_lfw_recalib"
LOG="logs/w600k_stage80_unlock_lfw_recalib.log"

mkdir -p "${OUT}" logs

{
    date
    /home/harve/.local/bin/uv run python -u - <<'PY'
from pathlib import Path

import numpy as np

from train_mfn_student_distill import export_tflite
from train_w600k_stage96_distill import build_model

out = Path("official_mobilefacenet/w600k_stage80_unlock_lfw_recalib")
out.mkdir(parents=True, exist_ok=True)

model = build_model(keep_channels=80)
model.load_weights("official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.weights.h5")

cache = np.load("official_mobilefacenet/w600k_stage80_pairft_a/aligned_pair_images.npz")
images = cache["images"].astype(np.uint8)
print(f"Loaded recalibration images: {len(images)}")
export_tflite(model, images, out / "w600k_stage96_pairft", min(2500, len(images)))
PY
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 1000 \
        --cfp-splits "" \
        --model dense128=official_mobilefacenet/w600k_dense128/w600k_dense128.int8.tflite \
        --model unlock_a=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.int8.tflite \
        --model unlock_a_float=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.float32.tflite \
        --model recalib="${OUT}/w600k_stage96_pairft.int8.tflite"
    date
} > "${LOG}" 2>&1
