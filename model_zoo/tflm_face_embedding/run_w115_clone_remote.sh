#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/iddistill_w115_v2lfw6_clone"
LOG_DIR="logs"
mkdir -p "${OUT}" "${LOG_DIR}"

{
    date
    /home/harve/.local/bin/uv run python -u train_mfn_student_from_tflite128.py \
        --width 1.15 \
        --embedding-dim 128 \
        --teacher-model official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.int8.tflite \
        --num-train 12000 \
        --epochs 40 \
        --batch-size 128 \
        --lr 0.001 \
        --out-dir "${OUT}" \
        --checkpoint-every 10 \
        --target-weight 1.0 \
        --pair-weight 6.0 \
        --augment \
        --num-calib 500
    date
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 120 \
        --cfp-splits 1-10 \
        --model v2lfw6=official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.int8.tflite \
        --model clone=official_mobilefacenet/iddistill_w115_v2lfw6_clone/mfn_w1.15_clone_128d.int8.tflite
    date
} > "${LOG_DIR}/iddistill_w115_v2lfw6_clone.log" 2>&1 &

echo $!
