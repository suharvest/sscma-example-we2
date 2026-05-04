#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT_DIR="official_mobilefacenet/student_distill_w1_pairft"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/s2_w1_pairft_e32.log"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_hardneg/mfn_w1_distill_128d.weights.h5"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

nohup /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
    --width 1.0 \
    --epochs 32 \
    --batch-size 128 \
    --lr 0.0001 \
    --out-dir "${OUT_DIR}" \
    --init-weights "${INIT_WEIGHTS}" \
    --checkpoint-every 4 \
    --distill-weight 0.3 \
    --positive-weight 2.0 \
    --negative-weight 8.0 \
    --negative-margin 0.05 \
    --num-calib 500 \
    > "${LOG_FILE}" 2>&1 &

echo $!
