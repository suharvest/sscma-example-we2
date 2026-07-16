#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT_DIR="official_mobilefacenet/student_distill_w1_hardneg"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/s2_w1_hardneg_e24.log"
SRC_CACHE="official_mobilefacenet/student_distill_w1_margin/mfn_w1_distill_128d_teacher512_11901.npz"
DST_CACHE="${OUT_DIR}/mfn_w1_distill_128d_teacher512_11901.npz"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_margin/mfn_w1_distill_128d.weights.h5"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
if [[ -f "${SRC_CACHE}" && ! -f "${DST_CACHE}" ]]; then
    cp "${SRC_CACHE}" "${DST_CACHE}"
fi

nohup /home/harve/.local/bin/uv run python -u train_mfn_student_distill.py \
    --width 1.0 \
    --num-train 12000 \
    --epochs 24 \
    --batch-size 128 \
    --lr 0.0002 \
    --num-calib 500 \
    --out-dir "${OUT_DIR}" \
    --init-weights "${INIT_WEIGHTS}" \
    --checkpoint-every 4 \
    --pair-weight 16 \
    --negative-weight 32 \
    --negative-margin 0.0 \
    --negative-teacher-threshold 0.20 \
    > "${LOG_FILE}" 2>&1 &

echo $!
