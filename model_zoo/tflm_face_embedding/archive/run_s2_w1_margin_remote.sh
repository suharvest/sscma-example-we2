#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT_DIR="official_mobilefacenet/student_distill_w1_margin"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/s2_w1_margin_e60.log"
SRC_CACHE="official_mobilefacenet/student_distill/mfn_w0.75_distill_128d_teacher512_11901.npz"
DST_CACHE="${OUT_DIR}/mfn_w1_distill_128d_teacher512_11901.npz"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
if [[ -f "${SRC_CACHE}" && ! -f "${DST_CACHE}" ]]; then
    cp "${SRC_CACHE}" "${DST_CACHE}"
fi

nohup /home/harve/.local/bin/uv run python -u train_mfn_student_distill.py \
    --width 1.0 \
    --num-train 12000 \
    --epochs 60 \
    --batch-size 128 \
    --num-calib 500 \
    --out-dir "${OUT_DIR}" \
    --checkpoint-every 5 \
    --pair-weight 8 \
    --negative-weight 12 \
    --negative-margin 0.08 \
    --augment \
    > "${LOG_FILE}" 2>&1 &

echo $!
