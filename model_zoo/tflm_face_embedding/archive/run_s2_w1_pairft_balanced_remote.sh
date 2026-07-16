#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT_DIR="official_mobilefacenet/student_distill_w1_pairft_balanced"
SRC_DIR="official_mobilefacenet/student_distill_w1_pairft"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/s2_w1_pairft_balanced_e16.log"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_hardneg/mfn_w1_distill_128d.weights.h5"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
for suffix in aligned_lfw.npz teacher512_3437.npz; do
    src="${SRC_DIR}/mfn_w1_pairft_128d_${suffix}"
    dst="${OUT_DIR}/mfn_w1_pairft_128d_${suffix}"
    if [[ -f "${src}" && ! -f "${dst}" ]]; then
        cp "${src}" "${dst}"
    fi
done

nohup /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
    --width 1.0 \
    --epochs 16 \
    --batch-size 128 \
    --lr 0.00008 \
    --out-dir "${OUT_DIR}" \
    --init-weights "${INIT_WEIGHTS}" \
    --checkpoint-every 4 \
    --distill-weight 1.0 \
    --positive-weight 0.8 \
    --negative-weight 12.0 \
    --negative-margin 0.03 \
    --num-calib 500 \
    > "${LOG_FILE}" 2>&1 &

echo $!
