#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

DISTILL_OUT="official_mobilefacenet/w600k_stage80_lfw_distill_a"
PAIR_OUT="official_mobilefacenet/w600k_stage80_unlock_lfw_f"
LOG="logs/w600k_stage80_unlock_lfw_f.log"

mkdir -p "${DISTILL_OUT}" "${PAIR_OUT}" logs

{
    date
    /home/harve/.local/bin/uv run python -u train_w600k_stage96_distill.py \
        --keep-channels 80 \
        --num-train 0 \
        --identity-dirs datasets/lfw \
        --epochs 10 \
        --batch-size 64 \
        --lr 0.000008 \
        --pair-weight 3.0 \
        --teacher-backend keras \
        --out-dir "${DISTILL_OUT}" \
        --num-calib 1000
    /home/harve/.local/bin/uv run python -u train_w600k_stage96_pair_finetune.py \
        --keep-channels 80 \
        --init-weights "${DISTILL_OUT}/w600k_stage96_distill.weights.h5" \
        --epochs 8 \
        --batch-size 64 \
        --lr 0.0000015 \
        --max-pairs 1200 \
        --cfp-splits "" \
        --hard-negative-pairs 2600 \
        --distill-weight 3.5 \
        --positive-weight 1.2 \
        --negative-weight 14.0 \
        --negative-margin 0.018 \
        --teacher-pair-weight 3.5 \
        --teacher-backend keras \
        --aligned-cache official_mobilefacenet/w600k_stage80_pairft_a/aligned_pair_images.npz \
        --out-dir "${PAIR_OUT}" \
        --num-calib 1000
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 1000 \
        --cfp-splits "" \
        --model dense128=official_mobilefacenet/w600k_dense128/w600k_dense128.int8.tflite \
        --model unlock_a=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.int8.tflite \
        --model distill_a="${DISTILL_OUT}/w600k_stage96_distill.int8.tflite" \
        --model unlock_f="${PAIR_OUT}/w600k_stage96_pairft.int8.tflite"
    date
} > "${LOG}" 2>&1
