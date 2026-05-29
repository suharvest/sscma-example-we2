#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/w600k_stage80_unlock_lfw_b"
LOG="logs/w600k_stage80_unlock_lfw_b.log"

mkdir -p "${OUT}" logs

{
    date
    /home/harve/.local/bin/uv run python -u train_w600k_stage96_pair_finetune.py \
        --keep-channels 80 \
        --init-weights official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.weights.h5 \
        --epochs 6 \
        --batch-size 64 \
        --lr 0.0000008 \
        --max-pairs 1200 \
        --cfp-splits "" \
        --hard-negative-pairs 4200 \
        --distill-weight 4.2 \
        --positive-weight 1.5 \
        --negative-weight 16.0 \
        --negative-margin 0.012 \
        --teacher-pair-weight 4.2 \
        --teacher-backend keras \
        --aligned-cache official_mobilefacenet/w600k_stage80_pairft_a/aligned_pair_images.npz \
        --out-dir "${OUT}" \
        --num-calib 500
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 120 \
        --cfp-splits 1-10 \
        --model dense128=official_mobilefacenet/w600k_dense128/w600k_dense128.int8.tflite \
        --model stage80_a=official_mobilefacenet/w600k_stage80_pairft_a/w600k_stage80_pairft.int8.tflite \
        --model unlock_a=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.int8.tflite \
        --model unlock_b="${OUT}/w600k_stage96_pairft.int8.tflite"
    date
} > "${LOG}" 2>&1
