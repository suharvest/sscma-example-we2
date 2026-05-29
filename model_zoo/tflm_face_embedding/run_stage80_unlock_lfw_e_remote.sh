#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/w600k_stage80_unlock_lfw_e"
LOG="logs/w600k_stage80_unlock_lfw_e.log"

mkdir -p "${OUT}" logs

{
    date
    /home/harve/.local/bin/uv run python -u train_w600k_stage96_pair_finetune.py \
        --keep-channels 80 \
        --init-weights official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.weights.h5 \
        --epochs 8 \
        --batch-size 64 \
        --lr 0.0000007 \
        --max-pairs 1200 \
        --extra-lfw-pairs 2500 \
        --cfp-splits "" \
        --hard-negative-pairs 1800 \
        --distill-weight 4.0 \
        --positive-weight 1.4 \
        --hard-positive-weight 0.0 \
        --negative-weight 12.0 \
        --negative-margin 0.018 \
        --teacher-pair-weight 4.0 \
        --teacher-backend keras \
        --out-dir "${OUT}" \
        --num-calib 500
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 1000 \
        --cfp-splits "" \
        --model dense128=official_mobilefacenet/w600k_dense128/w600k_dense128.int8.tflite \
        --model unlock_a=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.int8.tflite \
        --model unlock_d=official_mobilefacenet/w600k_stage80_unlock_lfw_d/w600k_stage96_pairft.int8.tflite \
        --model unlock_e="${OUT}/w600k_stage96_pairft.int8.tflite"
    date
} > "${LOG}" 2>&1
