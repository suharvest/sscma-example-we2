#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/w600k_stage80_unlock_lfw_d"
LOG="logs/w600k_stage80_unlock_lfw_d.log"

mkdir -p "${OUT}" logs
cp -n official_mobilefacenet/w600k_stage80_unlock_lfw_a/teacher_keras_dense128_pairs_4023.npz \
    "${OUT}/teacher_keras_dense128_pairs_4023.npz" 2>/dev/null || true

{
    date
    /home/harve/.local/bin/uv run python -u train_w600k_stage96_pair_finetune.py \
        --keep-channels 80 \
        --init-weights official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.weights.h5 \
        --epochs 6 \
        --batch-size 64 \
        --lr 0.0000007 \
        --max-pairs 1200 \
        --cfp-splits "" \
        --hard-negative-pairs 2200 \
        --distill-weight 4.5 \
        --positive-weight 1.3 \
        --hard-positive-weight 7.0 \
        --positive-margin 0.34 \
        --negative-weight 11.0 \
        --negative-margin 0.018 \
        --teacher-pair-weight 4.5 \
        --teacher-backend keras \
        --aligned-cache official_mobilefacenet/w600k_stage80_pairft_a/aligned_pair_images.npz \
        --out-dir "${OUT}" \
        --num-calib 500
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 120 \
        --cfp-splits 1-10 \
        --model dense128=official_mobilefacenet/w600k_dense128/w600k_dense128.int8.tflite \
        --model unlock_a=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.int8.tflite \
        --model unlock_b=official_mobilefacenet/w600k_stage80_unlock_lfw_b/w600k_stage96_pairft.int8.tflite \
        --model unlock_c=official_mobilefacenet/w600k_stage80_unlock_lfw_c/w600k_stage96_pairft.int8.tflite \
        --model unlock_d="${OUT}/w600k_stage96_pairft.int8.tflite"
    date
} > "${LOG}" 2>&1
