#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/w600k_stage80_unlock_lfw_qat_a"
LOG="logs/w600k_stage80_unlock_lfw_qat_a.log"

mkdir -p "${OUT}" logs
cp -n official_mobilefacenet/w600k_stage80_unlock_lfw_a/teacher_keras_dense128_pairs_4023.npz \
    "${OUT}/teacher_keras_dense128_pairs_4023.npz" 2>/dev/null || true

{
    date
    /home/harve/.local/bin/uv run python -u train_w600k_stage96_pair_finetune.py \
        --qat \
        --keep-channels 80 \
        --init-weights official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.weights.h5 \
        --epochs 8 \
        --batch-size 64 \
        --lr 0.000001 \
        --max-pairs 1200 \
        --cfp-splits "" \
        --hard-negative-pairs 0 \
        --distill-weight 4.0 \
        --positive-weight 1.2 \
        --negative-weight 12.0 \
        --negative-margin 0.018 \
        --teacher-pair-weight 4.0 \
        --teacher-backend keras \
        --aligned-cache official_mobilefacenet/w600k_stage80_pairft_a/aligned_pair_images.npz \
        --out-dir "${OUT}" \
        --num-calib 500
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 1000 \
        --cfp-splits "" \
        --model dense128=official_mobilefacenet/w600k_dense128/w600k_dense128.int8.tflite \
        --model unlock_a=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.int8.tflite \
        --model unlock_a_float=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.float32.tflite \
        --model qat_a="${OUT}/w600k_stage96_pairft.int8.tflite"
    date
} > "${LOG}" 2>&1
