#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/iddistill_v2_lfw13_lfwfirst"
LOG="logs/iddistill_v2_lfw13_lfwfirst.log"
DATASET="datasets/glint360k_balanced_200k_min4_112"
COMMON_CACHE="official_mobilefacenet/student_distill_w1_pairft_glint_iddistill50k_a"

mkdir -p "${OUT}" logs

{
    date
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --embedding-dim 128 \
        --epochs 8 \
        --batch-size 128 \
        --lr 0.0000010 \
        --out-dir "${OUT}" \
        --init-weights official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.weights.h5 \
        --checkpoint-every 4 \
        --distill-weight 5.0 \
        --positive-weight 5.0 \
        --negative-weight 16.0 \
        --negative-margin 0.012 \
        --threshold-weight 1.4 \
        --threshold 0.18 \
        --threshold-margin 0.08 \
        --teacher-pair-weight 4.0 \
        --hard-positive-fraction 0.70 \
        --hard-negative-fraction 0.50 \
        --mine-hard-positives 900 \
        --mine-hard-negatives 4000 \
        --mine-teacher-hard-negatives 1200 \
        --arcface-weight 0.003 \
        --arcface-scale 32.0 \
        --arcface-margin 0.14 \
        --arcface-min-images 2 \
        --arcface-steps-per-epoch 35 \
        --identity-distill-weight 0.55 \
        --identity-dirs "${DATASET}" \
        --max-identity-images 50000 \
        --aligned-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_aligned_lfw.npz" \
        --teacher-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_teacher512_5103.npz" \
        --identity-target-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_identity_targets_29651.npz" \
        --cfp-splits "" \
        --num-calib 500
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 120 \
        --cfp-splits 1-10 \
        --model w600k=official_mobilefacenet/w600k_mbf_int8.tflite \
        --model v2lfw6=official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.int8.tflite \
        --model v2lfw10=official_mobilefacenet/iddistill_v2_lfw10/mfn_w1_pairft_128d.int8.tflite \
        --model v2lfw13="${OUT}/mfn_w1_pairft_128d.int8.tflite"
    date
} > "${LOG}" 2>&1
