#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

SRC="official_mobilefacenet/iddistill_v2_w115_lfw1"
OUT="official_mobilefacenet/iddistill_v2_w115_repair1"
DATASET="datasets/glint360k_balanced_200k_min4_112"
COMMON_CACHE="official_mobilefacenet/student_distill_w1_pairft_glint_iddistill50k_a"
LOG_DIR="logs"

mkdir -p "${OUT}" "${LOG_DIR}"

{
    date
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.15 \
        --embedding-dim 128 \
        --epochs 8 \
        --batch-size 128 \
        --lr 0.000003 \
        --out-dir "${OUT}" \
        --init-weights "${SRC}/mfn_w1.15_pairft_128d.weights.h5" \
        --checkpoint-every 4 \
        --distill-weight 1.8 \
        --positive-weight 1.0 \
        --negative-weight 24.0 \
        --negative-margin 0.02 \
        --threshold-weight 2.0 \
        --threshold 0.20 \
        --threshold-margin 0.08 \
        --teacher-pair-weight 1.0 \
        --hard-positive-fraction 0.35 \
        --hard-negative-fraction 0.35 \
        --mine-hard-positives 300 \
        --mine-hard-negatives 3500 \
        --mine-teacher-hard-negatives 1500 \
        --arcface-weight 0.004 \
        --arcface-scale 32.0 \
        --arcface-margin 0.18 \
        --arcface-min-images 2 \
        --arcface-steps-per-epoch 50 \
        --identity-distill-weight 0.25 \
        --identity-dirs "${DATASET}" \
        --max-identity-images 50000 \
        --aligned-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_aligned_lfw.npz" \
        --teacher-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_teacher512_5103.npz" \
        --identity-target-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_identity_targets_29651.npz" \
        --arcface-head-in "${SRC}/mfn_w1.15_pairft_128d_arcface_head.npz" \
        --arcface-head-out "${OUT}/mfn_w1.15_pairft_128d_arcface_head.npz" \
        --cfp-splits "" \
        --num-calib 500
    date
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 120 \
        --cfp-splits 1-10 \
        --model v2lfw6=official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.int8.tflite \
        --model w115=official_mobilefacenet/iddistill_v2_w115_lfw1/mfn_w1.15_pairft_128d.int8.tflite \
        --model repair=official_mobilefacenet/iddistill_v2_w115_repair1/mfn_w1.15_pairft_128d.int8.tflite
    date
} > "${LOG_DIR}/iddistill_v2_w115_repair1.log" 2>&1 &

echo $!
