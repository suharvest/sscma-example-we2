#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/iddistill_v2_lfw9"
LOG="logs/iddistill_v2_lfw9.log"
DATASET="datasets/glint360k_balanced_200k_min4_112"
COMMON_CACHE="official_mobilefacenet/student_distill_w1_pairft_glint_iddistill50k_a"

mkdir -p "${OUT}" logs

{
    date
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --embedding-dim 128 \
        --epochs 6 \
        --batch-size 128 \
        --lr 0.0000012 \
        --out-dir "${OUT}" \
        --init-weights official_mobilefacenet/iddistill_v2_lfw8/mfn_w1_pairft_128d.weights.h5 \
        --checkpoint-every 3 \
        --distill-weight 4.0 \
        --positive-weight 3.0 \
        --negative-weight 9.0 \
        --negative-margin 0.022 \
        --threshold-weight 0.9 \
        --threshold 0.165 \
        --threshold-margin 0.055 \
        --teacher-pair-weight 3.0 \
        --hard-positive-fraction 0.55 \
        --hard-negative-fraction 0.30 \
        --mine-hard-positives 600 \
        --mine-hard-negatives 2200 \
        --mine-teacher-hard-negatives 900 \
        --arcface-weight 0.005 \
        --arcface-scale 32.0 \
        --arcface-margin 0.16 \
        --arcface-min-images 2 \
        --arcface-steps-per-epoch 50 \
        --identity-distill-weight 0.45 \
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
        --model v2lfw6=official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.int8.tflite \
        --model v2lfw7=official_mobilefacenet/iddistill_v2_lfw7/mfn_w1_pairft_128d.int8.tflite \
        --model v2lfw8=official_mobilefacenet/iddistill_v2_lfw8/mfn_w1_pairft_128d.int8.tflite \
        --model v2lfw9="${OUT}/mfn_w1_pairft_128d.int8.tflite"
    date
} > "${LOG}" 2>&1

mv scripts/face_embedding_next.sh "scripts/face_embedding_next.done.$(date +%Y%m%d_%H%M%S).sh" 2>/dev/null || true
