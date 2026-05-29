#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/iddistill_v2_lfw10"
LOG="logs/iddistill_v2_lfw10.log"
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
        --lr 0.0000014 \
        --out-dir "${OUT}" \
        --init-weights official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.weights.h5 \
        --checkpoint-every 3 \
        --distill-weight 3.6 \
        --positive-weight 2.4 \
        --negative-weight 11.5 \
        --negative-margin 0.024 \
        --threshold-weight 1.0 \
        --threshold 0.165 \
        --threshold-margin 0.055 \
        --teacher-pair-weight 2.6 \
        --hard-positive-fraction 0.38 \
        --hard-negative-fraction 0.34 \
        --mine-hard-positives 350 \
        --mine-hard-negatives 2400 \
        --mine-teacher-hard-negatives 700 \
        --arcface-weight 0.0045 \
        --arcface-scale 32.0 \
        --arcface-margin 0.16 \
        --arcface-min-images 2 \
        --arcface-steps-per-epoch 45 \
        --identity-distill-weight 0.42 \
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
        --model v2lfw9=official_mobilefacenet/iddistill_v2_lfw9/mfn_w1_pairft_128d.int8.tflite \
        --model v2lfw10="${OUT}/mfn_w1_pairft_128d.int8.tflite"
    date
} > "${LOG}" 2>&1
