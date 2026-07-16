#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

BASE="official_mobilefacenet/iddistill_w115_base"
OUT="official_mobilefacenet/iddistill_v2_w115_lfw1"
DATASET="datasets/glint360k_balanced_200k_min4_112"
COMMON_CACHE="official_mobilefacenet/student_distill_w1_pairft_glint_iddistill50k_a"
LOG_DIR="logs"

mkdir -p "${BASE}" "${OUT}" "${LOG_DIR}"

if [[ ! -f "${BASE}/mfn_w1.15_distill_128d_teacher512_11901.npz" ]]; then
    cp official_mobilefacenet/student_distill_w1_margin/mfn_w1_distill_128d_teacher512_11901.npz \
        "${BASE}/mfn_w1.15_distill_128d_teacher512_11901.npz"
fi

{
    date
    /home/harve/.local/bin/uv run python -u train_mfn_student_distill.py \
        --width 1.15 \
        --embedding-dim 128 \
        --num-train 12000 \
        --epochs 50 \
        --batch-size 128 \
        --lr 0.001 \
        --out-dir "${BASE}" \
        --checkpoint-every 10 \
        --target-weight 1.0 \
        --pair-weight 4.0 \
        --negative-weight 4.0 \
        --negative-margin 0.12 \
        --negative-teacher-threshold 0.30 \
        --augment \
        --num-calib 500
    date
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.15 \
        --embedding-dim 128 \
        --epochs 10 \
        --batch-size 128 \
        --lr 0.000004 \
        --out-dir "${OUT}" \
        --init-weights "${BASE}/mfn_w1.15_distill_128d.weights.h5" \
        --checkpoint-every 5 \
        --distill-weight 2.5 \
        --positive-weight 4.8 \
        --negative-weight 5.0 \
        --negative-margin 0.03 \
        --threshold-weight 0.35 \
        --threshold 0.18 \
        --threshold-margin 0.05 \
        --teacher-pair-weight 2.0 \
        --hard-positive-fraction 0.8 \
        --hard-negative-fraction 0.08 \
        --mine-hard-positives 800 \
        --mine-hard-negatives 1200 \
        --mine-teacher-hard-negatives 0 \
        --arcface-weight 0.012 \
        --arcface-scale 32.0 \
        --arcface-margin 0.20 \
        --arcface-min-images 2 \
        --arcface-steps-per-epoch 80 \
        --identity-distill-weight 0.70 \
        --identity-dirs "${DATASET}" \
        --max-identity-images 50000 \
        --aligned-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_aligned_lfw.npz" \
        --teacher-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_teacher512_5103.npz" \
        --identity-target-cache "${COMMON_CACHE}/mfn_w1_pairft_128d_identity_targets_29651.npz" \
        --cfp-splits "" \
        --num-calib 500
    date
    vela "${OUT}/mfn_w1.15_pairft_128d.int8.tflite" \
        --accelerator-config ethos-u55-64 \
        --optimise Performance
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 120 \
        --cfp-splits 1-10 \
        --model w600k=official_mobilefacenet/w600k_mbf_int8.tflite \
        --model v2lfw6=official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.int8.tflite \
        --model w115=official_mobilefacenet/iddistill_v2_w115_lfw1/mfn_w1.15_pairft_128d.int8.tflite
    date
} > "${LOG_DIR}/iddistill_v2_w115_lfw1.log" 2>&1 &

echo $!
