#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

OUT="official_mobilefacenet/w600k_stage80_we2friendly_leaky_a"
LOG="logs/w600k_stage80_we2friendly_leaky_a.log"

mkdir -p "${OUT}" logs

{
    date
    /home/harve/.local/bin/uv run python -u train_w600k_stage96_pair_finetune.py \
        --keep-channels 80 \
        --activation-override suspect-leaky \
        --init-weights official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.weights.h5 \
        --epochs 8 \
        --batch-size 64 \
        --lr 0.0000015 \
        --max-pairs 1200 \
        --cfp-splits "" \
        --hard-negative-pairs 2600 \
        --distill-weight 4.0 \
        --positive-weight 1.2 \
        --negative-weight 14.0 \
        --negative-margin 0.018 \
        --teacher-pair-weight 4.0 \
        --teacher-backend keras \
        --aligned-cache official_mobilefacenet/w600k_stage80_pairft_a/aligned_pair_images.npz \
        --out-dir "${OUT}" \
        --num-calib 500
    cp -f "${OUT}/w600k_stage96_pairft.int8.tflite" "${OUT}/w600k_stage80_we2friendly_leaky_a.int8.tflite"
    if command -v vela >/dev/null 2>&1; then
        vela "${OUT}/w600k_stage80_we2friendly_leaky_a.int8.tflite" \
            --accelerator-config ethos-u55-64 \
            --optimise Performance \
            --output-dir "${OUT}/vela"
    fi
    /home/harve/.local/bin/uv run python -u evaluate_embedding_models.py \
        --max-pairs 120 \
        --cfp-splits 1-10 \
        --model dense128=official_mobilefacenet/w600k_dense128/w600k_dense128.int8.tflite \
        --model unlock_a=official_mobilefacenet/w600k_stage80_unlock_lfw_a/w600k_stage96_pairft.int8.tflite \
        --model we2_leaky="${OUT}/w600k_stage80_we2friendly_leaky_a.int8.tflite" \
        --model v2lfw6=official_mobilefacenet/iddistill_v2_lfw6/mfn_w1_pairft_128d.int8.tflite
    date
} > "${LOG}" 2>&1
