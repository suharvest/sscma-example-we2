#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

DATASET_DIR="datasets/glint360k_subset_112"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_pairft_tpair_hn_a/mfn_w1_pairft_128d.weights.h5"
OUT_DIR="official_mobilefacenet/student_distill_w1_pairft_glint_arc_c"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

current_images() {
    if [[ -d "${DATASET_DIR}" ]]; then
        find "${DATASET_DIR}" -type f -name '*.jpg' | wc -l
    else
        echo 0
    fi
}

{
    date
    if [[ "$(current_images)" -lt 50000 ]]; then
        /home/harve/.local/bin/uv run python -u download_glint360k_subset.py \
            --output-dir "${DATASET_DIR}" \
            --start-shard 0 \
            --num-shards 8 \
            --max-images 50000 \
            --max-images-per-id 20
    fi
    date
    echo "dataset_images=$(current_images)"
    mkdir -p "${OUT_DIR}"
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --epochs 8 \
        --batch-size 128 \
        --lr 0.000008 \
        --out-dir "${OUT_DIR}" \
        --init-weights "${INIT_WEIGHTS}" \
        --checkpoint-every 4 \
        --distill-weight 0.8 \
        --positive-weight 1.0 \
        --negative-weight 12.0 \
        --negative-margin 0.03 \
        --threshold-weight 0.5 \
        --threshold 0.02 \
        --threshold-margin 0.04 \
        --teacher-pair-weight 0.5 \
        --mine-hard-negatives 1200 \
        --arcface-weight 0.05 \
        --arcface-scale 32.0 \
        --arcface-margin 0.25 \
        --arcface-min-images 2 \
        --arcface-steps-per-epoch 100 \
        --identity-dirs "${DATASET_DIR}" \
        --max-identity-images 10000 \
        --cfp-splits 2-10 \
        --cfp-max-pairs-per-split 80 \
        --num-calib 500
    date
} > "${LOG_DIR}/s2_w1_pairft_glint_arc_c.log" 2>&1 &

echo $!
