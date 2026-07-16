#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
export ALL_PROXY="socks5://127.0.0.1:7890"

DATASET_DIR="datasets/glint360k_balanced_200k_min4_112"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_pairft_tpair_hn_a/mfn_w1_pairft_128d.weights.h5"
OUT_DIR="official_mobilefacenet/student_distill_w1_pairft_glint_arc_bal200k_b"
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
    if [[ "$(current_images)" -lt 200000 ]]; then
        /home/harve/.local/bin/uv run python -u download_glint360k_subset.py \
            --output-dir "${DATASET_DIR}" \
            --start-shard 0 \
            --num-shards 64 \
            --max-images 200000 \
            --max-images-per-id 16 \
            --min-images-per-id 4 \
            --balanced-two-pass
    fi
    date
    echo "dataset_images=$(current_images)"
    mkdir -p "${OUT_DIR}"
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --epochs 10 \
        --batch-size 128 \
        --lr 0.000006 \
        --out-dir "${OUT_DIR}" \
        --init-weights "${INIT_WEIGHTS}" \
        --checkpoint-every 4 \
        --distill-weight 1.0 \
        --positive-weight 1.0 \
        --negative-weight 16.0 \
        --negative-margin 0.03 \
        --threshold-weight 1.2 \
        --threshold 0.08 \
        --threshold-margin 0.05 \
        --teacher-pair-weight 1.0 \
        --mine-hard-negatives 2000 \
        --arcface-weight 0.01 \
        --arcface-scale 32.0 \
        --arcface-margin 0.20 \
        --arcface-min-images 2 \
        --arcface-steps-per-epoch 100 \
        --identity-dirs "${DATASET_DIR}" \
        --max-identity-images 200000 \
        --cfp-splits 2-10 \
        --cfp-max-pairs-per-split 80 \
        --num-calib 500
    date
} > "${LOG_DIR}/s2_w1_pairft_glint_arc_bal200k_b.log" 2>&1 &

echo $!
