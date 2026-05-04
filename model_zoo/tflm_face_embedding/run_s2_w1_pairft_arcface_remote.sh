#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

SRC_CACHE_DIR="official_mobilefacenet/student_distill_w1_pairft_cfp_fb_a"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_pairft_tpair_hn_a/mfn_w1_pairft_128d.weights.h5"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

copy_cache() {
    local dst_dir="$1"
    mkdir -p "${dst_dir}"
    for suffix in aligned_lfw.npz teacher512_5103.npz; do
        local src="${SRC_CACHE_DIR}/mfn_w1_pairft_128d_${suffix}"
        local dst="${dst_dir}/mfn_w1_pairft_128d_${suffix}"
        if [[ -f "${src}" && ! -f "${dst}" ]]; then
            cp "${src}" "${dst}"
        fi
    done
}

run_variant() {
    local out_dir="$1"
    local log_file="$2"
    local arcface_weight="$3"
    local arcface_margin="$4"

    copy_cache "${out_dir}"
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --epochs 8 \
        --batch-size 128 \
        --lr 0.00001 \
        --out-dir "${out_dir}" \
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
        --arcface-weight "${arcface_weight}" \
        --arcface-scale 32.0 \
        --arcface-margin "${arcface_margin}" \
        --arcface-min-images 2 \
        --cfp-splits 2-10 \
        --cfp-max-pairs-per-split 80 \
        --num-calib 500 \
        > "${log_file}" 2>&1
}

{
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_arc_a" \
        "${LOG_DIR}/s2_w1_pairft_arc_a_e8.log" \
        0.05 0.25
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_arc_b" \
        "${LOG_DIR}/s2_w1_pairft_arc_b_e8.log" \
        0.10 0.35
    date
} > "${LOG_DIR}/s2_w1_pairft_arcface_sweep.log" 2>&1 &

echo $!
