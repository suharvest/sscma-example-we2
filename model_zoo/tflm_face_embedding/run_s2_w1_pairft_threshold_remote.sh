#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

SRC_CACHE_DIR="official_mobilefacenet/student_distill_w1_pairft_cfp_fb_a"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_pairft_cfp_fb_b/mfn_w1_pairft_128d.weights.h5"
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
    local threshold_weight="$3"
    local threshold="$4"
    local threshold_margin="$5"

    copy_cache "${out_dir}"
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --epochs 6 \
        --batch-size 128 \
        --lr 0.00002 \
        --out-dir "${out_dir}" \
        --init-weights "${INIT_WEIGHTS}" \
        --checkpoint-every 3 \
        --distill-weight 0.8 \
        --positive-weight 1.0 \
        --negative-weight 12.0 \
        --negative-margin 0.03 \
        --threshold-weight "${threshold_weight}" \
        --threshold "${threshold}" \
        --threshold-margin "${threshold_margin}" \
        --cfp-splits 2-10 \
        --cfp-max-pairs-per-split 80 \
        --num-calib 500 \
        > "${log_file}" 2>&1
}

{
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_thr_a" \
        "${LOG_DIR}/s2_w1_pairft_thr_a_e6.log" \
        0.5 0.02 0.04
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_thr_b" \
        "${LOG_DIR}/s2_w1_pairft_thr_b_e6.log" \
        1.0 0.02 0.04
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_thr_c" \
        "${LOG_DIR}/s2_w1_pairft_thr_c_e6.log" \
        0.75 0.04 0.04
    date
} > "${LOG_DIR}/s2_w1_pairft_threshold_sweep.log" 2>&1 &

echo $!
