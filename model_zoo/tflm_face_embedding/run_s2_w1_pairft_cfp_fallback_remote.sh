#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_pairft_balanced/mfn_w1_pairft_128d.weights.h5"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

copy_cache() {
    local src_dir="$1"
    local dst_dir="$2"
    mkdir -p "${dst_dir}"
    for suffix in aligned_lfw.npz teacher512_5103.npz; do
        local src="${src_dir}/mfn_w1_pairft_128d_${suffix}"
        local dst="${dst_dir}/mfn_w1_pairft_128d_${suffix}"
        if [[ -f "${src}" && ! -f "${dst}" ]]; then
            cp "${src}" "${dst}"
        fi
    done
}

run_variant() {
    local out_dir="$1"
    local log_file="$2"
    local distill_weight="$3"
    local positive_weight="$4"
    local negative_weight="$5"
    local negative_margin="$6"

    mkdir -p "${out_dir}"
    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --epochs 8 \
        --batch-size 128 \
        --lr 0.00003 \
        --out-dir "${out_dir}" \
        --init-weights "${INIT_WEIGHTS}" \
        --checkpoint-every 4 \
        --distill-weight "${distill_weight}" \
        --positive-weight "${positive_weight}" \
        --negative-weight "${negative_weight}" \
        --negative-margin "${negative_margin}" \
        --cfp-splits 2-10 \
        --cfp-max-pairs-per-split 80 \
        --num-calib 500 \
        > "${log_file}" 2>&1
}

{
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_cfp_fb_a" \
        "${LOG_DIR}/s2_w1_pairft_cfp_fb_a_e8.log" \
        1.0 0.8 12.0 0.03
    date
    copy_cache \
        "official_mobilefacenet/student_distill_w1_pairft_cfp_fb_a" \
        "official_mobilefacenet/student_distill_w1_pairft_cfp_fb_b"
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_cfp_fb_b" \
        "${LOG_DIR}/s2_w1_pairft_cfp_fb_b_e8.log" \
        0.8 1.0 12.0 0.03
    date
} > "${LOG_DIR}/s2_w1_pairft_cfp_fallback_sweep.log" 2>&1 &

echo $!
