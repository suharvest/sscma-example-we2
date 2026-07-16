#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

SRC_DIR="official_mobilefacenet/student_distill_w1_pairft"
INIT_WEIGHTS="official_mobilefacenet/student_distill_w1_hardneg/mfn_w1_distill_128d.weights.h5"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

run_variant() {
    local out_dir="$1"
    local log_file="$2"
    local distill_weight="$3"
    local positive_weight="$4"
    local negative_weight="$5"
    local negative_margin="$6"

    mkdir -p "${out_dir}"
    for suffix in aligned_lfw.npz teacher512_3437.npz; do
        local src="${SRC_DIR}/mfn_w1_pairft_128d_${suffix}"
        local dst="${out_dir}/mfn_w1_pairft_128d_${suffix}"
        if [[ -f "${src}" && ! -f "${dst}" ]]; then
            cp "${src}" "${dst}"
        fi
    done

    /home/harve/.local/bin/uv run python -u train_mfn_student_pair_finetune.py \
        --width 1.0 \
        --epochs 8 \
        --batch-size 128 \
        --lr 0.00005 \
        --out-dir "${out_dir}" \
        --init-weights "${INIT_WEIGHTS}" \
        --checkpoint-every 4 \
        --distill-weight "${distill_weight}" \
        --positive-weight "${positive_weight}" \
        --negative-weight "${negative_weight}" \
        --negative-margin "${negative_margin}" \
        --num-calib 500 \
        > "${log_file}" 2>&1
}

{
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_distill2" \
        "${LOG_DIR}/s2_w1_pairft_distill2_e8.log" \
        2.0 0.4 8.0 0.03
    date
    run_variant \
        "official_mobilefacenet/student_distill_w1_pairft_distill3" \
        "${LOG_DIR}/s2_w1_pairft_distill3_e8.log" \
        3.0 0.25 8.0 0.02
    date
} > "${LOG_DIR}/s2_w1_pairft_conservative_sweep.log" 2>&1 &

echo $!
