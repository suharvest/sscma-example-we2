#!/usr/bin/env bash
set -euo pipefail

cd /home/harve/gv2_face_train/tflm_face_embedding

LOG_DIR="logs"
WATCH_LOG="${LOG_DIR}/face_embedding_cron_check.log"
NEXT_SCRIPT="${1:-scripts/face_embedding_next.sh}"
mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"
}

active_jobs() {
    local jobs
    jobs="$(pgrep -af "train_mfn_student|evaluate_embedding_models.py|train_mfn_student_from_tflite128.py" \
        | grep -v "face_embedding_cron_check" \
        | grep -v "pgrep -af" \
        || true)"
    if [[ -n "${jobs}" ]]; then
        echo "${jobs}"
        return 0
    fi
    return 1
}

latest_result_hint() {
    local latest
    latest="$(ls -t logs/iddistill*.log 2>/dev/null | head -1 || true)"
    if [[ -n "${latest}" ]]; then
        log "latest log: ${latest}"
        grep -E "LFW|CFP-FP|TAR@FAR|balanced score|Score|FAILED|Done in|epoch " "${latest}" \
            | tail -40 >> "${WATCH_LOG}" || true
    fi
}

log "cron check start"
if active_jobs >/tmp/face_embedding_jobs.$$; then
    log "active training/eval found; skip launch"
    cat /tmp/face_embedding_jobs.$$ >> "${WATCH_LOG}"
    rm -f /tmp/face_embedding_jobs.$$
    exit 0
fi
rm -f /tmp/face_embedding_jobs.$$

latest_result_hint

if [[ -x "${NEXT_SCRIPT}" ]]; then
    log "launch next script: ${NEXT_SCRIPT}"
    nohup "${NEXT_SCRIPT}" >> "${WATCH_LOG}" 2>&1 &
    log "launched pid=$!"
else
    log "no executable next script at ${NEXT_SCRIPT}; leaving idle for manual decision"
fi
