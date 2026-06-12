#!/usr/bin/env bash
set -euo pipefail

cd /home1/irteam/SDPO/AntiSD

GRPO_RC_FILE="qwen3-4b-grpo.rc"
SD_RC_FILE="qwen3-4b-sd.rc"
WATCH_LOG="qwen3-4b-sd-after-grpo.log"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" >> "${WATCH_LOG}"
}

log "watcher started; waiting for ${GRPO_RC_FILE}"

while [[ ! -f "${GRPO_RC_FILE}" ]]; do
    sleep 60
done

grpo_rc="$(tr -d '[:space:]' < "${GRPO_RC_FILE}")"
log "detected GRPO completion rc=${grpo_rc}"

if [[ "${grpo_rc}" != "0" ]]; then
    log "GRPO failed; SD will not start"
    exit 1
fi

source .venv/bin/activate
source /home1/irteam/.config/sdpo/wandb.env

log "stopping any leftover Ray processes"
ray stop --force >> "${WATCH_LOG}" 2>&1 || true
sleep 10

ts="$(date +%Y%m%d_%H%M%S)"
for f in qwen3-4b-sd.log qwen3-4b-sd.wrapper.log "${SD_RC_FILE}"; do
    [[ -e "${f}" ]] && mv "${f}" "${f}.prev.${ts}"
done
rm -f "${SD_RC_FILE}"

export N_GPUS_PER_NODE=8
export NNODES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export CC=/opt/conda/bin/x86_64-conda-linux-gnu-gcc
export CXX=/opt/conda/bin/x86_64-conda-linux-gnu-g++
export TOTAL_TRAINING_STEPS=200
export MODEL_PATH=Qwen/Qwen3-4B

log "starting Qwen3-4B SD 200-step run"
bash recipe/antisd/run/qwen3-4b/sd.sh > qwen3-4b-sd.log 2> qwen3-4b-sd.wrapper.log
rc=$?
echo "${rc}" > "${SD_RC_FILE}"
log "SD finished rc=${rc}"
exit "${rc}"
