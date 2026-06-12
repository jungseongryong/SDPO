#!/bin/bash
# AntiSD on Qwen3-8B (no-think, 16k response, len_mask=12000).
# Thresholds (TP_TARGET / REACTIVATE_TARGET / SIGMA_REF_FIXED) are auto-
# calibrated from the first 5 warmup rollouts — see _launchers/launch_short.sh
# for the calibration logic and override env vars for ablations.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}" \
MODEL_TAG="QN" \
LEN_MASK="12000" \
LOSS_AGG_MODE="token-mean" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh"
