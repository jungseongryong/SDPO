#!/bin/bash
# RLSD on Qwen3-4B.
# Magnitude-only teacher/student token reweighting:
#   w_t = exp(sign(A) * (log p_teacher - log p_student))
# Applied to all rollouts.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}" \
MODEL_TAG="Q4B" \
CA_MODE="rlsd" \
RLSD_LAMBDA="${RLSD_LAMBDA:-0.5}" \
RLSD_EPS_W="${RLSD_EPS_W:-0.2}" \
TEACHER_UPDATE_RATE="1.0" \
TP_TARGET_RATIO="0.0" REACTIVATE_RATIO="0.0" \
LEN_MASK="12000" \
LOSS_AGG_MODE="seq-mean-token-mean" \
EXP_NAME_SUFFIX="rlsd" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh"
