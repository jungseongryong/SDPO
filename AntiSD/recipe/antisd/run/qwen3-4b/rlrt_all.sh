#!/bin/bash
# RLRT-All on Qwen3-4B.
# Same token-reweighting form as RLSD, but reverses the teacher/student ratio:
#   w_t = exp(sign(A) * (log p_student - log p_teacher))
# Applies the reversed-teacher weight to all rollouts; no r=1-only gate.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}" \
MODEL_TAG="Q4B" \
CA_MODE="rlrt_all" \
RLSD_LAMBDA="${RLSD_LAMBDA:-0.5}" \
RLSD_EPS_W="${RLSD_EPS_W:-1.0}" \
TEACHER_UPDATE_RATE="1.0" \
TP_TARGET_RATIO="0.0" REACTIVATE_RATIO="0.0" \
LEN_MASK="12000" \
LOSS_AGG_MODE="seq-mean-token-mean" \
EXP_NAME_SUFFIX="rlrt-all" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh"
