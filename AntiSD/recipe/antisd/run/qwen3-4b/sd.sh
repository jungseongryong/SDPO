#!/bin/bash
# Default Self-Distillation (t-s direction) on Qwen3-4B.
# Same launcher / hyperparameters as antisd.sh — only PRM_RENYI_SIGN is
# flipped to +1.0, which inverts the per-token gradient direction so the
# student is pulled TOWARD the privileged-context teacher (the failure mode
# the paper studies).
#
# Schmitt gate is disabled here (TP_TARGET_RATIO=0, REACTIVATE_RATIO=0):
# under t-s direction the gate would close exactly when distillation starts
# to "work", prematurely hiding the harm. Running gate-off keeps the SD
# signal active for the full run so its intrinsic dynamics are visible.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}" \
MODEL_TAG="Q4B" \
PRM_RENYI_SIGN="1.0" \
TP_TARGET_RATIO="0.0" REACTIVATE_RATIO="0.0" \
LEN_MASK="12000" \
LOSS_AGG_MODE="token-mean" \
EXP_NAME_SUFFIX="ts" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh"
