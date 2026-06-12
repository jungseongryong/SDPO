#!/bin/bash
# Default Self-Distillation (t-s direction) on OLMo-3-7B-Instruct.
# Mirror of antisd.sh with PRM_RENYI_SIGN flipped to +1.0 — see
# run/qwen3-8b/sd.sh for the full motivation. Schmitt gate also disabled.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-allenai/Olmo-3-7B-Instruct}" \
MODEL_TAG="ON" \
PRM_RENYI_SIGN="1.0" \
TP_TARGET_RATIO="0.0" REACTIVATE_RATIO="0.0" \
LEN_MASK="12000" \
LOSS_AGG_MODE="token-mean" \
EXP_NAME_SUFFIX="ts" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh"
