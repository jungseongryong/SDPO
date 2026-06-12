#!/bin/bash
# Default Self-Distillation (t-s direction) on OLMo-3-7B-Think.
# Mirror of antisd.sh with PRM_RENYI_SIGN flipped to +1.0 and Schmitt gate
# disabled — see run/qwen3-8b/sd.sh for the full motivation.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-allenai/Olmo-3-7B-Think}" \
MODEL_TAG="OT" \
PRM_RENYI_SIGN="1.0" \
TP_TARGET_RATIO="0.0" REACTIVATE_RATIO="0.0" \
WARMUP_STEPS="10" \
EXP_NAME_SUFFIX="ts" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_long.sh"
