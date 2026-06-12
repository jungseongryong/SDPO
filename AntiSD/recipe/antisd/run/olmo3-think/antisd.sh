#!/bin/bash
# AntiSD on OLMo-3-7B-Think (thinking, 32k response, len_mask=32768).
# Thresholds and σ_ref are auto-calibrated; warmup is bumped to 10 because
# thinking-mode responses have higher variance and need more samples for a
# stable median. See _launchers/launch_long.sh for the full default set.
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-allenai/Olmo-3-7B-Think}" \
MODEL_TAG="OT" \
WARMUP_STEPS="10" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_long.sh"
