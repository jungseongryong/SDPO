#!/bin/bash
# Policy-gradient SDPO baseline on Qwen3-4B.
# Uses grpo_ca with no outcome-reward advantage and no JSD/Renyi transform:
#   A_t = lambda * normalize(logp_teacher(sampled token) - logp_student(sampled token))

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}" \
MODEL_TAG="Q4B" \
PRM_FORWARD_MODE="none" \
PRM_CONSTRUCTION="raw" \
CA_LAMBDA_MODE="fixed" \
CA_LAMBDA="${CA_LAMBDA:-1.0}" \
ORM_WEIGHT="0.0" \
TP_TARGET_RATIO="0.0" REACTIVATE_RATIO="0.0" \
LEN_MASK="12000" \
LOSS_AGG_MODE="token-mean" \
EXP_NAME_SUFFIX="pg-sdpo" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh"
