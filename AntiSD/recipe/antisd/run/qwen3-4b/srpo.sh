#!/bin/bash
# SRPO on Qwen3-4B.
# Routes correct rollouts to GRPO and incorrect rollouts with teacher context
# to entropy-weighted top-k JSD distillation.
# LR follows launch_short.sh; EMA is fixed off to match the SDPO baseline.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}" \
MODEL_TAG="Q4B" \
LOSS_MODE="srpo" \
TEACHER_UPDATE_RATE="0.0" \
LOSS_AGG_MODE="seq-mean-token-mean" \
EXP_NAME_SUFFIX="srpo" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh" \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.srpo_beta=1.0
