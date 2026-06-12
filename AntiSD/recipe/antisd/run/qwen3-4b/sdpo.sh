#!/bin/bash
# Top-k JSD SDPO baseline on Qwen3-4B.
# Keeps the same run settings as the GRPO/AntiSD launcher and only switches
# the objective to SDPO's teacher-student distillation loss.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}" \
MODEL_TAG="Q4B" \
LOSS_MODE="sdpo" \
LOSS_AGG_MODE="token-mean" \
EXP_NAME_SUFFIX="sdpo-jsd" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/_launchers/launch_short.sh" \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0
