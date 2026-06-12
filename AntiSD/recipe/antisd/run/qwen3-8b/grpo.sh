#!/bin/bash
# v25-QN-GRPO-4node: Qwen3-8B GRPO baseline, 4 nodes, save every 10 steps.
set -e

WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"


MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="v25-QN-GRPO-4node-rollout16384-eval32768-mbs32-lr1e-6-${MODEL_NAME}"

# Repo root: auto-detect, validate, fall back if user's env var is a bad placeholder.
_SDPO_ROOT_AUTO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
if [[ -z "${SDPO_ROOT:-}" ]]; then
    SDPO_ROOT="$_SDPO_ROOT_AUTO"
elif [[ ! -f "${SDPO_ROOT}/verl/__init__.py" ]]; then
    echo "[launch] WARN: SDPO_ROOT='${SDPO_ROOT}' is not an AntiSD repo; using auto-detected '${_SDPO_ROOT_AUTO}'" >&2
    SDPO_ROOT="$_SDPO_ROOT_AUTO"
fi
NNODES="${NNODES:-1}"
if [[ -z "${N_GPUS_PER_NODE:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        N_GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        [[ "$N_GPUS_PER_NODE" == "0" ]] && N_GPUS_PER_NODE=8
    else
        N_GPUS_PER_NODE=8
    fi
fi


# ── Worker env vars (verl + Ray + PyTorch) ────────────────────────────────
export WANDB_API_KEY
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_V1=1
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${SDPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TASK="datasets/math"
export EXPERIMENT="${EXP_NAME}"

# Stale RAY_ADDRESS=auto in parent shell makes ray.init() try to connect to
# a non-existent cluster. Drop it for single-node; keep only explicit values.
if [[ -z "${RAY_ADDRESS:-}" || "${RAY_ADDRESS}" == "auto" ]]; then
    unset RAY_ADDRESS
fi

unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT \
      GROUP_RANK ROLE_RANK ROLE_NAME ROLE_WORLD_SIZE 2>/dev/null || true

echo "[launch] EXP_NAME = ${EXP_NAME}"
echo "[launch] python3 -m verl.trainer.main_ppo  (NNODES=${NNODES} N_GPUS_PER_NODE=${N_GPUS_PER_NODE})"


# Cap Ray prestart workers to avoid registration-storm hang on big-CPU hosts.
RAY_NUM_CPUS_ARG=""
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    RAY_NUM_CPUS_ARG="ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-32}"
fi

exec python3 -m verl.trainer.main_ppo \
    ${RAY_NUM_CPUS_ARG} \
    --config-name sdpo \
    max_model_len=35328 \
    data.train_files="[${SDPO_ROOT}/datasets/math/train.parquet]" \
    data.val_files="[${SDPO_ROOT}/datasets/math/aime25/test.parquet]" \
    data.filter_overlong_prompts_workers=64 \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.train_response_length=16384 \
    "data.apply_chat_template_kwargs={enable_thinking: false}" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.response_length=32768 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.policy_loss.loss_mode=grpo_ca \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=1.0 \
    actor_rollout_ref.actor.self_distillation.max_solution_tokens=3072 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=4096 \
    actor_rollout_ref.actor.self_distillation.solution_selection=random \
    actor_rollout_ref.actor.self_distillation.truncate_solution_at_correct_answer=true \
    actor_rollout_ref.actor.self_distillation.solution_source=group_only \
    actor_rollout_ref.actor.self_distillation.solution_content=full \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=true \
    actor_rollout_ref.actor.self_distillation.provide_ground_truth_in_feedback=false \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.ccir.enabled=False \
    actor_rollout_ref.actor.ccir.ca_mode=additive \
    actor_rollout_ref.actor.ccir.ca_lambda=0.0 \
    actor_rollout_ref.actor.ccir.orm_weight=1.0 \
    actor_rollout_ref.actor.ccir.prm_normalize_mode=sequence \
    actor_rollout_ref.actor.ccir.prm_entropy_neutral=none \
    actor_rollout_ref.actor.ccir.prm_anchor_to_orm=false \
    actor_rollout_ref.actor.ccir.prm_seq_demean=false \
    actor_rollout_ref.actor.ccir.prm_construction=reverse \
    actor_rollout_ref.actor.ccir.prm_gamma=1.0 \
    actor_rollout_ref.actor.ccir.si_mode=none \
    actor_rollout_ref.actor.ccir.si_reference=bare \
    actor_rollout_ref.actor.ccir.maxent_coeff=none \
    actor_rollout_ref.actor.ccir.maxent_alpha=0.0 \
    custom_reward_function.path=${SDPO_ROOT}/verl/utils/reward_score/math_feedback/__init__.py \
    trainer.nnodes=$NNODES \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.total_epochs=1 \
    trainer.group_name=math-grpo-ca \
    trainer.default_local_dir="${SDPO_ROOT}/sdpo_ent_ckpt/${EXP_NAME}" \
    "trainer.logger=['console','wandb']"
