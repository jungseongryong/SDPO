#!/bin/bash
# AntiSD long-context launcher: 32k response, thinking enabled.
# Used by sd.sh / antisd.sh under run/olmo3-think/.
#
# Single-node by default. We just `exec python3 -m verl.trainer.main_ppo`;
# verl's main_ppo bootstraps a local Ray cluster internally when none is
# running, so logs stream straight to the terminal. For multi-node, start
# your own cluster externally and pass NNODES + RAY_ADDRESS.
#
# Differences from launch_short.sh:
#   - max_model_len=67584, max_response_length=32768
#   - enable_thinking=true (Qwen-think / OLMo-think models)
#   - remove_thinking_from_demonstration=true (teacher's solution stripped of <think>)
#   - LEN_MASK default 32768 (matches longer responses)
#   - LOSS_AGG_MODE default seq-mean-token-mean (long-context stability)
#
# Common env-var knobs (see README for full list):
#   PRM_RENYI_SIGN   (default: -1.0)        AntiSD; set +1.0 for default SD
#   PRM_CONSTRUCTION (default: reverse), PRM_FORWARD_MODE (default: jsd_unbiased)
#   ENABLE_THINKING  (default: true), REMOVE_THINKING (default: true)
#   NNODES           (default: 1)
#   N_GPUS_PER_NODE  (default: auto-detect from nvidia-smi)
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODEL_TAG:?MODEL_TAG is required}"
# Auto-calibration defaults (override only for ablations).
TP_TARGET="${TP_TARGET:-0.0}"
REACTIVATE_TARGET="${REACTIVATE_TARGET:-0.0}"
SIGMA_REF_FIXED="${SIGMA_REF_FIXED:-0.0}"
TP_TARGET_RATIO="${TP_TARGET_RATIO:-0.93}"
REACTIVATE_RATIO="${REACTIVATE_RATIO:-1.07}"

PRM_CONSTRUCTION="${PRM_CONSTRUCTION:-reverse}"
PRM_FORWARD_MODE="${PRM_FORWARD_MODE:-jsd_unbiased}"
PRM_RENYI_SIGN="${PRM_RENYI_SIGN:--1.0}"
SOLUTION_MODE="${SOLUTION_MODE:-normal}"
CA_LAMBDA_MODE="${CA_LAMBDA_MODE:-teacher_perp}"
EXP_NAME_SUFFIX="${EXP_NAME_SUFFIX:-}"

# Long-response / thinking-model defaults
ENABLE_THINKING="${ENABLE_THINKING:-true}"
REMOVE_THINKING="${REMOVE_THINKING:-true}"

# Repo root: auto-detect, validate, fall back if user's env var is a bad placeholder.
_SDPO_ROOT_AUTO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
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
WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
LAMBDA_MIN="${LAMBDA_MIN:-0.0}"
LAMBDA_MAX="${LAMBDA_MAX:-0.5}"
PRM_CLIP="${PRM_CLIP:-5.0}"
LOG_CLIP="${LOG_CLIP:-5.0}"
K_SIGMA="${K_SIGMA:-2.0}"
LEN_MASK="${LEN_MASK:-32768}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
if [[ "$TP_TARGET" == "0.0" || "$TP_TARGET" == "0" ]]; then
    TP_TAG="r${TP_TARGET_RATIO//./p}"
else
    TP_TAG="${TP_TARGET//./p}"
fi
if [[ "$REACTIVATE_TARGET" == "0.0" || "$REACTIVATE_TARGET" == "0" ]]; then
    RT_TAG="r${REACTIVATE_RATIO//./p}"
else
    RT_TAG="${REACTIVATE_TARGET//./p}"
fi
LM_TAG="${LAMBDA_MIN//-/n}"; LM_TAG="${LM_TAG//./p}"
LAM_TAG=""
case "$LOSS_AGG_MODE" in
    token-mean) LAM_TAG="-tm";;
    seq-mean-token-mean) LAM_TAG="-smtm";;
    *) LAM_TAG="-${LOSS_AGG_MODE}";;
esac
LM_MASK_TAG="-lm${LEN_MASK}"
CAL_TAG="fixed"
if [[ "$TP_TARGET" == "0.0" || "$TP_TARGET" == "0" ]]; then
    CAL_TAG="auto"
fi
SUFFIX_TAG=""
if [[ -n "${EXP_NAME_SUFFIX}" ]]; then
    SUFFIX_TAG="-${EXP_NAME_SUFFIX}"
fi
EXP_NAME="v33-${MODEL_TAG}-long-jsdHys${LAM_TAG}${SUFFIX_TAG}-${CAL_TAG}-tp${TP_TAG}-rt${RT_TAG}-lmin${LM_TAG}${LM_MASK_TAG}-warm${WARMUP_STEPS}-rollout32768-eval32768-mbs32-lr1e-6-${MODEL_NAME}"

CONFIG_ARGS=(
    --config-name sdpo
    max_model_len=67584
    data.train_files="[${SDPO_ROOT}/datasets/math/train.parquet]"
    data.val_files="[${SDPO_ROOT}/datasets/math/aime25/test.parquet]"
    data.train_batch_size=32
    data.filter_overlong_prompts_workers=64
    data.max_prompt_length=2048
    data.max_response_length=32768
    data.train_response_length=32768
    "data.apply_chat_template_kwargs={enable_thinking: ${ENABLE_THINKING}}"
    actor_rollout_ref.model.path=$MODEL_PATH
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.val_kwargs.response_length=32768
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.policy_loss.loss_mode=grpo_ca
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=1.0
    actor_rollout_ref.actor.self_distillation.max_solution_tokens=3072
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=4096
    actor_rollout_ref.actor.self_distillation.solution_selection=random
    actor_rollout_ref.actor.self_distillation.truncate_solution_at_correct_answer=true
    actor_rollout_ref.actor.self_distillation.solution_source=group_only
    actor_rollout_ref.actor.self_distillation.solution_content=full
    actor_rollout_ref.actor.self_distillation.solution_mode=${SOLUTION_MODE}
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=true
    actor_rollout_ref.actor.self_distillation.provide_ground_truth_in_feedback=false
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=${REMOVE_THINKING}
    actor_rollout_ref.actor.ccir.enabled=False
    actor_rollout_ref.actor.ccir.ca_mode=additive
    actor_rollout_ref.actor.ccir.ca_lambda=0.1
    actor_rollout_ref.actor.ccir.ca_lambda_mode=${CA_LAMBDA_MODE}
    actor_rollout_ref.actor.ccir.ca_lambda_perp_target=$TP_TARGET
    actor_rollout_ref.actor.ccir.ca_lambda_perp_target_ratio=$TP_TARGET_RATIO
    actor_rollout_ref.actor.ccir.ca_lambda_perp_reactivate_target=$REACTIVATE_TARGET
    actor_rollout_ref.actor.ccir.ca_lambda_perp_reactivate_ratio=$REACTIVATE_RATIO
    actor_rollout_ref.actor.ccir.ca_lambda_perp_delta=0.10
    actor_rollout_ref.actor.ccir.ca_lambda_perp_alpha=2.0
    actor_rollout_ref.actor.ccir.ca_lambda_tppl_scope=per_seq
    actor_rollout_ref.actor.ccir.ca_lambda_min=$LAMBDA_MIN
    actor_rollout_ref.actor.ccir.ca_lambda_max=$LAMBDA_MAX
    actor_rollout_ref.actor.ccir.ca_lambda_perp_mask=3.0
    actor_rollout_ref.actor.ccir.ca_lambda_warmup_steps=$WARMUP_STEPS
    actor_rollout_ref.actor.ccir.ca_lambda_mean_shift=false
    actor_rollout_ref.actor.ccir.ca_lambda_mean_shift_always=false
    actor_rollout_ref.actor.ccir.orm_weight=1.0
    actor_rollout_ref.actor.ccir.prm_weight=0.0
    actor_rollout_ref.actor.ccir.prm_normalize=true
    actor_rollout_ref.actor.ccir.prm_normalize_mode=sequence
    actor_rollout_ref.actor.ccir.prm_entropy_neutral=none
    actor_rollout_ref.actor.ccir.prm_anchor_to_orm=false
    actor_rollout_ref.actor.ccir.prm_seq_demean=false
    actor_rollout_ref.actor.ccir.prm_clip=$PRM_CLIP
    actor_rollout_ref.actor.ccir.prm_construction=${PRM_CONSTRUCTION}
    actor_rollout_ref.actor.ccir.prm_gamma=1.0
    actor_rollout_ref.actor.ccir.prm_forward_mode=${PRM_FORWARD_MODE}
    actor_rollout_ref.actor.ccir.prm_renyi_sign=$PRM_RENYI_SIGN
    actor_rollout_ref.actor.ccir.prm_renyi_virtual_alpha=1.0
    actor_rollout_ref.actor.ccir.prm_forward_log_clip=$LOG_CLIP
    actor_rollout_ref.actor.ccir.prm_u_clip_mode=adaptive
    actor_rollout_ref.actor.ccir.prm_u_clip_k_sigma=$K_SIGMA
    actor_rollout_ref.actor.ccir.prm_u_clip_sigma_ref_fixed=$SIGMA_REF_FIXED
    actor_rollout_ref.actor.ccir.prm_length_mask_threshold=$LEN_MASK
    actor_rollout_ref.actor.ccir.si_mode=none
    actor_rollout_ref.actor.ccir.si_reference=bare
    actor_rollout_ref.actor.ccir.maxent_coeff=none
    actor_rollout_ref.actor.ccir.maxent_alpha=0.0
    actor_rollout_ref.actor.ccir.ccir_cross_problem=false
    custom_reward_function.path=${SDPO_ROOT}/verl/utils/reward_score/math_feedback/__init__.py
    trainer.nnodes=$NNODES
    trainer.test_freq=10
    trainer.save_freq=10
    trainer.max_actor_ckpt_to_keep=null
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE
    trainer.total_epochs=1
    trainer.group_name=math-grpo-ca
    trainer.default_local_dir="${SDPO_ROOT}/sdpo_ent_ckpt/${EXP_NAME}"
    "trainer.logger=['console','wandb']"
)


# Single-node default. verl's main_ppo.py handles Ray lifecycle internally
# (ray.init() bootstraps a local cluster when none is running). Multi-node:
# bring up your own cluster externally and pass NNODES=N + RAY_ADDRESS.

# ── Worker env vars ───────────────────────────────────────────────────────
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

unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT \
      GROUP_RANK ROLE_RANK ROLE_NAME ROLE_WORLD_SIZE 2>/dev/null || true

# Stale RAY_ADDRESS=auto in parent shell makes ray.init() try to connect to
# a non-existent cluster. Drop it for single-node; keep only explicit values.
if [[ -z "${RAY_ADDRESS:-}" || "${RAY_ADDRESS}" == "auto" ]]; then
    unset RAY_ADDRESS
fi

# Single-node bootstrap: cap Ray's pre-started worker count to avoid the
# registration storm on hosts with many CPUs (192 default → workers exceed
# registration timeout). Skipped when CONNECTING to an existing cluster.
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    CONFIG_ARGS+=( "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-32}" )
fi

echo "[launch] EXP_NAME = ${EXP_NAME}"
echo "[launch] python3 -m verl.trainer.main_ppo  (NNODES=${NNODES} N_GPUS_PER_NODE=${N_GPUS_PER_NODE})"
exec python3 -m verl.trainer.main_ppo "${CONFIG_ARGS[@]}"
