# AntiSD 
Reproduces the headline experiments from
*Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information*

## What's here

```
recipe/antisd/
├── README.md                # this file
├── _launchers/
│   ├── launch_short.sh      # 16k response, no-think  (qwen3-4b, qwen3-4b-inst, qwen3-8b, olmo3-instruct)
│   └── launch_long.sh       # 32k response, thinking  (olmo3-think)
└── run/                     # run/<model>/<method>.sh — 4 models × 3 methods = 12 scripts
    ├── qwen3-4b/            # Qwen3-4B
    │   ├── grpo.sh
    │   ├── sd.sh
    │   └── antisd.sh
    ├── qwen3-4b-inst/       # Qwen3-4B-Instruct-2507
    │   ├── grpo.sh          # GRPO baseline
    │   ├── sd.sh            # default Self-Distillation
    │   └── antisd.sh        # AntiSD (ours)
    ├── qwen3-8b/            # Qwen3-8B
    │   ├── grpo.sh
    │   ├── sd.sh
    │   └── antisd.sh
    ├── olmo3-instruct/      # OLMo-3-7B-Instruct
    │   ├── grpo.sh
    │   ├── sd.sh
    │   └── antisd.sh
    └── olmo3-think/         # OLMo-3-7B-Think (32k thinking-mode context)
        ├── grpo.sh
        ├── sd.sh
        └── antisd.sh
```

## Methods at a glance

| Method | Per-token PRM signal | `PRM_RENYI_SIGN` | 
|--------|----------------------|------------------|
| **GRPO** (baseline) | none — sequence-level reward only | n/a | 
| **SD** (default Self-Distillation) | descend `D(s‖t)` ≈ reverse-KL toward teacher (`t-s` direction) | `+1.0` |
| **AntiSD** (ours) | ascend bounded JSD between student and teacher (`s-t` direction with softplus shaping) | `-1.0` |

Both SD and AntiSD share the same on-policy self-distillation rollout pipeline and the same launcher; the only difference between them is `PRM_RENYI_SIGN` (direction of the per-token gradient), set via env var inside the run script.

GRPO does not use self-distillation rollouts at all — those scripts are self-contained for direct comparison.

## Running


```bash

bash recipe/antisd/run/qwen3-8b/antisd.sh         # AntiSD on Qwen3-8B 

bash recipe/antisd/run/qwen3-8b/sd.sh             # default Self-Distillation
bash recipe/antisd/run/qwen3-8b/grpo.sh           # GRPO baseline
```


```bash
export MODEL_PATH=/local/cache/Qwen3-8B       # default: HF Hub ID like Qwen/Qwen3-8B
export NNODES=4                               # default: 1 (multi-node — see below)
export N_GPUS_PER_NODE=4                      # default: nvidia-smi -L | wc -l
export SDPO_ROOT=/custom/path/to/AntiSD       # default: auto-detect
export RAY_ADDRESS=ray://my-head:10001        # multi-node only; unset → verl bootstraps locally
```

The launcher in `_launchers/launch_short.sh` (and `launch_long.sh` for the
thinking-mode variants) exposes the full set of algorithmic knobs as env
vars — see the header comment in those files for the list.


## Datasets

The launchers point at `${SDPO_ROOT}/datasets/math/train.parquet` and
`${SDPO_ROOT}/datasets/math/aime25/test.parquet`. Build both with one command:

```bash
bash data/prepare_antisd.sh                   # ~5 min, downloads from HuggingFace
```

This downloads + preprocesses:
- **DAPO-Math-17k** (train, ~17k problems)
- **AIME 2025** (eval, 30 problems × 4 samples)
- **AIME 2024** (eval)
- **HMMT February 2025** (eval)


Available eval sets: `aime25 aime26 aime_2024 amc23 beyondaime hmmt25 math500 minervamath olympiadbench`.
