#!/usr/bin/env python3
"""
Preprocess math datasets for AntiSD training and evaluation.

Training sets:
  - agentica-org/DeepScaleR-Preview-Dataset  (40k problems with step-by-step solutions)
  - open-thoughts/OpenThoughts-114k          (math subset ~40k, with full CoT reasoning)

Test sets:
  - math-ai/amc23
  - HuggingFaceH4/aime_2024
  - math-ai/aime25
  - HuggingFaceH4/MATH-500
  - math-ai/minervamath
  - math-ai/olympiadbench
  - ByteDance-Seed/BeyondAIME
  - MathArena/hmmt_feb_2025    (HMMT February 2025)

Usage:
    python data/preprocess_math_datasets.py --output_dir datasets/math
    python data/preprocess_math_datasets.py --output_dir datasets/math --datasets deepscaler aime25
    python data/preprocess_math_datasets.py --output_dir datasets/math --datasets openthoughts --max_samples 30000
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import datasets as hf_datasets


INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."

# Minimal instruction for datasets with their own CoT solutions (avoids "step by step" duplication
# when teacher template already provides reasoning guidance via reference solution).
INSTRUCTION_BOXED_ONLY = r"Put your final answer within \boxed{}."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_answer(answer):
    """Normalize answer to string format."""
    if answer is None:
        return ""
    if isinstance(answer, list):
        if len(answer) == 1:
            return str(answer[0])
        elif len(answer) > 1:
            return ", ".join(str(a) for a in answer)
        else:
            return ""
    return str(answer)


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract content from the last \boxed{...} in text.

    Returns the content inside \boxed{}, or None if not found.
    """
    if not text:
        return None
    last_boxed_start = text.rfind(r"\boxed{")
    if last_boxed_start == -1:
        return None

    depth = 0
    i = last_boxed_start + len(r"\boxed{")
    start = i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
        i += 1
    return None


def fix_solution_boxed(solution: str, answer: str) -> str:
    r"""Replace the last \boxed{...} in solution with \boxed{answer}.

    DeepScaleR solutions often end with choice labels like \boxed{A} or
    \boxed{\textbf{(A)}\ 26} from MCQ origins.  We replace the last \boxed{}
    with the correct answer from the answer field.
    """
    if not solution:
        return solution

    # Find the last \boxed{...} (handling nested braces)
    # Walk backwards to find the last \boxed
    last_boxed_start = solution.rfind(r"\boxed{")
    if last_boxed_start == -1:
        # No \boxed found, append one
        return solution.rstrip() + f"\n\\boxed{{{answer}}}"

    # Find matching closing brace
    depth = 0
    i = last_boxed_start + len(r"\boxed{")
    while i < len(solution):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                # Found the matching close brace
                return solution[:last_boxed_start] + f"\\boxed{{{answer}}}" + solution[i + 1:]
            depth -= 1
        i += 1

    # Malformed \boxed, just append
    return solution.rstrip() + f"\n\\boxed{{{answer}}}"


def _save_dataset(dataset, save_dir: Path, split: str):
    """Save dataset to parquet and write an example JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{split}.parquet"
    dataset.to_parquet(str(out_path))

    example_path = save_dir / f"{split}_example.json"
    with open(example_path, "w") as f:
        json.dump(dataset[0], f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(dataset)} examples -> {out_path}")


# ---------------------------------------------------------------------------
# Training: DeepScaleR
# ---------------------------------------------------------------------------

def process_deepscaler(output_dir: Path):
    """Process agentica-org/DeepScaleR-Preview-Dataset training set.

    Includes step-by-step solutions with corrected \\boxed{} answers.
    """
    print("\n" + "=" * 60)
    print("DeepScaleR-Preview-Dataset (train)")
    print("=" * 60)

    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    dataset = hf_datasets.load_dataset(data_source, split="train")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["problem"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        solution = example.get("solution", "") or ""
        solution = fix_solution_boxed(solution, answer)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": str(idx),
                "solution": solution,
                "has_solution": bool(solution),
            },
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "deepscaler", "train")
    return processed


# ---------------------------------------------------------------------------
# Training: OpenThoughts (math subset)
# ---------------------------------------------------------------------------

def process_openthoughts(output_dir: Path, max_samples: int = 30000):
    """Process open-thoughts/OpenThoughts-114k math subset.

    Uses the metadata config to get structured fields. Filters for math domain,
    extracts ground-truth answers from \\boxed{} in deepseek_solution, and stores
    the full CoT (deepseek_solution) as the teacher solution.

    Args:
        output_dir: Root output directory.
        max_samples: Maximum number of math examples to keep (default 30k, as in OPSD).
    """
    print("\n" + "=" * 60)
    print(f"OpenThoughts-114k math subset (train, max {max_samples})")
    print("=" * 60)

    data_source = "open-thoughts/OpenThoughts-114k"
    dataset = hf_datasets.load_dataset(data_source, "metadata", split="train")
    print(f"  Loaded {len(dataset)} total examples")

    # Filter for math domain
    math_dataset = dataset.filter(lambda x: x.get("domain") == "math")
    print(f"  Math examples: {len(math_dataset)}")

    # Subsample if needed
    if len(math_dataset) > max_samples:
        math_dataset = math_dataset.shuffle(seed=42).select(range(max_samples))
        print(f"  Subsampled to {len(math_dataset)} examples")

    # Pre-filter: only keep examples with extractable \boxed{} answers
    skipped = 0

    def process_fn(example, idx):
        nonlocal skipped
        # Use minimal instruction (just \boxed{} format) to avoid "step by step"
        # duplication — the OPSD-style teacher template already provides reasoning
        # guidance via the reference solution.
        question = example["problem"] + " " + INSTRUCTION_BOXED_ONLY

        # Extract answer from deepseek_solution's \boxed{}
        deepseek_sol = example.get("deepseek_solution") or ""
        answer = extract_boxed_answer(deepseek_sol)

        # Fallback: try ground_truth_solution
        if not answer:
            gt_sol = example.get("ground_truth_solution") or ""
            answer = extract_boxed_answer(gt_sol)

        if not answer:
            skipped += 1
            answer = ""  # will be filtered out below

        # Use deepseek_solution as the reference solution for teacher context.
        # This is a clean, concise solution (not the full reasoning trace).
        solution = deepseek_sol

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": str(idx),
                "solution": solution,
                "has_solution": bool(solution),
                "source": example.get("source", ""),
            },
        }

    processed = math_dataset.map(process_fn, with_indices=True, remove_columns=math_dataset.column_names)

    # Filter out examples without extractable answers
    before_filter = len(processed)
    processed = processed.filter(lambda x: bool(x["reward_model"]["ground_truth"]))
    after_filter = len(processed)
    print(f"  Skipped {before_filter - after_filter} examples without extractable \\boxed{{}} answer")
    print(f"  Final: {after_filter} examples")

    _save_dataset(processed, output_dir / "openthoughts", "train")
    return processed


# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------

def process_amc23(output_dir: Path):
    """math-ai/amc23 — {question, answer, url}"""
    print("\n" + "=" * 60)
    print("math-ai/amc23 (test)")
    print("=" * 60)

    data_source = "math-ai/amc23"
    dataset = hf_datasets.load_dataset(data_source, split="test")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["question"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": str(idx)},
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "amc23", "test")
    return processed


def process_aime_2024(output_dir: Path):
    """HuggingFaceH4/aime_2024 — {problem, solution, answer, url, year}"""
    print("\n" + "=" * 60)
    print("HuggingFaceH4/aime_2024 (test)")
    print("=" * 60)

    data_source = "HuggingFaceH4/aime_2024"
    dataset = hf_datasets.load_dataset(data_source, split="train")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["problem"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": str(idx)},
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "aime_2024", "test")
    return processed


def process_aime25(output_dir: Path):
    """math-ai/aime25 — {problem, answer}"""
    print("\n" + "=" * 60)
    print("math-ai/aime25 (test)")
    print("=" * 60)

    data_source = "math-ai/aime25"
    dataset = hf_datasets.load_dataset(data_source, split="test")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["problem"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": str(idx)},
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "aime25", "test")
    return processed


def process_aime26(output_dir: Path):
    """math-ai/aime26 — {problem, answer}. Same schema as aime25."""
    print("\n" + "=" * 60)
    print("math-ai/aime26 (test)")
    print("=" * 60)

    data_source = "math-ai/aime26"
    dataset = hf_datasets.load_dataset(data_source, split="test")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["problem"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": str(idx)},
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "aime26", "test")
    return processed


def process_math500(output_dir: Path):
    """HuggingFaceH4/MATH-500 — {problem, solution, answer, subject, level}"""
    print("\n" + "=" * 60)
    print("HuggingFaceH4/MATH-500 (test)")
    print("=" * 60)

    data_source = "HuggingFaceH4/MATH-500"
    dataset = hf_datasets.load_dataset(data_source, split="test")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["problem"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": str(idx),
                "subject": example.get("subject"),
                "level": example.get("level"),
            },
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "math500", "test")
    return processed


def process_minervamath(output_dir: Path):
    """math-ai/minervamath — {question, answer}"""
    print("\n" + "=" * 60)
    print("math-ai/minervamath (test)")
    print("=" * 60)

    data_source = "math-ai/minervamath"
    dataset = hf_datasets.load_dataset(data_source, split="test")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["question"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": str(idx)},
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "minervamath", "test")
    return processed


def process_olympiadbench(output_dir: Path):
    """math-ai/olympiadbench — {question, solution, final_answer, context, ...}"""
    print("\n" + "=" * 60)
    print("math-ai/olympiadbench (test)")
    print("=" * 60)

    data_source = "math-ai/olympiadbench"
    dataset = hf_datasets.load_dataset(data_source, split="test")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["question"]
        if example.get("context"):
            question = example["context"] + "\n\n" + question
        question = question + " " + INSTRUCTION
        answer = normalize_answer(example.get("final_answer", example.get("answer", "")))
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": str(idx),
                "difficulty": example.get("difficulty"),
                "subject": example.get("subject"),
                "subfield": example.get("subfield"),
            },
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "olympiadbench", "test")
    return processed


def process_deepmath(output_dir: Path):
    """Process zwhe99/DeepMath-103K training set.

    Includes three R1 reasoning traces per problem. We select the shortest
    valid solution (by character length) and fix its \\boxed{} answer.
    """
    print("\n" + "=" * 60)
    print("zwhe99/DeepMath-103K (train)")
    print("=" * 60)

    data_source = "zwhe99/DeepMath-103K"
    dataset = hf_datasets.load_dataset(data_source, split="train")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["question"] + " " + INSTRUCTION
        answer = normalize_answer(example["final_answer"])

        # Select shortest R1 solution (by character length)
        solutions = [
            example.get("r1_solution_1") or "",
            example.get("r1_solution_2") or "",
            example.get("r1_solution_3") or "",
        ]
        valid_solutions = [s for s in solutions if s.strip()]
        if valid_solutions:
            solution = min(valid_solutions, key=len)
            solution = fix_solution_boxed(solution, answer)
        else:
            solution = ""

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": str(idx),
                "solution": solution,
                "has_solution": bool(solution),
                "difficulty": example.get("difficulty", ""),
                "topic": example.get("topic", ""),
            },
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    # Filter out empty answers
    before_filter = len(processed)
    processed = processed.filter(lambda x: bool(x["reward_model"]["ground_truth"]))
    after_filter = len(processed)
    if before_filter != after_filter:
        print(f"  Filtered out {before_filter - after_filter} examples without answers")
    print(f"  Final: {after_filter} examples")

    _save_dataset(processed, output_dir / "deepmath", "train")
    return processed


def process_dapomath(output_dir: Path):
    """Process BytedTsinghua-SIA/DAPO-Math-17k training set.

    The canonical DAPO-Math dataset shipped with the verl recipe (~17k
    competition problems). It already follows verl's schema:
        prompt        : list[{"role": "user", "content": ...}]
        ability       : "math"
        reward_model  : {"style": "rule", "ground_truth": ...}
        extra_info    : {"split": "train", "index": ..., ...}
    so we just normalize the ground_truth field and pass everything else
    through unchanged.

    NOTE: this used to point at open-r1/DAPO-Math-17k-Processed (a cleaned
    fork), but our paper experiments were trained on the original dataset
    above; the processed fork has slightly different prompt wording.
    """
    print("\n" + "=" * 60)
    print("BytedTsinghua-SIA/DAPO-Math-17k (train)")
    print("=" * 60)

    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    dataset = hf_datasets.load_dataset(data_source, "default", split="train")
    print(f"  Loaded {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")

    def process_fn(example, idx):
        # The canonical dataset already provides chat-format prompt + verl-style
        # reward_model + extra_info. We keep them as-is and only normalize the
        # ground_truth (in case it comes as a list or non-string scalar).
        prompt = example.get("prompt", [])
        if isinstance(prompt, str):
            # Defensive fallback: wrap into chat format with the standard suffix.
            prompt = [{"role": "user", "content": prompt.strip() + " " + INSTRUCTION}]

        rm = example.get("reward_model", {}) or {}
        answer = normalize_answer(rm.get("ground_truth", ""))

        extra_info = dict(example.get("extra_info") or {})
        extra_info.setdefault("split", "train")
        extra_info.setdefault("index", str(idx))
        extra_info.setdefault("solution", "")
        extra_info.setdefault("has_solution", False)

        return {
            "data_source": example.get("data_source", data_source),
            "prompt": prompt,
            "ability": example.get("ability", "math"),
            "reward_model": {"style": rm.get("style", "rule"), "ground_truth": answer},
            "extra_info": extra_info,
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)

    before_filter = len(processed)
    processed = processed.filter(lambda x: bool(x["reward_model"]["ground_truth"]))
    after_filter = len(processed)
    if before_filter != after_filter:
        print(f"  Filtered out {before_filter - after_filter} examples without answers")
    print(f"  Final: {after_filter} examples")

    if len(processed) == 0:
        print("  WARNING: No examples after filtering!")
        return processed

    _save_dataset(processed, output_dir / "dapo-math", "train")
    return processed


def process_nemotron_math(output_dir: Path):
    """Process math problems from nvidia/Nemotron-Cascade-2-RL-data.

    Extracts AceReason (competition math) from MOPD split, plus STEM MCQA
    from multi-domain-RL. All have verifiable numerical/text answers.
    """
    print("\n" + "=" * 60)
    print("nvidia/Nemotron-Cascade-2-RL-data (math subset)")
    print("=" * 60)

    from huggingface_hub import hf_hub_download
    import json as _json

    data_source = "nvidia/Nemotron-Cascade-2-RL-data"
    all_examples = []

    # --- MOPD: AceReason (competition math) + STEM MCQA ---
    mopd_path = hf_hub_download(data_source, "MOPD/train.jsonl", repo_type="dataset")
    mopd_math = 0
    with open(mopd_path) as f:
        for line in f:
            ex = _json.loads(line)
            ds = ex.get("dataset", "")
            if ds == "acereason" and ex.get("expected_answer") is not None:
                # Competition math with numerical answer
                all_examples.append({
                    "question": ex["question"].strip(),
                    "answer": str(ex["expected_answer"]).strip(),
                    "source": "acereason",
                    "pass_rate": ex.get("pass_rate"),
                })
                mopd_math += 1
            elif ds == "nano_v3_sft_profiled_stem_mcqa" and ex.get("expected_answer") is not None:
                # STEM multiple choice — extract question from chat messages
                question = ex.get("question", "")
                if not question:
                    params = ex.get("responses_create_params", {})
                    msgs = params.get("input", [])
                    for m in msgs:
                        if m.get("role") == "user":
                            question = m["content"]
                            break
                if question:
                    all_examples.append({
                        "question": question.strip(),
                        "answer": str(ex["expected_answer"]).strip(),
                        "source": "stem_mcqa",
                        "pass_rate": ex.get("pass_rate"),
                    })
                    mopd_math += 1
    print(f"  MOPD: {mopd_math} math/STEM examples")

    # --- multi-domain-RL: more STEM MCQA ---
    mdrl_path = hf_hub_download(data_source, "multi-domain-RL/train.jsonl", repo_type="dataset")
    mdrl_count = 0
    with open(mdrl_path) as f:
        for line in f:
            ex = _json.loads(line)
            ds = ex.get("dataset", "")
            if ds == "nano_v3_sft_profiled_stem_mcqa" and ex.get("ground_truth") is not None:
                gt = ex["ground_truth"]
                # ground_truth can be structured — extract the answer
                if isinstance(gt, dict):
                    answer = str(gt.get("expected_answer", gt.get("answer", ""))).strip()
                elif isinstance(gt, str):
                    answer = gt.strip()
                else:
                    answer = str(gt).strip()

                # Extract question from chat messages
                question = ""
                params = ex.get("responses_create_params", {})
                msgs = params.get("input", [])
                for m in msgs:
                    if m.get("role") == "user":
                        question = m["content"]
                        break
                if question and answer:
                    all_examples.append({
                        "question": question.strip(),
                        "answer": answer,
                        "source": "stem_mcqa_mdrl",
                        "pass_rate": ex.get("pass_rate"),
                    })
                    mdrl_count += 1
    print(f"  multi-domain-RL STEM: {mdrl_count} examples")
    print(f"  Total: {len(all_examples)} examples")

    if not all_examples:
        print("  WARNING: No examples found!")
        return None

    # Convert to HF dataset
    from collections import Counter
    source_counts = Counter(ex["source"] for ex in all_examples)
    for src, cnt in source_counts.most_common():
        print(f"    {src}: {cnt}")

    def make_record(ex, idx):
        question = ex["question"]
        # Add boxed instruction if not already present
        if r"\boxed" not in question.lower():
            question = question + " " + INSTRUCTION
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ex["answer"]},
            "extra_info": {
                "split": "train",
                "index": str(idx),
                "solution": "",
                "has_solution": False,
                "source": ex["source"],
                "pass_rate": str(ex.get("pass_rate", "")),
            },
        }

    records = [make_record(ex, i) for i, ex in enumerate(all_examples)]
    dataset = hf_datasets.Dataset.from_list(records)

    # Filter empty answers
    before = len(dataset)
    dataset = dataset.filter(lambda x: bool(x["reward_model"]["ground_truth"]))
    after = len(dataset)
    if before != after:
        print(f"  Filtered {before - after} empty answers")
    print(f"  Final: {after} examples")

    _save_dataset(dataset, output_dir / "nemotron-math", "train")
    return dataset


def process_hmmt25(output_dir: Path):
    """MathArena/hmmt_feb_2025 — {problem, answer} (HMMT February 2025)"""
    print("\n" + "=" * 60)
    print("MathArena/hmmt_feb_2025 (test)")
    print("=" * 60)

    data_source = "MathArena/hmmt_feb_2025"
    # Try common split names; MathArena datasets usually have a single split
    dataset = None
    for split in ("test", "train", "default"):
        try:
            dataset = hf_datasets.load_dataset(data_source, split=split)
            break
        except Exception:
            continue
    if dataset is None:
        dataset = hf_datasets.load_dataset(data_source)
        # take first available split
        dataset = dataset[list(dataset.keys())[0]]
    print(f"  Loaded {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")

    # Field name normalization
    problem_key = "problem" if "problem" in dataset.column_names else "question"
    answer_key = "answer" if "answer" in dataset.column_names else "final_answer"

    def process_fn(example, idx):
        question = example[problem_key] + " " + INSTRUCTION
        answer = normalize_answer(example.get(answer_key))
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": str(idx)},
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "hmmt25", "test")
    return processed


def process_beyondaime(output_dir: Path):
    """ByteDance-Seed/BeyondAIME — {problem, answer}"""
    print("\n" + "=" * 60)
    print("ByteDance-Seed/BeyondAIME (test)")
    print("=" * 60)

    data_source = "ByteDance-Seed/BeyondAIME"
    dataset = hf_datasets.load_dataset(data_source, split="test")
    print(f"  Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        question = example["problem"] + " " + INSTRUCTION
        answer = normalize_answer(example["answer"])
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": str(idx)},
        }

    processed = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    _save_dataset(processed, output_dir / "beyondaime", "test")
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "deepscaler", "openthoughts", "deepmath", "dapomath", "nemotron_math",
    "amc23", "aime_2024", "aime25", "aime26", "math500",
    "minervamath", "olympiadbench", "beyondaime", "hmmt25",
]

PROCESSORS = {
    "deepscaler": process_deepscaler,
    "openthoughts": process_openthoughts,
    "deepmath": process_deepmath,
    "dapomath": process_dapomath,
    "nemotron_math": process_nemotron_math,
    "amc23": process_amc23,
    "aime_2024": process_aime_2024,
    "aime25": process_aime25,
    "aime26": process_aime26,
    "math500": process_math500,
    "minervamath": process_minervamath,
    "olympiadbench": process_olympiadbench,
    "beyondaime": process_beyondaime,
    "hmmt25": process_hmmt25,
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess math datasets for AntiSD")
    parser.add_argument(
        "--output_dir", type=str, default="datasets/math",
        help="Output directory (default: datasets/math)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS + ["all"], default=["all"],
        help="Which datasets to process (default: all)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=30000,
        help="Max samples for OpenThoughts math subset (default: 30000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    to_process = ALL_DATASETS if "all" in args.datasets else args.datasets

    print(f"Output: {output_dir}")
    print(f"Datasets: {to_process}")

    for name in to_process:
        if name == "openthoughts":
            PROCESSORS[name](output_dir, max_samples=args.max_samples)
        else:
            PROCESSORS[name](output_dir)

    # Print usage hint
    train_paths = []
    for d in ["deepscaler", "openthoughts", "deepmath", "dapo-math", "nemotron-math"]:
        p = output_dir / d / "train.parquet"
        if p.exists():
            train_paths.append(str(p))
    test_paths = [
        str(output_dir / d / "test.parquet")
        for d in ["amc23", "aime25", "aime26", "aime_2024", "math500", "minervamath", "olympiadbench", "beyondaime", "hmmt25"]
        if (output_dir / d / "test.parquet").exists()
    ]
    print("\n" + "=" * 60)
    print("Done! Usage in training scripts:")
    for tp in train_paths:
        print(f"  data.train_files=[{tp}]")
    print(f"  data.val_files=[{','.join(test_paths)}]")


if __name__ == "__main__":
    main()
