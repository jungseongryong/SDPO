"""SDPO-compatible reward scorer for code datasets.

Mirrors :mod:`verl.utils.reward_score.math_feedback`: returns a dict with the
``score``/``acc``/``pred``/``feedback``/``incorrect_format``/``truncated`` keys
that the SDPO self-distillation pipeline consumes.

Routing rule (matches the upstream `recipe/is_shape/code/reward_function.py`):

* ``data_source == "dolci"`` (and other assert-style sources)
  -> run the model code + each ground-truth assert in a sandbox subprocess.
* ``data_source == "livecodebench/code_generation_lite"`` (and other test-case
  sources) -> delegate to ``verl.utils.reward_score.prime_code``.
"""
from __future__ import annotations

import json
import multiprocessing
from typing import Any

# Datasets that ship asserts as ground_truth (Dolci-RLZero / HumanEval+ style).
ASSERT_CODE_SOURCES = {"dolci", "humanevalplus", "mbppplus"}
# Datasets that ship structured test cases (LCB / APPS / TACO style).
TEST_CASE_CODE_SOURCES = {
    "codecontests",
    "apps",
    "codeforces",
    "taco",
    "livecodebench/code_generation_lite",
}

# Cap on number of asserts to execute (prevents runaway eval cost).
MAX_TESTS = 20
# Per-assert wall-clock timeout in seconds.
ASSERT_TIMEOUT_S = 10


def _extract_python_code(text: str) -> str:
    """Pull out the model's python solution. Prefers ```python ...``` fences."""
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0]
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            return parts[1]
    return text


def _exec_one_assert(code_str: str, assert_str: str, result):
    try:
        ns: dict[str, Any] = {}
        exec(code_str, ns)  # noqa: S102
        exec(assert_str, ns)  # noqa: S102
        result.append(True)
    except Exception:
        result.append(False)


def _score_assert(solution_str: str, ground_truth) -> tuple[float, int, int, str]:
    """Return (fraction passed, passed, total, status string)."""
    code_solution = _extract_python_code(solution_str)

    asserts: list[str] = []
    try:
        if isinstance(ground_truth, str):
            parsed = json.loads(ground_truth)
            if isinstance(parsed, str):
                # Sometimes double-encoded; decode again.
                parsed = json.loads(parsed)
            asserts = parsed if isinstance(parsed, list) else [parsed]
        elif isinstance(ground_truth, (list, tuple)):
            asserts = list(ground_truth)
    except (json.JSONDecodeError, TypeError):
        if isinstance(ground_truth, str) and ground_truth.startswith("assert"):
            asserts = [ground_truth]

    if not asserts:
        return 0.0, 0, 0, "[NO_TESTS]"

    total = min(len(asserts), MAX_TESTS)
    passed = 0
    for i in range(total):
        manager = multiprocessing.Manager()
        result = manager.list()
        p = multiprocessing.Process(
            target=_exec_one_assert,
            args=(code_solution, str(asserts[i]), result),
        )
        p.start()
        p.join(timeout=ASSERT_TIMEOUT_S)
        if p.is_alive():
            p.kill()
            p.join()
        if result and result[0]:
            passed += 1

    return passed / total, passed, total, f"[ASSERT:{passed}/{total}]"


def _score_test_cases(solution_str: str, ground_truth) -> tuple[float, int, int, str]:
    """Delegate to prime_code. Handles compressed (base64+zlib+pickle) gt.

    Returns (fraction passed, passed, total, pred). passed/total are 0 when
    prime_code does not expose a metadata list (early all-pass shortcut or
    error path) — feedback should fall back to a count-free phrasing.
    """
    try:
        from verl.utils.reward_score.prime_code import compute_score as _prime
    except ImportError:
        return 0.0, 0, 0, "[CODE_EVAL_UNAVAILABLE]"

    if isinstance(ground_truth, str):
        try:
            json.loads(ground_truth)  # already JSON string -> use as-is
        except (json.JSONDecodeError, TypeError):
            try:
                import base64
                import pickle
                import zlib

                ground_truth = json.loads(
                    pickle.loads(zlib.decompress(base64.b64decode(ground_truth.encode("utf-8"))))
                )
            except Exception:  # noqa: BLE001 — let prime_code surface its own error
                pass

    try:
        result = _prime(solution_str, ground_truth, continuous=True)
    except Exception as e:  # noqa: BLE001
        return 0.0, 0, 0, f"[CODE_ERROR: {e!s}]"

    passed = 0
    total = 0
    if isinstance(result, tuple):
        success, meta = result
        score = float(success) if success is not False else 0.0
        # prime_code returns metadata_list (a list) on the per-sample path and
        # a single dict on the early all-pass shortcut. Only the list exposes
        # individual test outcomes — use it to recover (passed, total).
        if isinstance(meta, list) and meta:
            total = len(meta)
            for m in meta:
                tc = m.get("test_case") if isinstance(m, dict) else None
                res_str = tc.get("res") if isinstance(tc, dict) else None
                if isinstance(res_str, str) and "True" in res_str:
                    passed += 1
            # Sanity-clamp passed against the reported fraction, in case the
            # res string parsing missed an entry.
            if total > 0:
                passed = max(passed, int(round(score * total)))
                passed = min(passed, total)
    else:
        score = float(result)
    return score, passed, total, f"[CODE:{score:.2f}]"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
) -> dict:
    """Unified code-reward scorer compatible with SDPO's self-distillation."""
    extra_info = extra_info or {}
    was_truncated = bool(extra_info.get("truncated", False))

    src = (data_source or "").lower()
    if src in ASSERT_CODE_SOURCES:
        score, passed, total, pred = _score_assert(solution_str, ground_truth)
    elif src in TEST_CASE_CODE_SOURCES:
        score, passed, total, pred = _score_test_cases(solution_str, ground_truth)
    else:
        # Best-effort: if ground_truth looks like a list of asserts, try assert path.
        gt_str = ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth)
        if "assert " in gt_str:
            score, passed, total, pred = _score_assert(solution_str, ground_truth)
        else:
            score, passed, total, pred = _score_test_cases(solution_str, ground_truth)

    score = max(0.0, min(1.0, score))
    incorrect_format = ("```" not in solution_str) and ("def " not in solution_str)

    # Continuous-granularity feedback: mirrors the original recipe/is_shape
    # reward signal (fraction of test cases passed) instead of collapsing to
    # binary all-pass / not-all-pass.
    if total > 0:
        if passed >= total:
            feedback = f"Your code passes all {total} test cases."
        elif passed == 0:
            feedback = f"Your code does not pass any of the {total} test cases."
        else:
            feedback = f"Your code passes {passed} of {total} test cases."
    else:
        # Count unavailable (prime_code all-pass shortcut, eval error, no tests).
        if score >= 1.0:
            feedback = "Your code passes all test cases."
        elif score <= 0.0:
            feedback = "Your code does not pass any test cases."
        else:
            pct = int(round(score * 100))
            feedback = f"Your code passes about {pct}% of the test cases."

    return {
        "score": float(score),
        "acc": float(score),
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if (incorrect_format and was_truncated) else 0,
        "feedback": feedback,
    }
