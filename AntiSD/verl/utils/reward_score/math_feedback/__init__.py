"""Math reward function with math_verify for equivalence checking.

Uses math_verify (https://github.com/huggingface/Math-Verify) as the primary
verification method, which handles LaTeX parsing and mathematical equivalence.

Usage in training script:
    custom_reward_function.path=verl/utils/reward_score/math_feedback/__init__.py
"""

from math_verify import parse as mv_parse, verify as mv_verify


def _extract_last_boxed(text: str) -> str | None:
    r"""Extract content of the last \boxed{...} in text, handling nested braces."""
    last_start = text.rfind(r"\boxed{")
    if last_start == -1:
        return None
    depth = 0
    i = last_start + len(r"\boxed{")
    content_start = i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[content_start:i]
            depth -= 1
        i += 1
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if extra_info is None:
        extra_info = {}

    was_truncated = extra_info.get("truncated", False)

    # Extract only the last \boxed{} as the model's final answer
    # This prevents intermediate calculations (e.g. \boxed{12} for "12 edges")
    # from being mistaken as the final answer.
    last_boxed = _extract_last_boxed(solution_str)

    # Use math_verify for parsing and equivalence checking.
    # Only evaluate if a \boxed{} was found; no fallback to full-text parsing.
    # Wrap BOTH ground_truth and prediction in \boxed{} before parsing — math_verify's
    # raw-mode parser fails on many non-integer formats (\dfrac{a}{b}, \infty, Yes, etc.),
    # causing ~46% false-negatives on deepmath-style datasets. Wrapping in \boxed{}
    # provides a consistent parsing context and restores 100% self-verify rate while
    # being a no-op for integer-only datasets (e.g. dapo-math).
    correct = False
    pred = ""
    if last_boxed is not None:
        try:
            gold_parsed = mv_parse("\\boxed{" + ground_truth + "}", parsing_timeout=30)
            pred_parsed = mv_parse("\\boxed{" + last_boxed + "}", parsing_timeout=30)
            if pred_parsed:
                pred = str(pred_parsed[1]) if len(pred_parsed) > 1 else str(pred_parsed[0])
            if gold_parsed and pred_parsed:
                correct = mv_verify(gold_parsed, pred_parsed, timeout_seconds=30)
        except Exception:
            pass

    reward = 1.0 if correct else 0.0
    score = reward
    incorrect_format = pred == ""

    # Natural-language feedback — keep it simple to avoid leaking meta-information
    # (e.g. truncation warnings that create length-explosion feedback loops).
    if correct:
        feedback = "Your answer is correct."
    else:
        feedback = "Your answer is incorrect."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
