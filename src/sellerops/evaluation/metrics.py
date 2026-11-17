"""Task-level pass@k (any of k trials passes) and pass^k (all k trials pass)."""

from collections import defaultdict
from math import comb
from statistics import mean


def validate_repetitions(repetitions, k):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(k) is not int or not 1 <= k <= repetitions:
        raise ValueError("pass_k must be an integer between 1 and repetitions")


def pass_metrics(samples, *, repetitions, k, task_ids):
    """samples = (task_id, binary_score_or_None); None means evaluator failure.

    For each task with n trials and c successes: 1 - C(n-c,k)/C(n,k).
    Average tasks equally. Never drop missing/unscored trials from denominators.
    See https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
    """
    validate_repetitions(repetitions, k)
    grouped = defaultdict(list)
    for task_id, score in samples:
        grouped[task_id].append(score)
    keys = sorted({1, k})
    invalid = {f"pass@{size}": None for size in keys}
    if not task_ids or set(grouped) != set(task_ids):
        return invalid
    if any(len(scores) != repetitions or any(s not in (0, 1) for s in scores)
           for scores in grouped.values()):
        return invalid
    return {
        f"pass@{size}": mean(
            1 - comb(repetitions - int(sum(scores)), size) / comb(repetitions, size)
            for scores in grouped.values()
        ) for size in keys
    }


def pass_hat_metrics(samples, *, repetitions, k, task_ids):
    """τ-bench pass^k: probability that k independent trials of a task ALL pass.

    For each task with n trials and c successes: C(c,k)/C(n,k); tasks averaged
    equally. pass^1 equals pass@1; pass^k falls as k grows unless the agent is
    consistent. Same input contract as ``pass_metrics``.
    """
    validate_repetitions(repetitions, k)
    grouped = defaultdict(list)
    for task_id, score in samples:
        grouped[task_id].append(score)
    keys = sorted({1, k})
    invalid = {f"pass^{size}": None for size in keys}
    if not task_ids or set(grouped) != set(task_ids):
        return invalid
    if any(len(scores) != repetitions or any(s not in (0, 1) for s in scores)
           for scores in grouped.values()):
        return invalid
    return {
        f"pass^{size}": mean(comb(int(sum(scores)), size) / comb(repetitions, size)
                             for scores in grouped.values())
        for size in keys
    }
