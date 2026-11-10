"""Task-level pass@1/pass@k, shared by local reports and LangSmith summaries."""

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


def make_pass_summary(*, repetitions, k, task_ids, include_judge=False):
    """Publish experiment-level scores, rather than averaging per-run pass@k."""
    from .evaluators import grade_episode

    def pass_summary(runs, examples):
        samples = []
        judged_samples = []
        for run, example in zip(runs, examples, strict=False):
            score = None
            if not run.error and run.outputs:
                try:
                    report = grade_episode(example.outputs["task"], run.outputs)
                    score = int(report["decision"] == "pass")
                except Exception:
                    # The row evaluator reports details. An unknown outcome must
                    # invalidate the aggregate, never masquerade as agent failure.
                    pass
            samples.append((example.inputs["task_id"], score))
            if include_judge:
                from .judge import end_to_end_score
                judged_samples.append((example.inputs["task_id"],
                                       None if score is None else end_to_end_score(score, run.outputs)))
        metrics = pass_metrics(samples, repetitions=repetitions, k=k, task_ids=task_ids)
        if include_judge:
            metrics.update({f"end_to_end_{key}": value for key, value in pass_metrics(
                judged_samples, repetitions=repetitions, k=k, task_ids=task_ids).items()})
        return {"results": [
            {"key": key, "score": value, "comment": (
                "Incomplete or unscored trials" if value is None else
                f"Task-macro estimator; n={repetitions}; " +
                ("structured + prose judge" if key.startswith("end_to_end_") else "structured outcomes")
            )} for key, value in metrics.items()
        ]}

    return pass_summary
