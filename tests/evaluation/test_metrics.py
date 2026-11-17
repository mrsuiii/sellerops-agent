"""pass@k (any trial passes) vs pass^k (all trials pass)."""

from __future__ import annotations

import pytest

from sellerops.evaluation.metrics import pass_hat_metrics, pass_metrics


@pytest.fixture
def samples():
    # task a: 2 of 4 trials pass; task b: 4 of 4.
    return [("a", 1), ("a", 0), ("a", 1), ("a", 0), ("b", 1), ("b", 1), ("b", 1), ("b", 1)]


def test_pass_at_k(samples) -> None:
    result = pass_metrics(samples, repetitions=4, k=2, task_ids=["a", "b"])
    assert result["pass@1"] == pytest.approx(0.75)
    assert result["pass@2"] == pytest.approx((1 - 1 / 6 + 1) / 2)


def test_pass_hat_k(samples) -> None:
    result = pass_hat_metrics(samples, repetitions=4, k=2, task_ids=["a", "b"])
    assert result["pass^1"] == pytest.approx(0.75)
    assert result["pass^2"] == pytest.approx((1 / 6 + 1) / 2)


def test_incomplete_trials_invalidate(samples) -> None:
    assert pass_hat_metrics(samples[:-1], repetitions=4, k=2, task_ids=["a", "b"]) == {"pass^1": None, "pass^2": None}
    assert pass_metrics([("a", None), ("a", 1)], repetitions=2, k=2, task_ids=["a"]) == {"pass@1": None, "pass@2": None}
