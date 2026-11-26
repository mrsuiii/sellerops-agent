"""Regression tests from real eval failures (A3 error analysis, docs/error-analysis-2026-11-17.md).

Each fixture is a recorded conversation from a live eval run. The verifier
must keep failing it for the same reason, so a change to the verifier or the
answer keys cannot silently turn a known-bad answer into a pass. The prompt
checks guard the fixes that came out of the root-cause analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sellerops.evaluation.prompts import SELLER_SYSTEM_PROMPT
from sellerops.evaluation.seller_verifier import verify_conversation
from sellerops.harness.tools.seller import ProductPerformanceArgs, SalesSummaryArgs

FIXTURES = [json.loads(line) for line in
            (Path(__file__).parent / "fixtures" / "eval_regressions.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("case", FIXTURES, ids=[case["source"].split("/")[-1] for case in FIXTURES])
def test_recorded_failure_still_fails_for_the_same_reason(case, all_tasks) -> None:
    task = next(t for t in all_tasks if t["task_id"] == case["task_id"])
    snapshot = {"history": case["history"], "proposals": case["proposals"]}
    result = {"final_action": case["final_action"], "stop_reason": case["stop_reason"], "turns": case["turns"]}
    report = verify_conversation(task, snapshot, result)
    assert report["decision"] == "fail"
    assert case["failed_check"] in report["failed"], (case["why"], report["checks"])


@pytest.mark.parametrize(("fix", "text"), [
    ("RCA-1 caption: only requested fields, ask which", "Change only the"),
    ("RCA-1 caption: ask when fields are unspecified", "if they did not say"),
    ("RCA-2 clarification: all missing details in one question", "covering every missing detail"),
    ("RCA-2 clarification: product scope", "all products or one category"),
    ("RCA-3 trends from the tool", "compare_to_previous"),
])
def test_rca_fixes_stay_in_the_prompt(fix: str, text: str) -> None:
    assert text in " ".join(SELLER_SYSTEM_PROMPT.split()), fix


def test_rca_comparison_option_is_offered_to_the_model() -> None:
    for schema in (SalesSummaryArgs, ProductPerformanceArgs):
        assert "compare_to_previous" in schema.model_json_schema()["properties"]
