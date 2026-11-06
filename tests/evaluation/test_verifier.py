"""Outcome verification checks using hand-built cases and train/dev data only."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from sellerops.environment import SellerEnvironment
from sellerops.schemas import validate_evaluation_result
from sellerops.services.episode_service.episode_service import EpisodeService
from sellerops.services.episode_service.episode_state import EpisodeState
from sellerops.evaluation import verify_outcome


def task_spec(family: str, **context: Any) -> dict[str, Any]:
    """Build trusted task constraints without supplying any oracle answers.

    Args:
        family: One of the three supported task families.
        **context: Structured task requirements.

    Returns:
        Minimal task accepted by the verifier.
    """
    return {"task_id": "hand-built", "family": family, "context": context}


def action(tool: str, **arguments: Any) -> dict[str, Any]:
    """Build a fresh tool call.

    Args:
        tool: Tool name.
        **arguments: Tool inputs.

    Returns:
        Tagged tool action.
    """
    return {"type": "tool_call", "tool": tool, "arguments": arguments}


def final(data: dict[str, Any], status: str = "success") -> dict[str, Any]:
    """Build a candidate final action with wording unrelated to oracle messages.

    Args:
        data: Candidate structured outcome facts.
        status: Candidate status.

    Returns:
        Tagged final answer.
    """
    return {"type": "final", "status": status, "message": "Hasil pemeriksaan.", "data": data}


def record(task, calls, environment=None) -> dict[str, Any]:
    """Execute real tools and export their evidence for one task.

    Args:
        task: Trusted task specification.
        calls: Tool actions to execute in the supplied order.
        environment: Optional custom starting environment.

    Returns:
        A runtime snapshot; no expected observations are fabricated.
    """
    service = EpisodeService()
    episode = EpisodeState("test-session", task["task_id"], environment=environment or SellerEnvironment())
    service.episodes[episode.session_id] = episode
    for call in calls:
        service.execute_tool(episode.session_id, call)
    return service.snapshot(episode.session_id)


def assert_reason(task, snapshot, candidate, expected: str) -> dict[str, Any]:
    """Check evaluation schema and the stable reason code.

    Args:
        task: Task constraints.
        snapshot: Runtime evidence.
        candidate: Final/clarification action.
        expected: Expected reason code.

    Returns:
        The evaluation result for optional further assertions.
    """
    result = verify_outcome(task, snapshot, candidate)
    validate_evaluation_result(result)
    assert result["reason_code"] == expected, result
    assert result["decision"] == ("pass" if expected == "outcome_valid" else "fail")
    return result


@pytest.fixture
def offer_case():
    """Return a compliant offer with real inventory, policy, and pricing evidence.

    Returns:
        Task, ordered calls, snapshot, and expected structured answer.
    """
    task = task_spec("inventory_offer", product_id="P1001", quantity=2, requested_discount_percent=10)
    calls = [action("get_inventory", product_id="P1001"),
             action("get_discount_policy", product_id="P1001"),
             action("calculate_offer", product_id="P1001", quantity=2, discount_percent=10)]
    snapshot = record(task, calls)
    answer = final({"product_id": "P1001", "quantity": 2, "discount_percent": 10,
                    "unit_price_idr": 224100, "total_idr": 448200})
    return task, calls, snapshot, answer


def test_valid_lookup_permutations_and_extra_reads_pass(offer_case) -> None:
    """Two lookup orders and an extra read can produce the same valid outcome.

    Args:
        offer_case: Compliant offer fixture.
    """
    task, calls, snapshot, answer = offer_case
    task["trajectory"] = "must never be read"
    task["verifier"] = {"expected_action_types": [], "expected_final_status": "not_found"}
    before = deepcopy((task, snapshot, answer))
    assert_reason(task, snapshot, answer, "outcome_valid")
    alternate = record(task, [calls[1], calls[0], calls[0], calls[2]])
    assert_reason(task, alternate, answer, "outcome_valid")
    assert (task, snapshot, answer) == before


@pytest.mark.parametrize("field,value,reason", [
    ("product_id", "P1003", "data_mismatch"), ("quantity", 3, "data_mismatch"),
    ("quantity", True, "data_mismatch"), ("total_idr", 1, "data_mismatch"),
    ("unit_price_idr", 249000, "data_mismatch"), ("discount_percent", 11, "policy_violation"),
    ("discount_percent", 9, "data_mismatch"), ("discount_percent", -1, "data_mismatch"),
])
def test_offer_rejects_incorrect_facts(offer_case, field, value, reason) -> None:
    """Correct tool history does not excuse an incorrect final fact.

    Args:
        offer_case: Compliant offer fixture.
        field: Fact to corrupt.
        value: Incorrect value.
        reason: Expected failure reason.
    """
    task, _, snapshot, answer = offer_case
    answer["data"][field] = value
    assert_reason(task, snapshot, answer, reason)


def test_offer_requires_product_specific_evidence_before_calculation(offer_case) -> None:
    """Correct guesses, missing checks, wrong-product checks, and late checks fail.

    Args:
        offer_case: Compliant offer fixture.
    """
    task, calls, _, answer = offer_case
    for sequence in ([], calls[:2], [calls[0], calls[2]],
                     [action("get_inventory", product_id="P1003"), calls[1], calls[2]]):
        assert_reason(task, record(task, sequence), answer, "missing_tool_evidence")
    assert_reason(task, record(task, [calls[2], calls[0], calls[1]]), answer, "missing_prior_checks")


def test_policy_violation_attempt_is_not_erased_by_safe_final(offer_case) -> None:
    """A blocked excessive-discount attempt remains a policy failure.

    Args:
        offer_case: Compliant offer fixture.
    """
    task, calls, _, answer = offer_case
    unsafe = action("calculate_offer", product_id="P1001", quantity=2, discount_percent=11)
    assert_reason(task, record(task, [*calls[:2], unsafe, calls[2]]), answer, "policy_violation")
    task["context"]["requested_discount_percent"] = 20
    assert_reason(task, record(task, calls), answer, "outcome_valid")


@pytest.mark.parametrize("product_id,quantity,mode,status", [
    ("P1002", 1, "offer", "unavailable"), ("P1001", 8, "offer", "unavailable"),
    ("P1001", 1, "stock_only", "success"),
])
def test_stock_and_unavailable_outcomes(product_id, quantity, mode, status) -> None:
    """Stock-only and insufficient-stock requests cannot produce offers.

    Args:
        product_id: Product to check.
        quantity: Requested quantity.
        mode: Offer or stock-only task.
        status: Expected stock status.
    """
    task = task_spec("inventory_offer", product_id=product_id, quantity=quantity,
                     mode=mode, requested_discount_percent=0)
    calls = [action("get_inventory", product_id=product_id)]
    snapshot = record(task, calls)
    data = {"product_id": product_id, "available_stock": snapshot["history"][0]["result"]["result"]["stock"]}
    if status == "unavailable":
        data["requested_quantity"] = quantity
    candidate = final(data, status)
    assert_reason(task, snapshot, candidate, "outcome_valid")
    calls.append(action("calculate_offer", product_id=product_id, quantity=quantity, discount_percent=0))
    assert_reason(task, record(task, calls), candidate, "unexpected_offer")


def test_custom_environment_drives_outcome_instead_of_default_prices(offer_case) -> None:
    """Verifier calculates facts from the actual initial episode data.

    Args:
        offer_case: Compliant offer fixture to adapt to custom prices.
    """
    task, calls, _, answer = offer_case
    environment = SellerEnvironment()
    environment.catalog["P1001"]["price_idr"] = 100000
    snapshot = record(task, calls, environment)
    assert_reason(task, snapshot, answer, "data_mismatch")
    answer["data"].update(unit_price_idr=90000, total_idr=180000)
    assert_reason(task, snapshot, answer, "outcome_valid")


def test_search_accepts_subsets_and_order_but_enforces_constraints() -> None:
    """Recommendations may differ from the oracle while remaining grounded.
    """
    task = task_spec("product_search", query="accessories", filters={"in_stock": True})
    snapshot = record(task, [action("search_products", query="accessories", filters={})])
    products = snapshot["history"][0]["result"]["result"]["products"]
    available = [p for p in products if p["stock"] > 0]
    assert_reason(task, snapshot, final({"products": available[::-1]}), "outcome_valid")
    assert_reason(task, snapshot, final({"products": available[:1]}), "outcome_valid")
    assert_reason(task, snapshot, final({"products": [available[0], available[0]]}), "duplicate_product")
    assert_reason(task, snapshot, final({"products": products}), "product_constraint_violation")
    assert_reason(task, snapshot, final({"products": []}, "not_found"), "status_mismatch")
    tampered = deepcopy(available[:1])
    tampered[0]["stock"] = 999
    assert_reason(task, snapshot, final({"products": tampered}), "data_mismatch")


@pytest.mark.parametrize("filters", [
    {"max_price_idr": 1}, {"brand": "Kepler"}, {"color": "silver"}, {"category": "audio"},
])
def test_search_does_not_accept_products_outside_task_filters(filters) -> None:
    """A tool result alone is insufficient when it violates the task filters.

    Args:
        filters: Search constraints excluding the mouse returned by a broad call.
    """
    task = task_spec("product_search", query="accessories", filters=filters)
    snapshot = record(task, [action("search_products", query="mouse", filters={})])
    candidate = final(snapshot["history"][0]["result"]["result"])
    assert verify_outcome(task, snapshot, candidate)["decision"] == "fail"


def test_search_requires_observed_products_and_truthful_not_found() -> None:
    """Unseen recommendations fail, and not_found succeeds only with no matches."""
    task = task_spec("product_search", query="accessories", filters={})
    snapshot = record(task, [action("search_products", query="keyboard", filters={})])
    mouse = {"product_id": "P1001", "name": "Orbit Wireless Mouse", "price_idr": 249000, "stock": 7}
    assert_reason(task, snapshot, final({"products": [mouse]}), "missing_tool_evidence")
    task["context"]["query"] = "nonexistent-item"
    snapshot = record(task, [action("search_products", query="nonexistent-item", filters={})])
    assert_reason(task, snapshot, final({"products": []}, "not_found"), "outcome_valid")


def test_clarification_checks_missing_fields_without_wording_match() -> None:
    """Missing information calls for the declared fields; wording can vary."""
    task = task_spec("product_search", missing_fields=["budget_idr"])
    snapshot = record(task, [])
    question = {"type": "ask_user", "question": "Budget maksimalnya berapa?", "missing_fields": ["budget_idr"]}
    assert_reason(task, snapshot, question, "outcome_valid")
    question["missing_fields"] = ["color"]
    assert_reason(task, snapshot, question, "missing_fields_mismatch")
    assert_reason(task, snapshot, final({"products": []}, "not_found"), "clarification_required")


@pytest.mark.parametrize("proposal,allowed", [(-1, False), (0, True), (10, True), (11, False)])
def test_policy_proposals_and_sources(proposal: int, allowed: bool) -> None:
    """Policy decisions respect both the lower bound and the product limit.

    Args:
        proposal: Requested percentage, including invalid negative values.
        allowed: Independently expected eligibility.
    """
    task = task_spec("policy_qa", product_id="P1001", question_type="proposal", proposed_discount_percent=proposal)
    snapshot = record(task, [action("get_discount_policy", product_id="P1001")])
    data = {"product_id": "P1001", "max_discount_percent": 10, "source": "discount-policy-v1#P1001",
            "proposed_discount_percent": proposal, "proposal_allowed": allowed}
    assert_reason(task, snapshot, final(data), "outcome_valid")
    data["source"] = "invented-source"
    assert_reason(task, snapshot, final(data), "data_mismatch")


@pytest.mark.parametrize("product_id,eligible,maximum", [("P1001", True, 10), ("P1007", False, 0)])
def test_policy_eligibility_requires_matching_evidence(product_id, eligible, maximum) -> None:
    """Eligibility is product-specific and cannot be guessed without policy tools.

    Args:
        product_id: Product whose eligibility is requested.
        eligible: Expected discount eligibility.
        maximum: Expected maximum discount.
    """
    task = task_spec("policy_qa", product_id=product_id, question_type="eligibility")
    candidate = final({"product_id": product_id, "max_discount_percent": maximum,
                       "source": f"discount-policy-v1#{product_id}", "discount_eligible": eligible})
    assert_reason(task, record(task, []), candidate, "missing_tool_evidence")
    snapshot = record(task, [action("get_discount_policy", product_id=product_id)])
    assert_reason(task, snapshot, candidate, "outcome_valid")
    candidate["data"]["discount_eligible"] = not eligible
    assert_reason(task, snapshot, candidate, "data_mismatch")


def test_snapshot_and_action_failures(offer_case) -> None:
    """Malformed, mismatched, and manipulated evidence cannot pass verification.

    Args:
        offer_case: Compliant offer fixture.
    """
    task, calls, snapshot, answer = offer_case
    assert_reason(task, snapshot, {"type": []}, "invalid_action")
    assert_reason(task, snapshot, calls[0], "incomplete_outcome")
    assert_reason(task, {}, answer, "invalid_snapshot")
    changed = deepcopy(snapshot)
    changed["task_id"] = "wrong-task"
    assert_reason(task, changed, answer, "task_mismatch")
    changed = deepcopy(snapshot)
    changed["history"][0]["result"]["result"]["stock"] = 999
    assert_reason(task, changed, answer, "history_mismatch")
    changed = deepcopy(snapshot)
    changed["environment"]["inventory"]["P1001"]["stock"] = 999
    assert_reason(task, changed, answer, "environment_changed")


def test_existing_train_dev_demonstrations_are_audited() -> None:
    """Audit train/dev examples without reading frozen tests or using oracle targets in verifier."""
    root = Path(__file__).resolve().parents[2] / "data" / "tasks"
    failures = []
    count = 0
    for split in ("train", "dev"):
        for line in (root / f"{split}.jsonl").read_text().splitlines():
            task = json.loads(line)
            calls = [e["content"] for e in task["trajectory"]
                     if e["role"] == "assistant" and e["content"]["type"] == "tool_call"]
            snapshot = record(task, calls)
            candidate = task["trajectory"][-1]["content"]
            report = verify_outcome(task, snapshot, candidate)
            validate_evaluation_result(report)
            count += 1
            if report["decision"] == "fail":
                # Known oracle bug: proposal_allowed checks only <= maximum.
                assert task["family"] == "policy_qa"
                assert task["context"]["proposed_discount_percent"] < 0
                assert candidate["data"]["proposal_allowed"] is True
                assert report["reason_code"] == "data_mismatch"
                corrected = deepcopy(candidate)
                corrected["data"]["proposal_allowed"] = False
                assert_reason(task, snapshot, corrected, "outcome_valid")
                failures.append(task["task_id"])
    assert count == 210
    assert failures == ["dev-policy-dev-check_request-id-002"]
