"""Demo LLM budget: per-client sliding hour, global UTC day, only LLM routes pay."""

from __future__ import annotations

import pytest

from sellerops.seller import SellerError
from sellerops.workflows.approval.budget import LLMBudget


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0   # a fixed UTC moment

    def __call__(self) -> float:
        return self.now


def test_per_client_hour_and_per_day() -> None:
    clock = Clock()
    budget = LLMBudget(per_client_hour=2, per_day=3, clock=clock)
    budget.take("a")
    budget.take("a")
    with pytest.raises(SellerError) as error:
        budget.take("a")
    assert error.value.code == "llm_budget_exhausted" and "per hour" in error.value.message
    budget.take("b")                                   # another client still may
    with pytest.raises(SellerError) as error:          # but the day is used up for everyone
        budget.take("c")
    assert "today" in error.value.message
    assert budget.remaining("a") == {"enabled": True, "per_client_hour": 2, "remaining_hour": 0, "remaining_today": 0}

    clock.now += 24 * 3600                              # next day, an hour later too
    budget.take("a")
    assert budget.remaining("a")["remaining_hour"] == 1


def test_disabled_budget_never_refuses() -> None:
    budget = LLMBudget(per_client_hour=0, per_day=0, enabled=False)
    budget.take("a")
    assert budget.remaining("a") == {"enabled": False}


def test_only_llm_routes_take_from_the_budget(service) -> None:
    pytest.importorskip("httpx")
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver

    from sellerops.seller import api as seller_api
    from sellerops.store.api import app
    from sellerops.workflows.approval import api as approval_api
    from sellerops.workflows.approval.runner import ApprovalRunner

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
    from fake_models import call, fake

    draft = call("CaptionDraft", {"description": "Short.", "reason": "x"})
    runner = ApprovalRunner(service, fake(draft, draft), InMemorySaver())
    budget = LLMBudget(per_client_hour=1, per_day=10)
    app.dependency_overrides[seller_api.get_service] = lambda: service
    app.dependency_overrides[approval_api.get_runner] = lambda: runner
    app.dependency_overrides[approval_api.default_budget] = lambda: budget
    headers = {"X-Seller-User": "s1"}
    try:
        client = TestClient(app)
        created = client.post("/api/seller/caption-requests", headers=headers, json={"product_id": "FS180"})
        assert created.status_code == 201
        pid = created.json()["request"]["proposal"]["proposal_id"]
        assert client.get("/api/seller/llm-budget").json()["remaining_hour"] == 0
        refused = client.post(f"/api/seller/caption-proposals/{pid}/approval/decision", headers=headers,
                              json={"decision": "reject", "feedback": "lagi"})
        assert refused.status_code == 429 and refused.json()["code"] == "llm_budget_exhausted"
        assert service.get_proposal(pid)["status"] == "pending"            # refused before anything changed
        approved = client.post(f"/api/seller/caption-proposals/{pid}/approval/decision", headers=headers,
                               json={"decision": "approve"})                # approving is free
        assert approved.json()["stop_reason"] == "applied"
    finally:
        app.dependency_overrides.clear()


def test_refused_requests_are_free(service) -> None:
    pytest.importorskip("httpx")
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver

    from sellerops.seller import api as seller_api
    from sellerops.store.api import app
    from sellerops.workflows.approval import api as approval_api
    from sellerops.workflows.approval.runner import ApprovalRunner

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
    from fake_models import fake

    pid = service.propose_caption("FS180", "r", actor="agent", description="Long.")["proposal_id"]
    runner = ApprovalRunner(service, fake(), InMemorySaver())
    budget = LLMBudget(per_client_hour=5, per_day=10)
    app.dependency_overrides[seller_api.get_service] = lambda: service
    app.dependency_overrides[approval_api.get_runner] = lambda: runner
    app.dependency_overrides[approval_api.default_budget] = lambda: budget
    headers = {"X-Seller-User": "s1"}
    try:
        client = TestClient(app)
        client.get(f"/api/seller/caption-proposals/{pid}/approval")
        assert client.post("/api/seller/caption-requests", headers=headers, json={"product_id": "NOPE"}).status_code == 404
        assert client.post(f"/api/seller/caption-proposals/{pid}/approval/answer", headers=headers,
                           json={"answers": {"material": "x"}}).status_code == 409   # not asking
        assert client.post(f"/api/seller/caption-proposals/{pid}/approval/retry", headers=headers).status_code == 409
        assert client.get("/api/seller/llm-budget").json()["remaining_hour"] == 5
        client.post(f"/api/seller/caption-proposals/{pid}/approval/decision", headers=headers,
                    json={"decision": "reject", "feedback": "x"})             # runs the LLM: paid
        assert client.get("/api/seller/llm-budget").json()["remaining_hour"] == 4
        stale = client.post(f"/api/seller/caption-proposals/{pid}/approval/decision", headers=headers,
                            json={"decision": "reject", "feedback": "x"})     # the old version: refused, free
        assert stale.status_code == 409 and client.get("/api/seller/llm-budget").json()["remaining_hour"] == 4
    finally:
        app.dependency_overrides.clear()
