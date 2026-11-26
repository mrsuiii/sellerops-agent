"""Caption approval over HTTP: /caption-proposals/{id}/approval (runner + routes)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from sellerops.seller import api as seller_api
from sellerops.store.api import app
from sellerops.workflows.approval import api as approval_api
from sellerops.workflows.approval.runner import ApprovalRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from fake_models import call, fake  # noqa: E402

HEADERS = {"X-Seller-User": "seller-1"}
DRAFT = lambda text: call("CaptionDraft", {"description": text, "reason": "lebih pendek"})  # noqa: E731


@pytest.fixture
def make_client(service):
    def build(*script):
        runner = ApprovalRunner(service, fake(*script), InMemorySaver())
        app.dependency_overrides[seller_api.get_service] = lambda: service
        app.dependency_overrides[approval_api.get_runner] = lambda: runner
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def propose(service) -> str:
    return service.propose_caption("FS180", "more engaging", actor="agent",
                                   description="A very long description.")["proposal_id"]


def url(pid: str, action: str = "") -> str:
    return f"/api/seller/caption-proposals/{pid}/approval" + (f"/{action}" if action else "")


def test_first_read_starts_the_thread(make_client, service) -> None:
    client = make_client()
    pid = propose(service)
    body = client.get(url(pid)).json()
    assert body["state"] == "waiting" and body["proposal_id"] == pid
    assert body["request"]["proposal"]["proposal_id"] == pid and body["request"]["fields"] == ["description"]
    assert body["thread_id"] == f"{service.store_id}:caption-approval-{pid}"
    assert client.get(url(pid)).json() == body   # reading again changes nothing


def test_reject_revise_approve(make_client, service) -> None:
    client = make_client(DRAFT("Short and sweet."))
    pid = propose(service)

    rejected = client.post(url(pid, "decision"), headers=HEADERS,
                           json={"decision": "reject", "feedback": "terlalu panjang"}).json()
    revision = rejected["proposal_id"]
    assert rejected["state"] == "waiting" and revision != pid and rejected["rejections"] == 1
    assert rejected["request"]["proposal"]["changes"]["description"]["new"] == "Short and sweet."
    assert service.get_proposal(pid)["decided_by"] == "seller:seller-1"

    # the same thread from either end of the chain; the old version can no longer be decided
    assert client.get(url(revision)).json()["thread_id"] == rejected["thread_id"]
    stale = client.post(url(pid, "decision"), headers=HEADERS, json={"decision": "approve"})
    assert stale.status_code == 409 and stale.json()["code"] == "stale_decision"

    done = client.post(url(revision, "decision"), headers=HEADERS, json={"decision": "approve"}).json()
    assert (done["state"], done["stop_reason"]) == ("done", "applied")
    assert service.analytics().product("FS180")["description"] == "Short and sweet."

    closed = client.post(url(revision, "decision"), headers=HEADERS, json={"decision": "cancel"})
    assert closed.status_code == 409 and closed.json()["code"] == "approval_closed"


def test_requests_are_validated(make_client, service) -> None:
    client = make_client()
    pid = propose(service)
    assert client.post(url(pid, "decision"), json={"decision": "approve"}).status_code == 422   # no X-Seller-User
    assert client.post(url(pid, "decision"), headers=HEADERS,
                       json={"decision": "approve", "add_fields": ["name"]}).status_code == 422
    assert client.post(url(pid, "decision"), headers=HEADERS, json={"decision": "maybe"}).status_code == 422
    missing = client.get(url("CP-9999"))
    assert missing.status_code == 404 and missing.json()["code"] == "proposal_not_found"
    assert service.get_proposal(pid)["status"] == "pending"


def test_decided_outside_the_graph(make_client, service) -> None:
    client = make_client()
    pid = propose(service)
    service.approve(pid, actor="seller:seller-1")          # plain route, no thread
    response = client.get(url(pid))
    assert response.status_code == 404 and response.json()["code"] == "approval_not_found"


def test_chain_started_outside_the_graph(make_client, service) -> None:
    """A revision proposed without the graph still gets a thread, named after the chain's root."""
    client = make_client()
    pid = propose(service)
    service.reject(pid, actor="seller:seller-1", feedback="shorter")
    revision = service.propose_caption("FS180", "shorter", actor="agent", description="Short.",
                                       revision_of=pid)["proposal_id"]
    body = client.get(url(revision)).json()
    assert (body["state"], body["proposal_id"], body["root_proposal_id"], body["rejections"]) == (
        "waiting", revision, pid, 1)


def test_failed_step_can_be_retried(make_client, service) -> None:
    client = make_client(RuntimeError("model down"), DRAFT("Second try."))
    pid = propose(service)
    failed = client.post(url(pid, "decision"), headers=HEADERS, json={"decision": "reject", "feedback": "x"}).json()
    assert failed["state"] == "error" and "draft_revision" in failed["error"] and "model down" not in failed["error"]
    assert service.get_proposal(pid)["status"] == "rejected"   # the rejection was recorded before the LLM step

    nothing = client.post(url(pid, "decision"), headers=HEADERS, json={"decision": "approve"})
    assert nothing.status_code == 409 and nothing.json()["code"] == "approval_closed"

    retried = client.post(url(pid, "retry"), headers=HEADERS).json()
    assert retried["state"] == "waiting"
    assert retried["request"]["proposal"]["changes"]["description"]["new"] == "Second try."
    again = client.post(url(retried["proposal_id"], "retry"), headers=HEADERS)
    assert again.status_code == 409 and again.json()["code"] == "approval_not_failed"
