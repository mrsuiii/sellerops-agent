"""C3b runner + API: caption requests and questions to the seller.

The real approval graph cannot ask yet (that node is Nigka's), so these tests
drive the runner with a small stub graph that follows the contract in
``workflows/approval/questions.py``: input ``{"request_id"}``, a
``kind="question"`` interrupt, a first proposal with ``request_id``, then a
decision interrupt.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from sellerops.seller import api as seller_api
from sellerops.store.api import app
from sellerops.workflows.approval import api as approval_api
from sellerops.workflows.approval.questions import Answer, SellerQuestion, question_request
from sellerops.workflows.approval.runner import ApprovalRunner

HEADERS = {"X-Seller-User": "seller-1"}


class StubState(TypedDict, total=False):
    request_id: str
    product_id: str
    proposal_id: str
    answers: dict
    decision: str


def stub_graph(service, checkpointer, *, fail_once: list[bool] | None = None):
    """load → ask (interrupt question) → propose → wait (interrupt decision) → END."""

    def load(state: StubState) -> dict[str, Any]:
        return {"product_id": service.get_request(state["request_id"])["product_id"]}

    def ask(state: StubState) -> dict[str, Any]:
        raw = interrupt(question_request(
            [SellerQuestion(key="material", question="Bahannya apa?"),
             SellerQuestion(key="fit", question="Potongannya?")],
            product_id=state["product_id"], request_id=state["request_id"]))
        return {"answers": Answer.model_validate(raw).answers}

    def propose(state: StubState) -> dict[str, Any]:
        if fail_once and fail_once.pop():
            raise RuntimeError("LLM down")
        facts = {k: v["value"] for k, v in service.product_facts(state["product_id"]).items()}
        text = "Dress " + (facts.get("material") or "(bahan perlu dilengkapi)")
        proposal = service.propose_caption(state["product_id"], "draf pertama", actor="agent:caption-approval",
                                           description=text, request_id=state["request_id"],
                                           inputs={"seller_facts": facts})
        return {"proposal_id": proposal["proposal_id"]}

    def wait(state: StubState) -> dict[str, Any]:
        answer = interrupt({"kind": "decision", "proposal": service.get_proposal(state["proposal_id"]),
                            "fields": ["description"], "rejections": 0})
        return {"decision": answer["decision"]}

    builder = StateGraph(StubState)
    for name, node in [("load", load), ("ask", ask), ("propose", propose), ("wait", wait)]:
        builder.add_node(name, node)
    builder.add_edge(START, "load")
    builder.add_edge("load", "ask")
    builder.add_edge("ask", "propose")
    builder.add_edge("propose", "wait")
    builder.add_edge("wait", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.fixture
def make_client(service):
    def build(**kwargs):
        saver = InMemorySaver()
        runner = ApprovalRunner(service, None, saver, graph=stub_graph(service, saver, **kwargs))  # type: ignore[arg-type]
        app.dependency_overrides[seller_api.get_service] = lambda: service
        app.dependency_overrides[approval_api.get_runner] = lambda: runner
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def start(client, **body) -> dict[str, Any]:
    response = client.post("/api/seller/caption-requests", headers=HEADERS,
                           json={"product_id": "FS180", "instruction": "sebut bahannya"} | body)
    assert response.status_code == 201, response.text
    return response.json()


def test_request_asks_then_proposes_then_decides(make_client, service) -> None:
    client = make_client()
    view = start(client)
    assert (view["state"], view["waiting_for"], view["request_id"]) == ("waiting", "answer", "CR-0001")
    assert view["thread_id"] == f"{service.store_id}:caption-request-CR-0001"
    assert [q["key"] for q in view["request"]["questions"]] == ["material", "fit"]
    assert view["root_proposal_id"] is None
    assert client.get("/api/seller/caption-requests/CR-0001/approval").json() == view   # reading changes nothing

    answered = client.post("/api/seller/caption-requests/CR-0001/approval/answer", headers=HEADERS,
                           json={"answers": {"material": "katun", "fit": "  "}}).json()
    assert (answered["state"], answered["waiting_for"]) == ("waiting", "decision")
    pid = answered["proposal_id"]
    assert answered["root_proposal_id"] == pid

    # memory first: the facts are in the service, with who said them and where
    facts = service.product_facts("FS180")
    assert facts["material"] == {"value": "katun", "source": "CR-0001", "at": facts["material"]["at"],
                                 "by": "seller:seller-1"}
    assert facts["fit"]["value"] is None
    proposal = service.get_proposal(pid)
    assert proposal["request_id"] == "CR-0001" and proposal["inputs"]["seller_facts"]["material"] == "katun"
    assert service.get_request("CR-0001")["status"] == "proposed"

    # the proposal resolves to the request's thread
    by_proposal = client.get(f"/api/seller/caption-proposals/{pid}/approval").json()
    assert by_proposal["thread_id"] == answered["thread_id"]
    done = client.post(f"/api/seller/caption-proposals/{pid}/approval/decision", headers=HEADERS,
                       json={"decision": "approve"}).json()
    assert done["state"] == "done"


def test_answers_must_match_the_questions(make_client) -> None:
    client = make_client()
    start(client)
    url = "/api/seller/caption-requests/CR-0001/approval/answer"
    for body, status in [({"answers": {"material": "katun"}}, 422),               # fit missing
                         ({"answers": {"material": "a", "fit": "b", "x": "c"}}, 422),
                         ({"answers": {"Material": "a"}}, 422),
                         ({"answers": {}}, 422)]:
        assert client.post(url, headers=HEADERS, json=body).status_code == status
    assert client.post(url, json={"answers": {"material": "a", "fit": "b"}}).status_code == 422   # no user
    # waiting for an answer, not a decision
    decision = client.post("/api/seller/caption-proposals/CP-0001/approval/decision", headers=HEADERS,
                           json={"decision": "approve"})
    assert decision.status_code == 404                                          # no proposal exists yet


def test_decision_while_waiting_for_answer_is_refused(make_client, service) -> None:
    client = make_client()
    start(client)
    client.post("/api/seller/caption-requests/CR-0001/approval/answer", headers=HEADERS,
                json={"answers": {"material": "katun", "fit": "slim"}})
    again = client.post("/api/seller/caption-requests/CR-0001/approval/answer", headers=HEADERS,
                        json={"answers": {"material": "linen", "fit": "slim"}})
    assert again.status_code == 409 and again.json()["code"] == "not_waiting_for_answer"
    assert service.product_facts("FS180")["material"]["value"] == "katun"      # nothing stored on a refusal


def test_failed_step_after_answer_keeps_the_facts(make_client, service) -> None:
    client = make_client(fail_once=[True])
    start(client)
    failed = client.post("/api/seller/caption-requests/CR-0001/approval/answer", headers=HEADERS,
                         json={"answers": {"material": "katun", "fit": "slim"}}).json()
    assert failed["state"] == "error" and service.product_facts("FS180")["material"]["value"] == "katun"
    retried = client.post("/api/seller/caption-requests/CR-0001/approval/retry", headers=HEADERS).json()
    assert retried["waiting_for"] == "decision"


def test_cancelled_request_closes_its_thread(make_client) -> None:
    client = make_client()
    start(client)
    client.post("/api/seller/caption-requests/CR-0001/cancel", headers=HEADERS)
    view = client.get("/api/seller/caption-requests/CR-0001/approval").json()
    assert (view["state"], view["stop_reason"]) == ("done", "cancelled")
    refused = client.post("/api/seller/caption-requests/CR-0001/approval/answer", headers=HEADERS,
                          json={"answers": {"material": "a", "fit": "b"}})
    assert refused.status_code == 409 and refused.json()["code"] == "approval_closed"


def test_route_families_stay_apart(make_client) -> None:
    client = make_client()
    start(client)
    assert client.get("/api/seller/caption-proposals/CR-0001/approval").json()["code"] == "proposal_not_found"
    assert client.get("/api/seller/caption-requests/CP-0001/approval").json()["code"] == "request_not_found"
    bad = client.post("/api/seller/caption-requests", headers=HEADERS, json={"product_id": "FS180", "fields": ["price"]})
    assert bad.status_code == 422
