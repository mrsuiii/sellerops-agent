"""Seller chat over HTTP: replies, tool use, caption proposals, history, budget, errors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from sellerops.seller import api as seller_api
from sellerops.store.api import app
from sellerops.workflows.approval import api as approval_api
from sellerops.workflows.approval.budget import LLMBudget
from sellerops.workflows.chat import api as chat_api
from sellerops.workflows.chat.runner import ChatRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from fake_models import call, fake  # noqa: E402

HEADERS = {"X-Seller-User": "seller-1"}
THREAD = "a" * 32
URL = f"/api/seller/chat/threads/{THREAD}"


@pytest.fixture
def make_client(service):
    def build(*script, budget: LLMBudget | None = None):
        runner = ChatRunner(service, fake(*script), InMemorySaver())
        app.dependency_overrides[seller_api.get_service] = lambda: service
        app.dependency_overrides[chat_api.get_chat_runner] = lambda: runner
        if budget is not None:
            app.dependency_overrides[approval_api.default_budget] = lambda: budget
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def send(client: TestClient, text: str, thread: str = THREAD):
    return client.post(f"/api/seller/chat/threads/{thread}/messages", json={"text": text}, headers=HEADERS)


def test_plain_reply_and_tool_use(make_client) -> None:
    client = make_client(call("get_product", {"product_id": "FS180"}), AIMessage("Harganya Rp1.000.000."))
    body = send(client, "berapa harga FS180?").json()
    assert body["reply"] == "Harganya Rp1.000.000."
    assert body["tools"] == [{"name": "get_product", "ok": True}]
    assert body["proposals"] == []


def test_caption_proposal_is_created_for_the_seller(make_client, service) -> None:
    client = make_client(
        call("propose_caption_update", {"product_id": "FS180", "reason": "lebih menarik",
                                        "description": "Atasan ringan untuk harian."}),
        AIMessage("Usulan sudah dibuat dan menunggu persetujuanmu."),
    )
    body = send(client, "perbaiki deskripsi FS180").json()
    assert len(body["proposals"]) == 1
    proposal = service.get_proposal(body["proposals"][0])
    assert proposal["status"] == "pending" and proposal["created_by"] == "agent:chat"
    assert proposal["changes"]["description"]["new"] == "Atasan ringan untuk harian."


def test_history_keeps_the_conversation_and_hides_tool_steps(make_client) -> None:
    client = make_client(call("get_product", {"product_id": "FS180"}), AIMessage("Sudah dicek."),
                         AIMessage("Jawaban kedua."))
    send(client, "pertanyaan pertama")
    send(client, "pertanyaan kedua")
    messages = client.get(URL).json()["messages"]
    assert messages == [
        {"role": "user", "text": "pertanyaan pertama"},
        {"role": "agent", "text": "Sudah dicek."},
        {"role": "user", "text": "pertanyaan kedua"},
        {"role": "agent", "text": "Jawaban kedua."},
    ]


def test_threads_are_separate(make_client) -> None:
    client = make_client(AIMessage("satu"), AIMessage("dua"))
    send(client, "halo", thread="b" * 32)
    assert client.get(URL).json()["messages"] == []


def test_demo_budget_refuses_before_the_model_runs(make_client) -> None:
    budget = LLMBudget(per_client_hour=1, per_day=10, enabled=True)
    client = make_client(AIMessage("satu"), AIMessage("dua"), budget=budget)
    assert send(client, "pertama").status_code == 200
    refused = send(client, "kedua")
    assert refused.status_code == 429 and refused.json()["code"] == "llm_budget_exhausted"


def test_model_failure_is_503_and_the_thread_survives(make_client) -> None:
    client = make_client(ValueError("boom"), AIMessage("pulih"))
    failed = send(client, "pertama")
    assert failed.status_code == 503 and failed.json()["code"] == "agent_unavailable"
    assert send(client, "kedua").json()["reply"] == "pulih"


def test_bad_thread_id_is_rejected(make_client) -> None:
    client = make_client(AIMessage("x"))
    assert client.get("/api/seller/chat/threads/bukan-id").status_code == 422
    assert send(client, "halo", thread="A" * 32).status_code == 422


def test_empty_message_is_rejected(make_client) -> None:
    client = make_client(AIMessage("x"))
    assert client.post(f"/api/seller/chat/threads/{THREAD}/messages", json={"text": ""}, headers=HEADERS).status_code == 422
