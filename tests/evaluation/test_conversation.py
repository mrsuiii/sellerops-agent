"""Turn loop with a fake graph (the real graph is Session A1). No network."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sellerops.evaluation.conversation import pending_ask_user_id, run_conversation
from sellerops.evaluation.seller_verifier import verify_conversation
from sellerops.evaluation.user_sim import ScriptedUser, UserReply
from sellerops.simulator import SellerEpisode


class FakeGraph:
    """Plays back one state per invoke and records the inputs and configs."""

    def __init__(self, states: list[dict[str, Any]]) -> None:
        self.states = states
        self.inputs: list[dict[str, Any]] = []
        self.configs: list[dict[str, Any]] = []

    def invoke(self, graph_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(graph_input)
        self.configs.append(config)
        return self.states[len(self.inputs) - 1]


def ask(question: str, call_id: str) -> dict[str, Any]:
    message = AIMessage("", tool_calls=[{"name": "ask_user", "args": {"question": question}, "id": call_id}])
    return {"messages": [message], "stop_reason": "ask_user",
            "final_action": {"type": "ask_user", "question": question}}


def final(data: dict[str, Any]) -> dict[str, Any]:
    action = {"type": "final", "message": "Selesai.", "data": data}
    return {"messages": [AIMessage("", tool_calls=[{"name": "final", "args": action, "id": "f"}])],
            "stop_reason": "final", "final_action": action}


def test_ask_user_reply_goes_back_as_the_tool_result(dev_tasks, solve) -> None:
    task = dev_tasks["seller-dev-sales-summary-toko-tanpa-periode"]
    episode = SellerEpisode()
    answer = solve(task, episode)
    graph = FakeGraph([ask("Periode berapa hari?", "call_ask_1"), final(answer["data"])])

    result = run_conversation(graph, task, ScriptedUser(), thread_id="th-1")

    first, second = graph.inputs
    assert isinstance(first["messages"][0], HumanMessage)
    assert first["messages"][0].content == task["opening_message"]
    reply = second["messages"][0]
    assert isinstance(reply, ToolMessage) and reply.tool_call_id == "call_ask_1"
    assert reply.content == "Periode yang dimaksud: 7 hari terakhir."
    # Per-turn fields are reset on every turn; one thread for the conversation.
    assert all(i["final_action"] is None and i["stop_reason"] is None for i in graph.inputs)
    assert {c["configurable"]["thread_id"] for c in graph.configs} == {"th-1"}

    assert result["ended_by"] == "final" and len(result["turns"]) == 2
    assert [m["role"] for m in result["transcript"]] == ["user", "agent", "user", "agent"]
    report = verify_conversation(task, episode.snapshot(), result)
    assert report["decision"] == "pass" and report["metrics"]["asked_clarification"]


def test_conversation_stops_at_max_turns(dev_tasks) -> None:
    task = dev_tasks["seller-dev-sales-summary-toko-tanpa-periode"]
    graph = FakeGraph([ask(f"Q{i}?", f"c{i}") for i in range(10)])
    result = run_conversation(graph, task, ScriptedUser(), max_turns=3)
    assert result["ended_by"] == "max_turns" and len(graph.inputs) == 3
    assert result["final_action"] is None


def test_agent_stop_without_final(dev_tasks) -> None:
    task = dev_tasks["seller-dev-sales-summary-omzet-30-hari"]
    graph = FakeGraph([{"messages": [AIMessage("teks biasa")], "stop_reason": "no_tool_call",
                        "final_action": None}])
    result = run_conversation(graph, task, ScriptedUser())
    assert result["ended_by"] == "agent_stopped" and result["final_action"] is None


def test_user_can_end_the_conversation(dev_tasks) -> None:
    class Quitter:
        def reply(self, task, question, transcript):
            return UserReply("", stop=True)

    task = dev_tasks["seller-dev-sales-summary-toko-tanpa-periode"]
    result = run_conversation(FakeGraph([ask("Q?", "c1")]), task, Quitter())
    assert result["ended_by"] == "user_stopped"


def test_pending_ask_user_id_requires_the_call() -> None:
    assert pending_ask_user_id([ask("Q?", "c9")["messages"][0]]) == "c9"
    with pytest.raises(ValueError):
        pending_ask_user_id([AIMessage("hi")])
