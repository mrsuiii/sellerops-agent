"""Readable offline examples: real graph/service, scripted model completions."""

import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from sellerops.agent import nodes
from sellerops.agent.graph import build_graph
from sellerops.services.episode_service.episode_service import EpisodeService


CONFIG = {"configurable": {
    "router_model_name": "qwen-base", "action_model_name": "sellerops-sft",
    "max_model_calls": 8,
}}
ACT = {"decision": "act", "text": ""}
STOCK = {"type": "tool_call", "tool": "get_inventory", "arguments": {"product_id": "P1001"}}


def setup_run(monkeypatch, outputs):
    """Replace only inference; validation, routing, and tools remain real."""
    script = iter(outputs)
    calls = []

    def generate(messages, config):
        calls.append((messages, config["configurable"]["model_name"]))
        return json.dumps(next(script))

    monkeypatch.setattr(nodes, "_generate_text", generate)
    service = EpisodeService()
    episode = service.create_episode("learning-example")
    state = {"messages": [HumanMessage(content="Cek stok P1001")],
             "session_id": episode.session_id, "model_calls": 0, "retry_count": 0}
    return build_graph(service), service, state, calls


def test_direct_chat_skips_sft_and_tools(monkeypatch):
    graph, service, state, calls = setup_run(monkeypatch, [{"decision": "reply", "text": "Halo!"}])
    state["messages"] = [HumanMessage(content="Halo")]
    result = graph.invoke(state, CONFIG)
    assert result["response"] == "Halo!"
    assert result["status"] == "completed"
    assert [alias for _, alias in calls] == ["qwen-base"]
    assert service.snapshot(state["session_id"])["history"] == []


def test_router_sft_tool_router_round_trip(monkeypatch):
    graph, service, state, calls = setup_run(monkeypatch, [
        ACT, STOCK, {"decision": "reply", "text": "Stok P1001 tersedia 7 unit."},
    ])
    result = graph.invoke(state, CONFIG)
    assert [alias for _, alias in calls] == ["qwen-base", "sellerops-sft", "qwen-base"]
    assert result["model_calls"] == 3
    assert len(result["messages"]) == 4  # user, SFT action, tool, router answer
    request, observation = result["messages"][1:3]
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == request.tool_calls[0]["id"]
    assert service.snapshot(state["session_id"])["step_count"] == 1
    # Router decisions are absent from SFT history; observations reach router.
    assert calls[1][0][-1] == {"role": "user", "content": "Cek stok P1001"}
    transcript = json.loads(calls[2][0][-1]["content"])
    assert json.loads(transcript[-1]["content"])["ok"] is True


@pytest.mark.parametrize("candidate", [
    {"type": "ask_user", "question": "Berapa unit?", "missing_fields": ["quantity"]},
    {"type": "final", "status": "not_found", "message": "Tidak ditemukan.", "data": {}},
])
def test_sft_non_tool_actions_return_to_router(monkeypatch, candidate):
    graph, service, state, calls = setup_run(monkeypatch, [
        ACT, candidate, {"decision": "ask", "text": "Bisa beri detail tambahan?"},
    ])
    result = graph.invoke(state, CONFIG)
    assert result["status"] == "waiting_user"
    assert len(calls) == 3
    assert service.snapshot(state["session_id"])["history"] == []


def test_budget_stops_loop_even_without_tool_calls(monkeypatch):
    candidate = {"type": "ask_user", "question": "Berapa unit?", "missing_fields": ["quantity"]}
    graph, _, state, calls = setup_run(monkeypatch, [ACT, candidate, ACT])
    result = graph.invoke(state, {"configurable": {**CONFIG["configurable"], "max_model_calls": 3}})
    assert len(calls) == result["model_calls"] == 3
    assert result["status"] == "failed"


@pytest.mark.parametrize("output,options", [
    ({"decision": "invented", "text": "x"}, {}),
    ({"decision": "reply", "text": ""}, {}),
    ({"decision": "act", "text": "private plan"}, {"strict_router_schema": True}),
    ({"decision": "act", "text": "", "extra": 1}, {}),
])
def test_invalid_router_output_never_executes_tools(monkeypatch, output, options):
    graph, service, state, calls = setup_run(monkeypatch, [output] * 3)
    result = graph.invoke(state, {"configurable": {**CONFIG["configurable"], **options}})
    assert result["status"] == "failed" and result["failure_reason"] == "retry_exhausted"
    assert result["model_calls"] == len(calls) == 3 and result["retry_count"] == 2
    assert service.snapshot(state["session_id"])["history"] == []


def test_clarification_continues_with_manual_history(monkeypatch):
    graph, service, state, _ = setup_run(monkeypatch, [
        {"decision": "ask", "text": "Produk apa?"},
        ACT, STOCK, {"decision": "reply", "text": "Stok tersedia 7 unit."},
    ])
    state["messages"] = [HumanMessage(content="Cek stok")]
    first = graph.invoke(state, CONFIG)
    second_input = {**first, "messages": first["messages"] + [HumanMessage(content="P1001")]}
    second = graph.invoke(second_input, CONFIG)
    assert first["status"] == "waiting_user"
    assert second["status"] == "completed"
    assert second["model_calls"] == 4  # same task budget across clarification turns
    assert len(second["messages"]) == 6
    assert service.snapshot(state["session_id"])["step_count"] == 1


def test_act_with_text_is_normalized_by_default(monkeypatch):
    """The 2026-10-06 dev failure: act plus an explanation. The text is dropped."""
    graph, service, state, calls = setup_run(monkeypatch, [
        {"decision": "act", "text": "Saya akan mengecek stok P1001."}, STOCK,
        {"decision": "act", "text": "lanjut"},
        {"type": "final", "status": "success", "message": "Stok P1001 tersedia 7 unit.",
         "data": {"product_id": "P1001", "available_stock": 7}},
        {"decision": "reply", "text": "Stok P1001 tersedia 7 unit."},
    ])
    result = graph.invoke(state, CONFIG)
    assert result["status"] == "completed"
    assert result["retry_count"] == 0
    assert [alias for _, alias in calls] == [
        "qwen-base", "sellerops-sft", "qwen-base", "sellerops-sft", "qwen-base"]
    # Router text never reaches SFT history.
    assert all("Saya akan" not in m["content"] for m in calls[1][0])
    assert service.snapshot(state["session_id"])["step_count"] == 1


RULE = {"configurable": {**CONFIG["configurable"], "router_mode": "rule"}}


def test_rule_router_runs_sft_and_tools_without_router_model(monkeypatch):
    final = {"type": "final", "status": "success", "message": "Stok P1001 tersedia 7 unit.",
             "data": {"product_id": "P1001", "available_stock": 7}}
    graph, service, state, calls = setup_run(monkeypatch, [STOCK, final])
    result = graph.invoke(state, RULE)
    assert [alias for _, alias in calls] == ["sellerops-sft", "sellerops-sft"]
    assert result["model_calls"] == 2
    assert result["status"] == "completed"
    assert result["response"] == final["message"]
    assert result["final_action"] == final
    assert result["messages"][-1].content == final["message"]
    assert service.snapshot(state["session_id"])["step_count"] == 1


def test_rule_router_turns_ask_user_into_question_and_resumes(monkeypatch):
    ask = {"type": "ask_user", "question": "Produk apa yang dimaksud?", "missing_fields": ["product_id"]}
    final = {"type": "final", "status": "success", "message": "Stok tersedia 7 unit.",
             "data": {"product_id": "P1001", "available_stock": 7}}
    graph, service, state, calls = setup_run(monkeypatch, [ask, STOCK, final])
    state["messages"] = [HumanMessage(content="Cek stok")]
    first = graph.invoke(state, RULE)
    assert first["status"] == "waiting_user" and first["response"] == ask["question"]
    second = graph.invoke({**first, "messages": first["messages"] + [HumanMessage(content="P1001")]}, RULE)
    assert second["status"] == "completed"
    assert second["model_calls"] == 3
    assert len(calls) == 3 and all(alias == "sellerops-sft" for _, alias in calls)


def test_rule_router_still_respects_model_budget(monkeypatch):
    graph, service, state, calls = setup_run(monkeypatch, [STOCK] * 10)
    result = graph.invoke(state, {"configurable": {**RULE["configurable"], "max_model_calls": 3}})
    assert result["status"] == "failed" and result["failure_reason"] == "budget_exhausted"
    assert len(calls) == 3


def test_unknown_router_mode_is_rejected(monkeypatch):
    graph, _, state, _ = setup_run(monkeypatch, [])
    with pytest.raises(ValueError, match="router_mode"):
        graph.invoke(state, {"configurable": {**CONFIG["configurable"], "router_mode": "magic"}})
