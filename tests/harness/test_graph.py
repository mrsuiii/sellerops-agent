"""Agent graph behavior (Session A1 definition of done). No network.

The model is ``ScriptedChatModel`` (no API call). The tests fix the outward
behavior only; how the state, nodes and edges are designed is up to the
implementation.

World: the seller simulator (``SellerEpisode`` + seller tools) and the eval
terminal tools ``final`` / ``ask_user``.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from sellerops.evaluation.conversation import run_conversation
from sellerops.evaluation.prompts import SELLER_SYSTEM_PROMPT, opening_message
from sellerops.evaluation.seller_tools import SELLER_TOOL_NAMES, make_seller_tools
from sellerops.evaluation.seller_verifier import verify_conversation
from sellerops.evaluation.terminal_tools import TERMINAL_TOOLS
from sellerops.evaluation.user_sim import ScriptedUser
from sellerops.harness.graph import build_agent_graph
from sellerops.simulator import SellerEpisode

SALES_30 = "seller-dev-sales-summary-omzet-30-hari"  # direct question, no clarification
SALES_UNSPECIFIED = "seller-dev-sales-summary-toko-tanpa-periode"  # period is hidden info
_ids = itertools.count(1)


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """An AIMessage that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def sales_final(task: dict[str, Any]) -> AIMessage:
    e = task["expected"]
    return tool_call("final", {"message": f"Omzet Rp{e['revenue_idr']:,} dari {e['orders']} order.",
                               "data": {"period": e["period"], "revenue_idr": e["revenue_idr"],
                                        "orders": e["orders"]}})


def build(model, episode: SellerEpisode, *, max_steps: int = 8, terminal_tools=TERMINAL_TOOLS, checkpointer=None,
          **options):
    # prompt_caching=True: these tests check the Anthropic request shape with a scripted model
    # (auto-detection would turn caching off for a non-ChatAnthropic model).
    options.setdefault("prompt_caching", True)
    return build_agent_graph(model, make_seller_tools(episode), system_prompt=SELLER_SYSTEM_PROMPT,
                             terminal_tools=terminal_tools, max_steps=max_steps, checkpointer=checkpointer,
                             **options)


def run(model, task: dict[str, Any], *, max_steps: int = 8, recursion_limit: int = 50,
        terminal_tools=TERMINAL_TOOLS):
    episode = SellerEpisode()
    graph = build(model, episode, max_steps=max_steps, terminal_tools=terminal_tools)
    state = graph.invoke({"messages": [opening_message(task)]}, {"recursion_limit": recursion_limit})
    return state, episode


def test_happy_path_passes_verifier(scripted_model, dev_tasks) -> None:
    task = dev_tasks[SALES_30]
    model = scripted_model(tool_call("get_sales_summary", {"period": "last_30_days"}), sales_final(task))
    state, episode = run(model, task)

    assert state["stop_reason"] == "final"
    assert state["final_action"]["type"] == "final"
    result = {"final_action": state["final_action"], "stop_reason": state["stop_reason"], "turns": [state]}
    assert verify_conversation(task, episode.snapshot(), result)["decision"] == "pass"
    assert len(model.calls) == 2
    # The tool result went back to the model, paired with its call id.
    tool_messages = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == model.responses[0].tool_calls[0]["id"]
    assert json.loads(tool_messages[0].content)["ok"] is True


def test_tools_are_bound_and_prompt_is_cached(scripted_model, dev_tasks) -> None:
    model = scripted_model(tool_call("final", {"message": "Belum ada data.", "data": {}}))
    run(model, dev_tasks[SALES_30])
    names = {getattr(t, "name", None) or getattr(t, "__name__", None) for t in model.bound_tools}
    assert SELLER_TOOL_NAMES <= names
    assert {"FinalAnswer", "AskUser"} <= names or {"final", "ask_user"} <= names
    assert model.call_kwargs[0].get("cache_control") == {"type": "ephemeral"}


def test_ask_user_ends_the_turn(scripted_model, dev_tasks) -> None:
    model = scripted_model(tool_call("ask_user", {"question": "Periode berapa hari?"}))
    state, _ = run(model, dev_tasks[SALES_UNSPECIFIED])
    assert state["stop_reason"] == "ask_user"
    assert state["final_action"] == {"type": "ask_user", "question": "Periode berapa hari?"}
    assert len(model.calls) == 1


def test_plain_text_reply_stops_without_final(scripted_model, dev_tasks) -> None:
    model = scripted_model(AIMessage("Penjualan bagus kok."))
    state, _ = run(model, dev_tasks[SALES_30])
    assert state.get("final_action") is None
    assert state["stop_reason"] != "final"


def test_step_limit_stops_a_looping_model(scripted_model, dev_tasks) -> None:
    model = scripted_model(*[tool_call("get_sales_summary", {"period": "last_7_days"}) for _ in range(20)])
    state, episode = run(model, dev_tasks[SALES_30], max_steps=3)
    assert state["stop_reason"] == "max_steps"
    assert state.get("final_action") is None
    assert len(episode.snapshot()["history"]) <= 3
    assert len(model.calls) <= 4


def test_invalid_final_never_becomes_a_final_action(scripted_model, dev_tasks) -> None:
    model = scripted_model(tool_call("final", {"message": "", "data": "x"}))
    state, _ = run(model, dev_tasks[SALES_30], max_steps=3)
    # Either the graph stops, or it reports the error to the model and continues
    # (the script then runs out and the model replies in plain text).
    assert state.get("final_action") is None
    assert state["stop_reason"] != "final"


def test_tool_exception_does_not_crash_the_graph(scripted_model, dev_tasks) -> None:
    # Arguments that fail validation inside the tool (unknown period).
    model = scripted_model(tool_call("get_sales_summary", {"period": "yesterday"}))
    state, episode = run(model, dev_tasks[SALES_30])
    assert state["stop_reason"] != "final"
    assert episode.snapshot()["history"] == []


def test_without_terminal_tools_a_text_reply_is_a_normal_end(scripted_model, dev_tasks) -> None:
    # Production chat mode: no terminal tools, the model answers in text.
    reply = AIMessage("Omzet 30 hari terakhir naik.")
    model = scripted_model(tool_call("get_sales_summary", {"period": "last_30_days"}), reply)
    state, _ = run(model, dev_tasks[SALES_30], terminal_tools=())
    assert state["messages"][-1].content == reply.content
    assert state.get("final_action") is None
    names = {getattr(t, "name", None) or getattr(t, "__name__", None) for t in model.bound_tools}
    assert not names & {"final", "ask_user", "FinalAnswer", "AskUser"}


def test_multi_turn_conversation_with_scripted_user(scripted_model, dev_tasks) -> None:
    # Turn 1: the agent asks for the period. Turn 2 (same thread): it gets the
    # answer as the ask_user tool result, looks the numbers up and finishes.
    from langgraph.checkpoint.memory import InMemorySaver

    task = dev_tasks[SALES_UNSPECIFIED]
    model = scripted_model(
        tool_call("ask_user", {"question": "Periode berapa hari?"}),
        tool_call("get_sales_summary", {"period": "last_7_days"}),
        sales_final(task),
    )
    episode = SellerEpisode()
    graph = build(model, episode, checkpointer=InMemorySaver())
    result = run_conversation(graph, task, ScriptedUser())

    assert result["ended_by"] == "final"
    assert [t["stop_reason"] for t in result["turns"]] == ["ask_user", "final"]
    report = verify_conversation(task, episode.snapshot(), result, store=episode.store)
    assert report["decision"] == "pass", report
    # The second model call saw the whole thread, ending with the seller's reply.
    second_call = model.calls[1]
    assert isinstance(second_call[-1], ToolMessage)
    assert second_call[-1].content == "Periode yang dimaksud: 7 hari terakhir."


def test_invalid_final_is_reported_and_the_model_can_recover(scripted_model, dev_tasks) -> None:
    task = dev_tasks[SALES_30]
    model = scripted_model(
        tool_call("final", {"message": "", "data": "x"}),
        tool_call("get_sales_summary", {"period": "last_30_days"}),
        sales_final(task),
    )
    state, episode = run(model, task)
    assert state["stop_reason"] == "final"
    error = next(m for m in model.calls[1] if isinstance(m, ToolMessage))
    assert error.status == "error" and "invalid arguments for final" in error.content
    result = {"final_action": state["final_action"], "stop_reason": "final", "turns": [state]}
    assert verify_conversation(task, episode.snapshot(), result)["decision"] == "pass"


def test_unknown_tool_and_mixed_terminal_call_get_error_results(scripted_model, dev_tasks) -> None:
    mixed = AIMessage(content="", tool_calls=[
        {"name": "get_sales_summary", "args": {"period": "last_7_days"}, "id": "m1"},
        {"name": "final", "args": {"message": "ok", "data": {}}, "id": "m2"},
    ])
    model = scripted_model(tool_call("delete_store", {}), mixed)
    state, episode = run(model, dev_tasks[SALES_30])
    unknown = [m for m in model.calls[1] if isinstance(m, ToolMessage)][-1]
    assert unknown.status == "error" and "unknown tool" in unknown.content
    results = {m.tool_call_id: m for m in state["messages"] if isinstance(m, ToolMessage)}
    assert results["m1"].status == "success" and results["m2"].status == "error"
    assert state.get("final_action") is None  # final mixed with another call is not applied
    assert len(episode.snapshot()["history"]) == 1


def test_step_budget_resets_every_turn(scripted_model, dev_tasks) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    task = dev_tasks[SALES_UNSPECIFIED]
    model = scripted_model(
        tool_call("get_sales_summary", {"period": "last_90_days"}),
        tool_call("ask_user", {"question": "Periode berapa hari?"}),
        tool_call("get_sales_summary", {"period": "last_7_days"}),
        sales_final(task),
    )
    episode = SellerEpisode()
    graph = build(model, episode, max_steps=2, checkpointer=InMemorySaver())
    result = run_conversation(graph, task, ScriptedUser())
    # Two tool calls in total, one per turn: without the per-turn reset the second
    # turn would hit the 2-step budget before calling final.
    assert [t["stop_reason"] for t in result["turns"]] == ["ask_user", "final"]
