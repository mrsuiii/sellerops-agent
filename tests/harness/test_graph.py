"""Agent graph behavior (Session A1 definition of done). No network.

These tests are skipped while ``build_agent_graph`` still raises
NotImplementedError. They fix the outward behavior only; how the state,
nodes and edges are designed is up to the implementation.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from sellerops.evaluation.sim_tools import make_sim_tools
from sellerops.evaluation.terminal_tools import TERMINAL_TOOLS
from sellerops.evaluation.verifier import verify_outcome
from sellerops.harness.graph import build_agent_graph
from sellerops.harness.prompts import task_message
from sellerops.simulator.episode_service import EpisodeService

SEARCH_TASK = "dev-search-dev-brand-id-000"  # Comet backpack, expects P1008
COMET_BACKPACK = {"name": "Comet Everyday Backpack", "price_idr": 499000, "product_id": "P1008", "stock": 5}


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """An AIMessage that calls one tool."""
    tool_call.counter += 1
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{tool_call.counter}"}])


tool_call.counter = 0


def run(model, task: dict[str, Any], *, max_steps: int = 8, recursion_limit: int = 50,
        terminal_tools=TERMINAL_TOOLS):
    service = EpisodeService()
    episode = service.create_episode(task["task_id"])
    try:
        graph = build_agent_graph(model, make_sim_tools(service, episode.session_id),
                                  terminal_tools=terminal_tools, max_steps=max_steps)
    except NotImplementedError:
        pytest.skip("Session A1: build_agent_graph not implemented yet")
    state = graph.invoke({"messages": [task_message(task)]}, {"recursion_limit": recursion_limit})
    return state, service.snapshot(episode.session_id)


def test_happy_path_passes_verifier(scripted_model, dev_tasks) -> None:
    task = dev_tasks[SEARCH_TASK]
    model = scripted_model(
        tool_call("search_products", {"query": "backpack", "filters": {"brand": "Comet", "in_stock": True}}),
        tool_call("final", {"status": "success", "message": "Produk yang sesuai: Comet Everyday Backpack.",
                            "data": {"products": [COMET_BACKPACK]}}),
    )
    state, snapshot = run(model, task)

    assert state["stop_reason"] == "final"
    assert state["final_action"]["type"] == "final"
    assert verify_outcome(task, snapshot, state["final_action"])["decision"] == "pass"
    assert len(model.calls) == 2
    # The tool result went back to the model, paired with its call id.
    tool_messages = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == model.responses[0].tool_calls[0]["id"]


def test_tools_are_bound_and_prompt_is_cached(scripted_model, dev_tasks) -> None:
    model = scripted_model(tool_call("final", {"status": "not_found", "message": "Tidak ada.",
                                               "data": {"products": []}}))
    run(model, dev_tasks[SEARCH_TASK])
    names = {getattr(t, "name", None) or getattr(t, "__name__", None) for t in model.bound_tools}
    assert {"search_products", "get_inventory", "get_discount_policy", "calculate_offer"} <= names
    assert {"FinalAnswer", "AskUser"} <= names or {"final", "ask_user"} <= names
    assert model.call_kwargs[0].get("cache_control") == {"type": "ephemeral"}


def test_ask_user_ends_the_episode(scripted_model, dev_tasks) -> None:
    model = scripted_model(tool_call("ask_user", {"question": "Merek apa?", "missing_fields": ["brand"]}))
    state, _ = run(model, dev_tasks[SEARCH_TASK])
    assert state["stop_reason"] == "ask_user"
    assert state["final_action"]["type"] == "ask_user"
    assert len(model.calls) == 1


def test_plain_text_reply_stops_without_final(scripted_model, dev_tasks) -> None:
    model = scripted_model(AIMessage("Produknya ada, Comet Everyday Backpack."))
    state, _ = run(model, dev_tasks[SEARCH_TASK])
    assert state.get("final_action") is None
    assert state["stop_reason"] != "final"


def test_step_limit_stops_a_looping_model(scripted_model, dev_tasks) -> None:
    search = {"query": "backpack", "filters": {}}
    model = scripted_model(*[tool_call("search_products", search) for _ in range(20)])
    state, snapshot = run(model, dev_tasks[SEARCH_TASK], max_steps=3)
    assert state["stop_reason"] == "max_steps"
    assert state.get("final_action") is None
    assert len(snapshot["history"]) <= 3
    assert len(model.calls) <= 4


def test_invalid_final_never_becomes_a_final_action(scripted_model, dev_tasks) -> None:
    model = scripted_model(tool_call("final", {"status": "done", "message": "", "data": "x"}))
    state, _ = run(model, dev_tasks[SEARCH_TASK], max_steps=3)
    # Either the graph stops, or it reports the error to the model and continues
    # (the script then runs out and the model replies in plain text).
    assert state.get("final_action") is None
    assert state["stop_reason"] != "final"


def test_tool_exception_does_not_crash_the_graph(scripted_model, dev_tasks) -> None:
    # Arguments that fail validation inside the tool (empty query).
    model = scripted_model(tool_call("search_products", {"query": "", "filters": {}}))
    state, _ = run(model, dev_tasks[SEARCH_TASK])
    assert state["stop_reason"] != "final"


def test_without_terminal_tools_a_text_reply_is_a_normal_end(scripted_model, dev_tasks) -> None:
    # Production chat mode: no terminal tools, the model answers in text.
    reply = AIMessage("Ada satu produk: Comet Everyday Backpack.")
    model = scripted_model(
        tool_call("search_products", {"query": "backpack", "filters": {"brand": "Comet"}}),
        reply,
    )
    state, _ = run(model, dev_tasks[SEARCH_TASK], terminal_tools=())
    assert state["messages"][-1].content == reply.content
    assert state.get("final_action") is None
    names = {getattr(t, "name", None) or getattr(t, "__name__", None) for t in model.bound_tools}
    assert not names & {"final", "ask_user", "FinalAnswer", "AskUser"}
