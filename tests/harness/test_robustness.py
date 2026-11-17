"""Session A2: retries, fallbacks, budgets, structured tool errors, adversarial calls. No network."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fake_models import APIError, call, fake  # noqa: E402  (tests/harness is on sys.path)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

from sellerops.evaluation.prompts import SELLER_SYSTEM_PROMPT, opening_message
from sellerops.evaluation.seller_tools import make_seller_tools
from sellerops.evaluation.terminal_tools import TERMINAL_TOOLS
from sellerops.harness.errors import is_transient_error
from sellerops.harness.graph import build_agent_graph, repair_tool_pairs
from sellerops.simulator import SellerEpisode

SALES_30 = "seller-dev-sales-summary-omzet-30-hari"
FAST_RETRY = RetryPolicy(initial_interval=0.0, jitter=False, max_attempts=2, retry_on=is_transient_error)


def final(**data: Any) -> AIMessage:
    return call("final", {"message": "Selesai.", "data": data})


def graph_for(model, episode=None, **options):
    episode = episode or SellerEpisode()
    options.setdefault("retry_policy", FAST_RETRY)
    return build_agent_graph(model, make_seller_tools(episode), system_prompt=SELLER_SYSTEM_PROMPT,
                             terminal_tools=TERMINAL_TOOLS, **options), episode


def run(model, task, **options):
    graph, episode = graph_for(model, **options)
    return graph.invoke({"messages": [opening_message(task)]}), episode


def errors(state) -> list[dict[str, Any]]:
    return [json.loads(m.content)["error"] for m in state["messages"]
            if isinstance(m, ToolMessage) and m.status == "error"]


# --- error classification -------------------------------------------------------------------

@pytest.mark.parametrize(("error", "transient"), [
    (APIError(429), True), (APIError(529), True), (APIError(503), True), (APIError(408), True),
    (APIError(400), False), (APIError(401), False), (APIError(404), False),
    (TimeoutError(), True), (ConnectionError(), True), (ValueError("bug"), False),
    (type("APIConnectionError", (Exception,), {})(), True), (type("APITimeoutError", (Exception,), {})(), True),
])
def test_is_transient_error(error: Exception, transient: bool) -> None:
    assert is_transient_error(error) is transient


# --- retry and fallback ---------------------------------------------------------------------

def test_transient_model_error_is_retried(dev_tasks) -> None:
    model = fake(APIError(529), final(revenue_idr=1))
    state, _ = run(model, dev_tasks[SALES_30])
    assert state["stop_reason"] == "final" and len(model.calls) == 2


def test_permanent_model_error_fails_fast(dev_tasks) -> None:
    model = fake(APIError(400), final())
    with pytest.raises(APIError):
        run(model, dev_tasks[SALES_30])
    assert len(model.calls) == 1


def test_retry_gives_up_after_max_attempts(dev_tasks) -> None:
    model = fake(APIError(529), APIError(529), APIError(529))
    with pytest.raises(APIError):
        run(model, dev_tasks[SALES_30])
    assert len(model.calls) == 2


def test_fallback_model_answers_when_primary_fails(dev_tasks) -> None:
    primary, backup = fake(APIError(529)), fake(final(revenue_idr=1))
    state, _ = run(primary, dev_tasks[SALES_30], fallback_models=[backup], retry_policy=None,
                   prompt_caching=True)
    assert state["stop_reason"] == "final"
    assert len(primary.calls) == 1 and len(backup.calls) == 1
    assert backup.kwargs[0].get("cache_control") == {"type": "ephemeral"}  # same kwargs as the primary


def test_prompt_caching_is_off_for_non_anthropic_models(dev_tasks) -> None:
    model = fake(final())
    run(model, dev_tasks[SALES_30])
    assert "cache_control" not in model.kwargs[0]
    system = model.calls[0][0]
    assert isinstance(system, SystemMessage) and isinstance(system.content, str)


# --- budgets ----------------------------------------------------------------------------------

def test_token_budget_ends_the_turn_after_tool_results(dev_tasks) -> None:
    model = fake(*[call("get_sales_summary", {"period": "last_7_days"}, tokens=600) for _ in range(5)])
    state, episode = run(model, dev_tasks[SALES_30], max_turn_tokens=1000)
    assert state["stop_reason"] == "token_budget"
    assert len(model.calls) == 2 and len(episode.snapshot()["history"]) == 2
    assert isinstance(state["messages"][-1], ToolMessage)  # no unanswered tool_use left behind


def test_time_budget_ends_the_turn(dev_tasks) -> None:
    model = fake(*[call("get_sales_summary", {"period": "last_7_days"}) for _ in range(5)])
    state, _ = run(model, dev_tasks[SALES_30], max_turn_seconds=0)
    assert state["stop_reason"] == "time_budget" and len(model.calls) == 1


def test_budgets_reset_every_turn(dev_tasks) -> None:
    model = fake(call("get_sales_summary", {"period": "last_7_days"}, tokens=600),
                 call("ask_user", {"question": "Periode?"}),
                 call("get_sales_summary", {"period": "last_30_days"}, tokens=600), final(revenue_idr=1))
    graph, _ = graph_for(model, max_turn_tokens=1000, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    first = graph.invoke({"messages": [opening_message(dev_tasks[SALES_30])]}, config)
    assert first["stop_reason"] == "ask_user"
    ask_id = first["messages"][-1].tool_calls[0]["id"]
    second = graph.invoke({"messages": [ToolMessage("30 hari", tool_call_id=ask_id, name="ask_user")]}, config)
    assert second["stop_reason"] == "final"


# --- structured tool errors and adversarial calls ----------------------------------------------

def test_tool_errors_have_codes(dev_tasks) -> None:
    mixed = AIMessage("", tool_calls=[{"name": "get_product", "args": {"product_id": "FS180"}, "id": "x1"},
                                      {"name": "final", "args": {"message": "m", "data": {}}, "id": "x2"}])
    model = fake(call("delete_store", {}), call("get_sales_summary", {"period": "yesterday"}), mixed,
                 call("final", {"message": "", "data": {}}))
    state, _ = run(model, dev_tasks[SALES_30])
    codes = [e["code"] for e in errors(state)]
    assert codes == ["unknown_tool", "invalid_arguments", "terminal_not_alone", "invalid_arguments"]


def test_business_errors_reach_the_model_and_it_can_recover(dev_tasks) -> None:
    too_long = "x" * 700
    model = fake(call("get_product", {"product_id": "FS999"}),
                 call("propose_caption_update", {"product_id": "FS180", "reason": "r", "description": too_long}),
                 call("propose_caption_update", {"product_id": "FS180", "reason": "r", "description": "Dress pea."}),
                 call("final", {"message": "Usulan dibuat.", "data": {"product_id": "FS180"}}))
    episode = SellerEpisode()
    before = episode.store.product("FS180")["caption_version"]
    state, _ = run(model, dev_tasks[SALES_30], episode=episode)
    observations = [json.loads(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]
    assert observations[0]["error"]["code"] == "product_not_found"
    assert observations[1]["ok"] is False  # caption over the 600-character limit
    assert observations[2]["ok"] is True and observations[2]["result"]["status"] == "pending"
    assert state["stop_reason"] == "final"
    assert [p["status"] for p in episode.snapshot()["proposals"]] == ["pending"]
    assert episode.store.product("FS180")["caption_version"] == before  # never applied by the agent


# --- continuing a thread after final ------------------------------------------------------------

def test_repair_tool_pairs_answers_dangling_calls() -> None:
    ai_final = AIMessage("", tool_calls=[{"name": "final", "args": {}, "id": "f"}])
    ai_two = AIMessage("", tool_calls=[{"name": "a", "args": {}, "id": "a1"}, {"name": "b", "args": {}, "id": "b1"}])
    messages = [HumanMessage("q"), ai_final, HumanMessage("lagi"), ai_two,
                ToolMessage("ok", tool_call_id="a1", name="a")]
    repaired = repair_tool_pairs(messages)
    kinds = [(type(m).__name__, getattr(m, "tool_call_id", None)) for m in repaired]
    assert kinds == [("HumanMessage", None), ("AIMessage", None), ("ToolMessage", "f"), ("HumanMessage", None),
                     ("AIMessage", None), ("ToolMessage", "a1"), ("ToolMessage", "b1")]
    assert repair_tool_pairs(repaired) == repaired  # nothing left to repair


def test_new_user_message_after_final_gets_a_valid_prompt(dev_tasks) -> None:
    model = fake(final(revenue_idr=1), final(revenue_idr=2))
    graph, _ = graph_for(model, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [opening_message(dev_tasks[SALES_30])]}, config)
    state = graph.invoke({"messages": [HumanMessage("Kalau 7 hari?")]}, config)
    assert state["stop_reason"] == "final"
    prompt = model.calls[1]
    final_call = prompt[2]
    assert isinstance(final_call, AIMessage) and isinstance(prompt[3], ToolMessage)
    assert prompt[3].tool_call_id == final_call.tool_calls[0]["id"]
    # The state itself is unchanged: no synthetic message was stored.
    assert sum(isinstance(m, ToolMessage) for m in state["messages"]) == 0
