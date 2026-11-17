"""Eval runner plumbing with a fake agent model (no network)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sellerops.evaluation.seller_eval import run_eval, run_trial, summarize, summary_markdown
from sellerops.evaluation.user_sim import ScriptedUser

OMZET_30 = "seller-dev-sales-summary-omzet-30-hari"
CRASH_MARKER = "Gimana penjualan"


class SalesAgent(BaseChatModel):
    """Stateless: asks for the 30-day summary, then reports it with `final`."""

    @property
    def _llm_type(self) -> str:
        return "sales-agent"

    def bind_tools(self, tools: Any, **kwargs: Any) -> SalesAgent:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        if any(CRASH_MARKER in str(m.content) for m in messages):
            raise RuntimeError("overloaded")
        last = messages[-1]
        if isinstance(last, ToolMessage) and last.name == "get_sales_summary":
            result = json.loads(last.content)["result"]
            args = {"message": f"Omzet Rp{result['revenue_idr']:,}.",
                    "data": {"period": "last_30_days", "revenue_idr": result["revenue_idr"],
                             "orders": result["orders"]}}
            message = AIMessage("", tool_calls=[{"name": "final", "args": args, "id": "f1"}])
        else:
            message = AIMessage("", tool_calls=[{"name": "get_sales_summary", "args": {"period": "last_30_days"},
                                                 "id": "c1"}],
                                usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                                                "input_token_details": {"cache_read": 80}},
                                # UsageMetadataCallbackHandler keys usage by model name (ChatAnthropic sets it).
                                response_metadata={"model_name": "fake-agent"})
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_trial_pass_records_verdict_usage_and_history(dev_tasks) -> None:
    record = run_trial(dev_tasks[OMZET_30], 0, agent_model=SalesAgent(), make_user=ScriptedUser, run_id="r1")
    assert record["score"] == 1 and record["failed"] == [] and record["error"] is None
    assert record["history"][0]["tool"] == "get_sales_summary"
    assert record["usage"]["input_tokens"] == 100 and record["usage"]["cache_read"] == 80
    json.dumps(record, default=str)  # JSONL-serializable


def test_run_eval_summary_counts_crashes_as_unscored(dev_tasks) -> None:
    tasks = [dev_tasks[OMZET_30], dev_tasks["seller-dev-sales-summary-toko-tanpa-periode"],  # crashes
             dev_tasks["seller-dev-top-category-kategori-omzet-terbesar"]]  # wrong family answer
    records = run_eval(tasks, k=2, concurrency=3, agent_model=SalesAgent(), make_user=ScriptedUser, run_id="r2")
    assert [(r["task_id"], r["trial"]) for r in records] == sorted((t["task_id"], i) for t in tasks for i in range(2))
    by_task = {r["task_id"]: r for r in records}
    assert by_task[OMZET_30]["score"] == 1
    crashed = [r for r in records if r["score"] is None]
    assert crashed and all("overloaded" in r["error"] for r in crashed)

    summary = summarize(records, k=2)
    assert summary["errors"] == len(crashed)
    assert summary["overall"]["pass@1"] is None  # a crash invalidates the aggregate, never hides
    assert summary["by_family"]["sales_summary"]["tasks"] >= 1
    assert summary["by_family"]["top_category"]["pass^2"] == 0
    config = {"run_id": "r2", "agent_model": "fake", "agent_effort": "medium", "user": "scripted",
              "split": "dev", "data_version": "v", "judge_model": None}
    assert "| **all** |" in summary_markdown(summary, config)


def test_summary_without_crashes(dev_tasks) -> None:
    records = run_eval([dev_tasks[OMZET_30]], k=3, agent_model=SalesAgent(), make_user=ScriptedUser)
    summary = summarize(records, k=3)
    assert summary["overall"] == {"pass@1": 1.0, "pass@3": 1.0, "pass^1": 1.0, "pass^3": 1.0}
    assert summary["usage_per_trial"]["input_tokens"] == 100.0


class ClarifyingAgent(BaseChatModel):
    """Asks for the period first, then answers with the period the seller gave."""

    @property
    def _llm_type(self) -> str:
        return "clarifying-agent"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ClarifyingAgent:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage) and last.name == "ask_user":
            period = "last_7_days" if "7 hari" in last.content else "last_30_days"
            calls = [{"name": "get_sales_summary", "args": {"period": period}, "id": "s1"}]
        elif isinstance(last, ToolMessage) and last.name == "get_sales_summary":
            result = json.loads(last.content)["result"]
            calls = [{"name": "final", "id": "f1", "args": {
                "message": "Ringkasan.", "data": {"period": result["period"]["name"],
                                                  "revenue_idr": result["revenue_idr"], "orders": result["orders"]}}}]
        else:
            calls = [{"name": "ask_user", "args": {"question": "Periode berapa hari?"}, "id": "a1"}]
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=calls))])


def test_multi_turn_trial_carries_the_reply_to_the_next_turn(dev_tasks) -> None:
    # Regression: a "run_id" metadata key made every turn after the first a no-op.
    task = dev_tasks["seller-dev-sales-summary-toko-tanpa-periode"]
    record = run_trial(task, 0, agent_model=ClarifyingAgent(), make_user=ScriptedUser, run_id="r3")
    assert [t["stop_reason"] for t in record["turns"]] == ["ask_user", "final"]
    assert record["score"] == 1, record["checks"]
