"""Seller eval runner: tasks × k trials → verifier (+ judge) → pass@k / pass^k.

One trial = one conversation with fresh state:

    SellerEpisode() → make_seller_tools(episode) → build_agent_graph(..., InMemorySaver())
      → run_conversation(graph, task, user)            # agent ⇄ user simulator
      → verify_conversation(task, snapshot, result)    # pass / fail + checks
      → judge.assess(...)  (optional, only for verifier passes)

The graph is built per trial because the tools are closures over that trial's
episode (its history is what the verifier grades). The chat models are shared:
they hold no conversation state.

Scores: 1 = verifier pass, 0 = fail, None = the trial crashed (API error,
recursion limit). ``pass_metrics`` / ``pass_hat_metrics`` return None when any
trial is None, so a crash is never silently counted as a pass or a fail.

Tracing: every invoke carries ``run_name`` / ``tags`` / ``metadata`` (task,
trial, model, ``eval_run_id``; never a ``run_id`` metadata key, see
``run_trial``), so LangSmith runs can be filtered per task and trial.
Token usage is summed per trial with ``UsageMetadataCallbackHandler`` (agent
model only; the user simulator and the judge are counted apart).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, quantiles
from time import perf_counter
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from ..harness.graph import build_agent_graph
from ..harness.trajectory import message_to_dict
from ..schemas import MAX_AGENT_STEPS
from ..simulator import SellerEpisode
from .conversation import DEFAULT_MAX_TURNS, run_conversation
from .judge import JudgeError, MessageJudge
from .metrics import pass_hat_metrics, pass_metrics
from .prompts import SELLER_SYSTEM_PROMPT
from .seller_tools import make_seller_tools
from .seller_verifier import verify_conversation
from .terminal_tools import TERMINAL_TOOLS
from .user_sim import UserSimulator

UserFactory = Callable[[], UserSimulator]
_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read", "cache_creation")


def _sum_usage(handler: UsageMetadataCallbackHandler) -> dict[str, int]:
    total = dict.fromkeys(_USAGE_KEYS, 0)
    for usage in handler.usage_metadata.values():
        total["input_tokens"] += usage.get("input_tokens", 0)
        total["output_tokens"] += usage.get("output_tokens", 0)
        details = usage.get("input_token_details") or {}
        total["cache_read"] += details.get("cache_read", 0) or 0
        total["cache_creation"] += details.get("cache_creation", 0) or 0
    return total


def thread_id_for(run_id: str, task_id: str, trial: int) -> str:
    return f"{run_id}-{task_id}-t{trial}"


def run_trial(task: dict[str, Any], trial: int, *, agent_model: BaseChatModel, make_user: UserFactory,
              judge: MessageJudge | None = None, max_turns: int = DEFAULT_MAX_TURNS,
              max_steps: int = MAX_AGENT_STEPS, run_id: str = "", tags: list[str] | None = None,
              metadata: dict[str, Any] | None = None, checkpointer: Any = None,
              graph_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run and grade one conversation; never raises for agent/API failures.

    ``checkpointer``: shared saver (e.g. SQLite, for replay later); default a
    fresh ``InMemorySaver``. ``graph_options`` go to ``build_agent_graph``
    (fallback models, budgets, context budget).
    """
    episode = SellerEpisode()
    graph = build_agent_graph(agent_model, make_seller_tools(episode), system_prompt=SELLER_SYSTEM_PROMPT,
                              terminal_tools=TERMINAL_TOOLS, max_steps=max_steps,
                              checkpointer=checkpointer or InMemorySaver(), **(graph_options or {}))
    thread_id = thread_id_for(run_id, task["task_id"], trial)
    usage = UsageMetadataCallbackHandler()
    run_config = {
        "run_name": f"{task['task_id']}#t{trial}",
        "tags": ["seller-eval", *(tags or [])],
        # Not "run_id": LangGraph reads that metadata key, and with it every turn after the
        # first returned the previous state unchanged (found in the first live eval run).
        "metadata": {"task_id": task["task_id"], "family": task["family"], "trial": trial,
                     "eval_run_id": run_id, **(metadata or {})},
        "callbacks": [usage],
    }
    record: dict[str, Any] = {"task_id": task["task_id"], "family": task["family"], "trial": trial,
                              "thread_id": thread_id, "score": None, "error": None}

    def messages() -> list[dict[str, Any]]:
        try:
            state = graph.get_state({"configurable": {"thread_id": thread_id}})
            return [message_to_dict(m) for m in (state.values or {}).get("messages", [])]
        except Exception:  # nothing checkpointed yet
            return []

    started = perf_counter()
    try:
        result = run_conversation(graph, task, make_user(), max_turns=max_turns,
                                  thread_id=thread_id, run_config=run_config)
    except Exception as error:  # API failure, recursion limit, …: recorded, not fatal for the run
        record.update(error=f"{type(error).__name__}: {str(error)[:300]}",
                      latency_s=round(perf_counter() - started, 3), usage=_sum_usage(usage),
                      history=episode.snapshot()["history"], messages=messages())
        return record
    latency = round(perf_counter() - started, 3)
    snapshot = episode.snapshot()
    report = verify_conversation(task, snapshot, result, store=episode.store)
    record.update(
        score=1 if report["decision"] == "pass" else 0,
        decision=report["decision"], failed=report["failed"], checks=report["checks"],
        metrics=report["metrics"], ended_by=result["ended_by"], stop_reason=result["stop_reason"],
        latency_s=latency, usage=_sum_usage(usage), final_action=result["final_action"],
        transcript=result["transcript"], turns=result["turns"], history=snapshot["history"],
        proposals=snapshot["proposals"], messages=messages(),
    )
    if judge is not None and record["score"] == 1:
        try:
            record["judge"] = judge.assess(task, snapshot, result)
        except JudgeError as error:
            record["judge"] = {"status": "error", "pass": None, "reason": str(error)}
    return record


def run_eval(tasks: list[dict[str, Any]], *, k: int, concurrency: int = 4, progress: Callable | None = None,
             **trial_kwargs: Any) -> list[dict[str, Any]]:
    """All tasks × k trials, in parallel threads; records sorted by (task, trial)."""
    jobs = [(task, trial) for task in tasks for trial in range(k)]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_trial, task, trial, **trial_kwargs) for task, trial in jobs]
        for future in futures:
            records.append(future.result())
            if progress:
                progress(records[-1], len(records), len(jobs))
    return sorted(records, key=lambda r: (r["task_id"], r["trial"]))


def _pass_table(records: list[dict[str, Any]], k: int) -> dict[str, float | None]:
    task_ids = sorted({r["task_id"] for r in records})
    samples = [(r["task_id"], r["score"]) for r in records]
    out = pass_metrics(samples, repetitions=k, k=k, task_ids=task_ids)
    out.update(pass_hat_metrics(samples, repetitions=k, k=k, task_ids=task_ids))
    return out


def _pct(values: list[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return round(quantiles(values, n=100, method="inclusive")[q - 1], 3)


def summarize(records: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
    graded = [r for r in records if r["score"] is not None]
    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    usage = {key: sum(r.get("usage", {}).get(key, 0) for r in records) for key in _USAGE_KEYS}
    judged = [r["judge"] for r in records if r.get("judge", {}).get("status") == "scored"]
    clarify = [r for r in graded if r["metrics"]["needs_clarification"]]
    return {
        "trials": len(records),
        "tasks": len({r["task_id"] for r in records}),
        "k": k,
        "errors": sum(r["score"] is None for r in records),
        "overall": _pass_table(records, k),
        "by_family": {family: {"tasks": len({r["task_id"] for r in rows}), **_pass_table(rows, k)}
                      for family, rows in sorted(by_family.items())},
        "failed_checks": dict(Counter(check for r in graded for check in r["failed"]).most_common()),
        "stop_reasons": dict(Counter(r.get("stop_reason") for r in graded).most_common()),
        "latency_s": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95)},
        "avg_tool_calls": round(mean(r["metrics"]["tool_calls"] for r in graded), 2) if graded else None,
        "avg_agent_turns": round(mean(r["metrics"]["agent_turns"] for r in graded), 2) if graded else None,
        "asked_when_needed": (sum(r["metrics"]["asked_clarification"] for r in clarify), len(clarify)),
        "usage_total": usage,
        "usage_per_trial": {key: round(value / len(records), 1) for key, value in usage.items()} if records else {},
        "judge": {"scored": len(judged), "pass": sum(bool(j["pass"]) for j in judged),
                  "errors": sum(r.get("judge", {}).get("status") == "error" for r in records)},
    }


def _fmt(value: Any) -> str:
    return "–" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def summary_markdown(summary: dict[str, Any], config: dict[str, Any]) -> str:
    k = summary["k"]
    cols = sorted({"pass@1", f"pass@{k}", "pass^1", f"pass^{k}"}, key=lambda c: (c[4], c))
    lines = [f"# Seller eval {config['run_id']}", "",
             f"model `{config['agent_model']}` (effort {config['agent_effort']}), user `{config['user']}`, "
             f"split `{config['split']}`, k={k}, data `{config['data_version']}`", "",
             "| family | tasks | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    rows = [("**all**", {"tasks": summary["tasks"], **summary["overall"]}), *summary["by_family"].items()]
    for name, row in rows:
        lines.append(f"| {name} | {row['tasks']} | " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    asked, needed = summary["asked_when_needed"]
    usage = summary["usage_per_trial"]
    lines += ["",
              f"- trials {summary['trials']}, crashed {summary['errors']}",
              f"- failed checks: {summary['failed_checks'] or 'none'}",
              f"- stop reasons: {summary['stop_reasons']}",
              f"- latency per conversation p50 {_fmt(summary['latency_s']['p50'])}s, "
              f"p95 {_fmt(summary['latency_s']['p95'])}s",
              f"- avg tool calls {_fmt(summary['avg_tool_calls'])}, avg agent turns {_fmt(summary['avg_agent_turns'])}",
              f"- asked clarification when needed: {asked}/{needed}",
              f"- agent tokens per trial: input {usage.get('input_tokens')}, output {usage.get('output_tokens')}, "
              f"cache read {usage.get('cache_read')}, cache write {usage.get('cache_creation')}"]
    judge = summary["judge"]
    if config.get("judge_model"):
        lines.append(f"- judge `{config['judge_model']}` on verifier passes: {judge['pass']}/{judge['scored']} "
                     f"pass, {judge['errors']} judge errors")
    return "\n".join(lines) + "\n"
