"""Agent graph: model ⇄ tools, written with StateGraph (Sessions A1, A2, A4).

Shape:

    START → start_turn ─┬────────────────▶ model ──(tool calls)──▶ tools ──▶ model …
                        └─▶ summarize ─────▶ │                        │
                          (context too big)  └──(stop_reason set)──▶ END ◀──(stop_reason set)

- ``start_turn`` resets the per-turn fields (``steps``, token and time
  budget, ``final_action``, ``stop_reason``). Every ``graph.invoke`` is one
  turn; with a checkpointer the messages of earlier turns stay in the thread.
- ``summarize`` (A4, only with ``max_context_tokens``) runs at the start of a
  turn when the previous prompt was larger than the budget: older messages are
  summarized by a model and removed with ``RemoveMessage``; the summary is
  sent to the model after the system prompt.
- ``model`` calls the chat model with the system prompt (+ summary) in front
  of the thread. A reply without tool calls ends the turn: ``"answer"`` in
  chat mode (no terminal tools), ``"no_tool_call"`` in eval mode.
- ``tools`` answers every tool call of the last AIMessage:
  - a terminal tool (``final`` / ``ask_user``) called *alone* with valid
    arguments ends the turn with its action and no ToolMessage (the caller
    answers a pending ``ask_user`` with a ToolMessage on the next turn);
  - invalid terminal arguments, a terminal tool mixed with other calls, an
    unknown tool, invalid arguments or a tool exception become
    ``ToolMessage(status="error")`` with ``{"ok": false, "error": {"code",
    "message"}}`` (the same shape as business errors) so the model can
    correct itself; each counts as a step;
  - after the calls, the per-turn budgets are checked: ``max_steps``,
    ``max_turn_tokens``, ``max_turn_seconds``. Budgets are only enforced here,
    after every call of the AIMessage has a result, so the thread never ends a
    turn with an unanswered ``tool_use`` (the Messages API would reject it).
- The routing functions only read ``stop_reason`` (and ``context_tokens``
  after ``start_turn``); nodes decide why to stop.

Robustness (A2) — see ``harness/errors.py`` for the full error map:

- ``RetryPolicy`` on the ``model`` and ``summarize`` nodes retries transient
  API errors (``is_transient_error``) with backoff. LangGraph re-runs the whole
  node, which is safe because these nodes have no side effects. The ``tools``
  node has none: a tool may write (caption proposals), so it is never retried
  blindly; tool failures go back to the model instead.
- ``fallback_models``: ``bound.with_fallbacks([...])`` — when the primary
  call raises (after the SDK's own retries), the same prompt goes to the next
  model. Fallbacks must accept the same tools and, with prompt caching on,
  ``cache_control`` (i.e. other Claude models).
- Before each model call, AIMessage tool calls without a result (e.g. a
  ``final`` from an earlier turn followed by a new user message) get a
  synthetic result in the *prompt* (not in state), so a continued thread
  stays valid for the API.

Runtime context: the graph is compiled with ``context_schema=AgentContext``.
A tool that declares a ``runtime: ToolRuntime`` parameter (hidden from the
model's schema by LangChain) gets one built from the node's ``Runtime``
(context, store, stream writer) plus the current state, config and call id —
what ``ToolNode`` does, written out here.

Dependencies arrive as arguments; this module never imports
``sellerops.evaluation`` or ``sellerops.simulator``.

Framework notes:

- ``model.bind_tools(...)`` is done once here; it returns a new runnable whose
  request carries the tool schemas. ``cache_control`` is passed per call so
  ChatAnthropic puts a cache breakpoint on the last block (tools → system →
  earlier turns are reused); ``cached_system_message`` adds a breakpoint on
  the system prompt itself. ``prompt_caching=None`` turns this on only for
  ChatAnthropic models.
- ``max_steps`` is our own budget. ``recursion_limit`` (invoke config) counts
  graph super-steps (each node run is one) and is only a last-resort guard:
  one turn uses about ``2 * max_steps + 3`` super-steps.
- Node functions may declare ``runtime: Runtime[...]`` and ``config``;
  LangGraph injects them by parameter name.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from pydantic import BaseModel, ValidationError

from ..schemas import MAX_AGENT_STEPS, canonical_json
from .errors import is_transient_error
from .models import CACHE_CONTROL, cached_system_message, supports_prompt_caching
from .state import AgentContext, AgentState

# Node-level retry on top of the SDK's own retries (see harness/errors.py).
DEFAULT_MODEL_RETRY = RetryPolicy(initial_interval=2.0, backoff_factor=2.0, max_interval=30.0,
                                  max_attempts=2, jitter=True, retry_on=is_transient_error)

DEFAULT_SUMMARY_PROMPT = """\
You compress the earlier part of a conversation between a user and an \
assistant that uses tools, so the assistant can continue without it. Write a \
concise summary in the user's language with: what the user asked and \
decided, every answer the user gave to the assistant's questions, the key \
tool results with their exact numbers and ids, what the assistant already \
told the user, and anything still open. Do not add facts. Plain text only."""

_FINAL_DELIVERED = "Delivered to the user."
_NOT_EXECUTED = "Not executed: the turn ended before this call ran."


def tool_error(call: dict[str, Any], code: str, message: str) -> ToolMessage:
    """Error result for one tool call, in the same shape as business errors."""
    content = canonical_json({"ok": False, "error": {"code": code, "message": message}})
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"], status="error")


def _short(error: Exception, limit: int = 1500) -> str:
    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= limit else text[:limit] + " …"


def _injected_args(tool: BaseTool) -> set[str]:
    """Tool parameters hidden from the model (e.g. ``runtime: ToolRuntime``)."""
    try:
        return set(tool.get_input_schema().model_fields) - set(tool.tool_call_schema.model_fields)
    except Exception:  # tools without Pydantic schemas: nothing to inject
        return set()


def repair_tool_pairs(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Give every AIMessage tool call a result, as the Messages API requires.

    Unanswered calls (a ``final`` from an earlier turn, a call cut off by a
    budget) get a synthetic ToolMessage right after the existing results of
    their AIMessage. Only the prompt is repaired; the state is left as is.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    out: list[AnyMessage] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        for call in pending:
            text = _FINAL_DELIVERED if call["name"] == "final" else _NOT_EXECUTED
            out.append(ToolMessage(content=text, tool_call_id=call["id"], name=call["name"]))
        pending.clear()

    for message in messages:
        if not isinstance(message, ToolMessage):
            flush()
        out.append(message)
        if isinstance(message, AIMessage):
            pending.extend(c for c in message.tool_calls if c["id"] not in answered)
    flush()
    return out


def summary_cut_index(messages: Sequence[AnyMessage], keep_messages: int) -> int:
    """Index where the kept tail starts (0 = nothing to summarize).

    The cut is placed before a HumanMessage or an AIMessage, never before a
    ToolMessage, so every kept tool result still follows its tool call. At
    least ``keep_messages`` messages are kept.
    """
    target = len(messages) - keep_messages
    cuts = [i for i, m in enumerate(messages) if 0 < i <= target and isinstance(m, HumanMessage | AIMessage)]
    return max(cuts, default=0)


def _render(messages: Sequence[BaseMessage], limit: int = 2000) -> str:
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"USER: {message.text}")
        elif isinstance(message, AIMessage):
            if message.text:
                lines.append(f"ASSISTANT: {message.text}")
            for call in message.tool_calls:
                lines.append(f"ASSISTANT CALLS {call['name']}({canonical_json(call['args'])})")
        elif isinstance(message, ToolMessage):
            text = message.text
            lines.append(f"RESULT {message.name}: {text[:limit]}{' …' if len(text) > limit else ''}")
    return "\n".join(lines)


def build_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    terminal_tools: Sequence[type[BaseModel]] = (),
    max_steps: int = MAX_AGENT_STEPS,
    checkpointer: Any = None,
    store: Any = None,
    fallback_models: Sequence[BaseChatModel] = (),
    retry_policy: RetryPolicy | None = DEFAULT_MODEL_RETRY,
    prompt_caching: bool | None = None,
    max_turn_tokens: int | None = None,
    max_turn_seconds: float | None = None,
    max_context_tokens: int | None = None,
    keep_messages: int = 6,
    summary_model: BaseChatModel | None = None,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
):
    """Compile the agent graph for one set of tools (eval or production)."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if keep_messages < 2:
        raise ValueError("keep_messages must be at least 2")
    tools_by_name = {tool.name: tool for tool in tools}
    terminals = {schema.model_config["title"]: schema for schema in terminal_tools}
    overlap = tools_by_name.keys() & terminals.keys()
    if overlap:
        raise ValueError(f"tool names used twice: {sorted(overlap)}")
    needs_runtime = {name for name, tool in tools_by_name.items() if "runtime" in _injected_args(tool)}

    caching = supports_prompt_caching(model) if prompt_caching is None else prompt_caching
    all_tools = [*tools, *terminal_tools]

    def bind(chat_model: BaseChatModel) -> Any:
        return chat_model.bind_tools(all_tools) if all_tools else chat_model

    bound = bind(model)
    if fallback_models:
        # with_fallbacks: on an exception from the primary, the same input (and kwargs such
        # as cache_control) goes to each fallback in order; the first success wins.
        bound = bound.with_fallbacks([bind(m) for m in fallback_models])
    invoke_kwargs = {"cache_control": CACHE_CONTROL} if caching else {}
    system_message = cached_system_message(system_prompt) if caching else SystemMessage(system_prompt)
    text_reply_reason = "no_tool_call" if terminals else "answer"
    summarizer = summary_model or model

    def budget_stop(steps: int, turn_tokens: int, started: float | None) -> str | None:
        if steps >= max_steps:
            return "max_steps"
        if max_turn_tokens is not None and turn_tokens >= max_turn_tokens:
            return "token_budget"
        if max_turn_seconds is not None and started is not None and time.time() - started >= max_turn_seconds:
            return "time_budget"
        return None

    def start_turn(state: AgentState) -> dict[str, Any]:
        return {"steps": 0, "turn_tokens": 0, "turn_started_at": time.time(),
                "final_action": None, "stop_reason": None}

    def summarize(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        cut = summary_cut_index(messages, keep_messages)
        if cut == 0:
            return {"context_tokens": 0}
        previous = state.get("summary")
        source = (f"Earlier summary:\n{previous}\n\nConversation to add:\n" if previous else "") + _render(messages[:cut])
        response = summarizer.invoke([SystemMessage(summary_prompt), HumanMessage(source)])
        return {"summary": response.text.strip(), "context_tokens": 0,
                "messages": [RemoveMessage(id=m.id) for m in messages[:cut]]}

    def call_model(state: AgentState) -> dict[str, Any]:
        prompt: list[AnyMessage] = [system_message]
        if state.get("summary"):
            prompt.append(HumanMessage(f"<conversation_summary>\n{state['summary']}\n</conversation_summary>"))
        prompt.extend(repair_tool_pairs(state["messages"]))
        response = bound.invoke(prompt, **invoke_kwargs)
        usage = getattr(response, "usage_metadata", None) or {}
        update: dict[str, Any] = {
            "messages": [response],
            "turn_tokens": (state.get("turn_tokens") or 0) + usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "context_tokens": usage.get("input_tokens", 0),
        }
        if not getattr(response, "tool_calls", None):
            update["stop_reason"] = text_reply_reason
        return update

    def run_tools(state: AgentState, config: RunnableConfig, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        calls = last.tool_calls
        steps = state.get("steps") or 0
        turn_tokens = state.get("turn_tokens") or 0
        started = state.get("turn_started_at")

        # A terminal tool called alone finishes the turn (when its arguments are valid).
        if len(calls) == 1 and calls[0]["name"] in terminals:
            call = calls[0]
            try:
                action = terminals[call["name"]].model_validate(call["args"]).to_action()
            except Exception as error:  # ValidationError / SchemaError: let the model retry
                steps += 1
                update = {"messages": [tool_error(call, "invalid_arguments",
                                                  f"invalid arguments for {call['name']}: {_short(error)}")],
                          "steps": steps}
                reason = budget_stop(steps, turn_tokens, started)
                if reason:
                    update["stop_reason"] = reason
                return update
            return {"final_action": action, "stop_reason": call["name"]}

        messages: list[ToolMessage] = []
        for call in calls:
            steps += 1
            name = call["name"]
            if name in terminals:
                messages.append(tool_error(call, "terminal_not_alone",
                                           f"{name} must be called alone, after the other tool results "
                                           "are back. It was not applied."))
                continue
            tool = tools_by_name.get(name)
            if tool is None:
                available = sorted([*tools_by_name, *terminals])
                messages.append(tool_error(call, "unknown_tool", f"unknown tool {name!r}; available: {available}"))
                continue
            args = dict(call["args"])
            if name in needs_runtime:
                args["runtime"] = ToolRuntime(state=state, context=runtime.context, config=config,
                                              stream_writer=runtime.stream_writer, tool_call_id=call["id"],
                                              store=runtime.store)
            try:
                output = tool.invoke(args)
            except ValidationError as error:
                messages.append(tool_error(call, "invalid_arguments", _short(error)))
                continue
            except Exception as error:  # a failing tool: report, don't crash
                messages.append(tool_error(call, "tool_error", _short(error)))
                continue
            content = output if isinstance(output, str) else canonical_json(output)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))

        update: dict[str, Any] = {"messages": messages, "steps": steps}
        reason = budget_stop(steps, turn_tokens, started)
        if reason:
            update["stop_reason"] = reason
        return update

    def after_start(state: AgentState) -> str:
        if max_context_tokens is not None and (state.get("context_tokens") or 0) >= max_context_tokens:
            return "summarize"
        return "model"

    def after_model(state: AgentState) -> str:
        return END if state.get("stop_reason") else "tools"

    def after_tools(state: AgentState) -> str:
        return END if state.get("stop_reason") else "model"

    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("start_turn", start_turn)
    graph.add_node("model", call_model, retry_policy=retry_policy)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "start_turn")
    if max_context_tokens is not None:
        graph.add_node("summarize", summarize, retry_policy=retry_policy)
        graph.add_conditional_edges("start_turn", after_start, ["summarize", "model"])
        graph.add_edge("summarize", "model")
    else:
        graph.add_edge("start_turn", "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["model", END])
    return graph.compile(checkpointer=checkpointer, store=store)
