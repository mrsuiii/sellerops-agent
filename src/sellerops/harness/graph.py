"""Agent graph: model ⇄ tools, written with StateGraph (Session A1).

Shape:

    START → start_turn → model ──(tool calls)──▶ tools ──▶ model …
                           │                       │
                           └──(stop_reason set)──▶ END ◀──(stop_reason set)

- ``start_turn`` resets the per-turn fields (``steps``, ``final_action``,
  ``stop_reason``). Every ``graph.invoke`` is one turn; with a checkpointer the
  messages of earlier turns stay in the thread.
- ``model`` calls the chat model with the system prompt in front of the thread.
  A reply without tool calls ends the turn: ``"answer"`` in chat mode (no
  terminal tools), ``"no_tool_call"`` in eval mode (the task was not finished).
- ``tools`` answers every tool call of the last AIMessage:
  - a terminal tool (``final`` / ``ask_user``) called *alone* with valid
    arguments ends the turn with its action and no ToolMessage (the caller
    answers a pending ``ask_user`` with a ToolMessage on the next turn);
  - invalid terminal arguments, a terminal tool mixed with other calls, an
    unknown tool or a tool exception become ``ToolMessage(status="error")``
    so the model can correct itself; each counts as a step;
  - when ``steps`` reaches ``max_steps`` the turn ends with ``"max_steps"``.
- The routing functions only read ``stop_reason``; nodes decide why to stop.

Dependencies arrive as arguments; this module never imports
``sellerops.evaluation`` or ``sellerops.simulator``:

- ``tools``: ordinary tools (eval: ``make_seller_tools``; production:
  ``sellerops.harness.tools``);
- ``system_prompt``: required (eval: ``SELLER_SYSTEM_PROMPT``);
- ``terminal_tools``: Pydantic classes; tool name =
  ``schema.model_config["title"]``, ``schema.model_validate(args).to_action()``
  gives the action. They are bound to the model but never executed.

Framework notes:

- ``model.bind_tools(...)`` is done once here; it returns a new runnable whose
  request carries the tool schemas. ``cache_control`` is passed per call so
  ChatAnthropic puts a cache breakpoint on the last block (tools → system →
  earlier turns are reused); ``cached_system_message`` adds a breakpoint on
  the system prompt itself.
- ``max_steps`` is our own budget. ``recursion_limit`` (invoke config) counts
  graph super-steps (each node run is one) and is only a last-resort guard:
  one turn uses about ``2 * max_steps + 2`` super-steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from ..schemas import MAX_AGENT_STEPS
from .models import CACHE_CONTROL, cached_system_message
from .state import AgentState


def _error_message(call: dict[str, Any], text: str) -> ToolMessage:
    return ToolMessage(content=f"Error: {text}", tool_call_id=call["id"], name=call["name"], status="error")


def _short(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= 1500 else text[:1500] + " …"


def build_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    terminal_tools: Sequence[type[BaseModel]] = (),
    max_steps: int = MAX_AGENT_STEPS,
    checkpointer=None,
):
    """Compile the agent graph for one set of tools (eval or production)."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    tools_by_name = {tool.name: tool for tool in tools}
    terminals = {schema.model_config["title"]: schema for schema in terminal_tools}
    overlap = tools_by_name.keys() & terminals.keys()
    if overlap:
        raise ValueError(f"tool names used twice: {sorted(overlap)}")

    bound = model.bind_tools([*tools, *terminal_tools]) if (tools or terminal_tools) else model
    system_message = cached_system_message(system_prompt)
    text_reply_reason = "no_tool_call" if terminals else "answer"

    def start_turn(state: AgentState) -> dict[str, Any]:
        return {"steps": 0, "final_action": None, "stop_reason": None}

    def call_model(state: AgentState) -> dict[str, Any]:
        response = bound.invoke([system_message, *state["messages"]], cache_control=CACHE_CONTROL)
        update: dict[str, Any] = {"messages": [response]}
        if not getattr(response, "tool_calls", None):
            update["stop_reason"] = text_reply_reason
        return update

    def run_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        calls = last.tool_calls
        steps = state.get("steps") or 0

        # A terminal tool called alone finishes the turn (when its arguments are valid).
        if len(calls) == 1 and calls[0]["name"] in terminals:
            call = calls[0]
            try:
                action = terminals[call["name"]].model_validate(call["args"]).to_action()
            except Exception as error:  # ValidationError / SchemaError: let the model retry
                steps += 1
                update = {"messages": [_error_message(call, f"invalid arguments for {call['name']}: "
                                                            f"{_short(error)}")], "steps": steps}
                if steps >= max_steps:
                    update["stop_reason"] = "max_steps"
                return update
            return {"final_action": action, "stop_reason": call["name"]}

        messages: list[ToolMessage] = []
        for call in calls:
            steps += 1
            name = call["name"]
            if name in terminals:
                messages.append(_error_message(call, f"{name} must be called alone, after the other "
                                                     "tool results are back. It was not applied."))
                continue
            tool = tools_by_name.get(name)
            if tool is None:
                available = sorted([*tools_by_name, *terminals])
                messages.append(_error_message(call, f"unknown tool {name!r}; available: {available}"))
                continue
            try:
                output = tool.invoke(call["args"])
            except Exception as error:  # bad arguments or a failing tool: report, don't crash
                messages.append(_error_message(call, _short(error)))
                continue
            content = output if isinstance(output, str) else str(output)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))

        update: dict[str, Any] = {"messages": messages, "steps": steps}
        if steps >= max_steps:
            update["stop_reason"] = "max_steps"
        return update

    def after_model(state: AgentState) -> str:
        return END if state.get("stop_reason") else "tools"

    def after_tools(state: AgentState) -> str:
        return END if state.get("stop_reason") else "model"

    graph = StateGraph(AgentState)
    graph.add_node("start_turn", start_turn)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["model", END])
    return graph.compile(checkpointer=checkpointer)
