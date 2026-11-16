"""Agent graph: model ⇄ tools, written with StateGraph. Session A1 — TODO (Nigka).

Target shape:

    START → model ──(ordinary tool calls)──▶ tools ──▶ model …
              └──(final / ask_user / step limit / no tool call)──▶ END

Pieces already available:

- ``sellerops.harness.models``: ``get_chat_model()``, ``CACHE_CONTROL``,
  ``cached_system_message()``
- ``sellerops.harness.prompts``: ``SYSTEM_PROMPT``, ``task_message()``

Everything else arrives as an argument. Never import from
``sellerops.evaluation`` or ``sellerops.services`` here:

- ``tools``: ordinary tools to execute (eval:
  ``sellerops.evaluation.sim_tools.make_sim_tools``; production:
  ``sellerops.harness.tools``, Track C / B);
- ``terminal_tools``: Pydantic classes the model calls to finish (eval:
  ``sellerops.evaluation.terminal_tools.TERMINAL_TOOLS``; production chat:
  none). Generic protocol: tool name = ``schema.model_config["title"]``;
  on a call, ``schema.model_validate(args).to_action()`` gives the final
  action (or raises when the arguments are invalid). Never execute them.

Think about how the graph should treat a reply without tool calls when no
terminal tools are given (a chat agent in production: a normal answer) versus
when they are (an eval task: the task was not finished).

Framework notes:

- ``StateGraph(AgentState)`` + ``add_node`` / ``add_edge`` /
  ``add_conditional_edges`` + ``compile(checkpointer=...)``.
- A node is a function ``(state) -> partial update``. A routing function used
  in ``add_conditional_edges`` returns the name of the next node (or ``END``)
  and must not change state.
- ``model.bind_tools(tools)`` returns a new runnable; bind once in
  ``build_agent_graph``, not on every call.
- Tool results go back as ``ToolMessage(content=..., tool_call_id=call["id"])``,
  one per tool call in the AIMessage, all in the same update.
- ``recursion_limit`` (passed in the invoke config) counts graph super-steps,
  not model calls; it is a last-resort guard, not your step limit.
- Reference only (do not import): ``langgraph.prebuilt.ToolNode`` and
  ``langchain.agents.create_agent`` solve the same problem.

Interface the tests expect:

    graph = build_agent_graph(model, tools, max_steps=8)
    state = graph.invoke({"messages": [task_message(task)]})   # eval: terminal_tools=TERMINAL_TOOLS
    state["final_action"], state["stop_reason"], state["messages"]
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..schemas import MAX_AGENT_STEPS


def build_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    terminal_tools: Sequence[type[BaseModel]] = (),
    max_steps: int = MAX_AGENT_STEPS,
    system_prompt: str | None = None,
    checkpointer=None,
):
    """Compile the agent graph for one set of tools (eval or production)."""
    # TODO(A1-2): bind tools + terminal_tools to the model (with cache_control).
    # TODO(A1-3): model node — prepend the system prompt, call the model,
    #   return the AIMessage (and update the step counter if a model call counts).
    # TODO(A1-4): tools node — for each tool call: a terminal tool →
    #   schema.model_validate(args).to_action() (no ToolMessage needed if the
    #   graph ends); otherwise run the matching tool
    #   and append a ToolMessage. What happens on an unknown tool name or an
    #   exception from the tool?
    # TODO(A1-5): routing (should_continue) — END with which stop_reason when:
    #   a terminal tool was called, the model replied without tool calls,
    #   the step limit was reached, or a terminal call had invalid arguments?
    raise NotImplementedError("Session A1: build the agent graph")
