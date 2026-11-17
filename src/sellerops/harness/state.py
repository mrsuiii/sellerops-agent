"""State and runtime context of the agent graph (Sessions A1, A2, A4).

LangGraph keeps one state dict per thread. Each node returns a *partial
update*; for every key, the key's reducer decides how the update is merged:

- no reducer (plain annotation): the new value replaces the old one;
- ``Annotated[list[AnyMessage], add_messages]``: new messages are appended
  (a message with an existing ``id`` replaces that message, and
  ``RemoveMessage(id=...)`` deletes one — the summarize node uses this).

With a checkpointer, the state survives between ``graph.invoke`` calls on the
same ``thread_id``. ``messages``, ``summary`` and ``context_tokens`` describe
the whole conversation; ``steps``, ``turn_tokens``, ``turn_started_at``,
``final_action`` and ``stop_reason`` are *per-turn* fields that the graph's
``start_turn`` node resets at the start of every invoke.

Runtime context is different from state: it is passed per invoke
(``graph.invoke(input, config, context=AgentContext(...))``), is read-only,
is never checkpointed and never shown to the model. It carries who is asking
(user, store) so tools can scope their data without the model choosing ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

StopReason = Literal[
    "final",          # a terminal tool `final` was called with valid arguments
    "ask_user",       # a terminal tool `ask_user` was called: the turn waits for the user
    "answer",         # chat mode (no terminal tools): the model answered in text
    "no_tool_call",   # eval mode: the model replied in text instead of finishing
    "max_steps",      # the per-turn step budget ran out
    "token_budget",   # the per-turn token budget ran out (A2)
    "time_budget",    # the per-turn wall-clock budget ran out (A2)
]


@dataclass(frozen=True)
class AgentContext:
    """Per-invoke runtime context (read-only, not checkpointed, not seen by the model)."""

    user_id: str | None = None
    store_id: str | None = None
    locale: str = "id-ID"
    extra: dict[str, Any] = field(default_factory=dict)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Tool executions in the current turn (ordinary tools, including failed
    # calls and rejected terminal calls). A model call alone is not a step:
    # it either ends the turn or produces tool calls that are counted.
    steps: int
    # Tokens (input + output, all model calls) and start time of the current turn.
    turn_tokens: int
    turn_started_at: float
    # The tagged action from a terminal tool, e.g. {"type": "final", ...}.
    final_action: dict[str, Any] | None
    # Why the turn ended; None while the graph is still running.
    stop_reason: StopReason | None
    # A4 context budget: summary of the messages removed so far, and the prompt
    # size (input tokens) of the latest model call, which triggers summarizing.
    summary: str | None
    context_tokens: int
