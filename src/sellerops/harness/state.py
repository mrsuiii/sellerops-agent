"""State of the agent graph (Session A1).

LangGraph keeps one state dict per thread. Each node returns a *partial
update*; for every key, the key's reducer decides how the update is merged:

- no reducer (plain annotation): the new value replaces the old one;
- ``Annotated[list[AnyMessage], add_messages]``: new messages are appended
  (a message with an existing ``id`` replaces that message instead).

With a checkpointer, the state survives between ``graph.invoke`` calls on the
same ``thread_id``. That is why ``steps``, ``final_action`` and
``stop_reason`` are *per-turn* fields: the graph's ``start_turn`` node resets
them at the start of every invoke, while ``messages`` keeps the whole
conversation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

StopReason = Literal[
    "final",          # a terminal tool `final` was called with valid arguments
    "ask_user",       # a terminal tool `ask_user` was called: the turn waits for the user
    "answer",         # chat mode (no terminal tools): the model answered in text
    "no_tool_call",   # eval mode: the model replied in text instead of finishing
    "max_steps",      # the per-turn step budget ran out
]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Tool executions in the current turn (ordinary tools, including failed
    # calls and rejected terminal calls). A model call alone is not a step:
    # it either ends the turn or produces tool calls that are counted.
    steps: int
    # The tagged action from a terminal tool, e.g. {"type": "final", ...}.
    final_action: dict[str, Any] | None
    # Why the turn ended; None while the graph is still running.
    stop_reason: StopReason | None
