"""State of the agent graph. Session A1 — TODO (Nigka).

LangGraph keeps one state dict per run. Each node returns a *partial update*;
for every key, the key's reducer decides how the update is merged:

- no reducer (plain annotation): the new value replaces the old one;
- ``Annotated[list[AnyMessage], add_messages]``: new messages are appended
  (a message with an existing ``id`` replaces that message instead).

The graph tests (``tests/harness/test_graph.py``) read these keys from the
final state: ``messages``, ``final_action``, ``stop_reason``. Add whatever else
your nodes need (e.g. a step counter).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # TODO(A1-1): counter for the step limit. What exactly counts as one step:
    #   a model call, or a tool execution? (EpisodeService counts tool calls.)
    # TODO(A1-1): final_action: dict[str, Any] | None  — the tagged final / ask_user action
    # TODO(A1-1): stop_reason: str — why the graph ended
    #   (e.g. "final", "ask_user", "max_steps", "no_tool_call", "invalid_output")

