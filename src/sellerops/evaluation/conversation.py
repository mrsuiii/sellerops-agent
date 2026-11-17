"""Turn loop for multi-turn eval: agent → user simulator → agent …

One conversation = one ``thread_id`` on a graph compiled with a checkpointer,
so every ``graph.invoke`` continues the same state (messages accumulate via
``add_messages``). Each invoke is one *agent turn* and must end with a
``stop_reason``:

- ``"ask_user"``: the turn ended with a question. The runner gets the seller's
  reply from the user simulator and sends it back as the **result of the
  pending ``ask_user`` tool call** (``ToolMessage`` with that call's id). The
  Messages API requires every ``tool_use`` to be followed by its
  ``tool_result``, so a plain ``HumanMessage`` would be rejected there.
- ``"final"``: the conversation is over; the verifier grades it.
- anything else (``"max_steps"``, ``"no_tool_call"``, ``"invalid_output"``, …):
  the conversation ends unfinished.

Per-turn fields: the graph state keeps ``final_action`` / ``stop_reason``
from the previous turn unless the next input overwrites them, so every turn's
input carries ``turn_reset`` (default: both set to ``None``). If the graph
keeps more per-turn fields (e.g. a step counter), pass them in ``turn_reset``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .user_sim import UserSimulator

DEFAULT_MAX_TURNS = 5
DEFAULT_TURN_RESET: Mapping[str, Any] = {"final_action": None, "stop_reason": None}


def pending_ask_user_id(messages: list[Any]) -> str:
    """Id of the ``ask_user`` call in the last AIMessage (the one the reply answers)."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "ask_user":
                    return call["id"]
            break
    raise ValueError("stop_reason is 'ask_user' but the last AIMessage has no ask_user call")


def run_conversation(graph: Any, task: dict[str, Any], user: UserSimulator, *,
                     max_turns: int = DEFAULT_MAX_TURNS, thread_id: str | None = None,
                     turn_reset: Mapping[str, Any] = DEFAULT_TURN_RESET,
                     recursion_limit: int = 50) -> dict[str, Any]:
    """Run one task to the end; returns a JSON-serializable result for the verifier."""

    thread_id = thread_id or f"{task['task_id']}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    transcript: list[dict[str, str]] = [{"role": "user", "text": task["opening_message"]}]
    turns: list[dict[str, Any]] = []
    next_input: Any = HumanMessage(task["opening_message"])
    state: dict[str, Any] = {}
    ended_by = "max_turns"

    for _ in range(max_turns):
        started = perf_counter()
        state = graph.invoke({"messages": [next_input], **turn_reset}, config)
        stop_reason, action = state.get("stop_reason"), state.get("final_action")
        turns.append({"stop_reason": stop_reason, "final_action": action,
                      "latency_s": round(perf_counter() - started, 3)})
        if stop_reason != "ask_user":
            ended_by = "final" if stop_reason == "final" else "agent_stopped"
            break
        question = action["question"] if isinstance(action, dict) else ""
        transcript.append({"role": "agent", "text": question})
        reply = user.reply(task, question, transcript[:-1])
        transcript.append({"role": "user", "text": reply.text})
        turns[-1]["user_reply"] = reply.text
        if reply.stop:
            ended_by = "user_stopped"
            break
        next_input = ToolMessage(content=reply.text, tool_call_id=pending_ask_user_id(state["messages"]),
                                 name="ask_user")

    final = state.get("final_action") if state.get("stop_reason") == "final" else None
    if final is not None:
        transcript.append({"role": "agent", "text": final.get("message", "")})
    return {
        "task_id": task["task_id"],
        "thread_id": thread_id,
        "ended_by": ended_by,
        "stop_reason": state.get("stop_reason"),
        "final_action": final,
        "turns": turns,
        "transcript": transcript,
    }
