"""Persistent threads, trajectory export and replay (Session A4).

- ``open_sqlite_checkpointer(path)``: a ``SqliteSaver`` on a file, so threads
  outlive the process (``InMemorySaver`` is for tests). Every super-step of
  the graph writes one checkpoint: the full state after that step, keyed by
  ``thread_id`` + ``checkpoint_id``.
- ``export_trajectory(graph, config)``: the thread as JSON — every message
  (with tool calls, results, usage) plus the checkpoint timeline from
  ``graph.get_state_history(config)`` (newest first in LangGraph; returned
  oldest first here), each with the node that wrote it.
- ``replay(graph, thread_id, checkpoint_id)``: time travel. Invoking with
  ``input=None`` and a ``checkpoint_id`` in the config makes LangGraph load
  that checkpoint and run the nodes that were *next* at that point. The new
  steps are written as a fork of the thread; the original history stays.
  Nodes run again for real: the model is called again (so the output can
  differ: same state, new sample) and tools run against whatever the caller
  wired in. Tool side effects that happened before the checkpoint are not
  restored by LangGraph; the caller must rebuild that world (the eval replay
  script re-runs the earlier tool calls on a fresh simulator episode).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver


def open_sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    """A file-backed checkpointer usable from several threads (eval runs trials in parallel)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    item: dict[str, Any] = {"role": message.type, "id": message.id, "text": message.text}
    if isinstance(message, AIMessage):
        if message.tool_calls:
            item["tool_calls"] = [{"id": c["id"], "name": c["name"], "args": c["args"]} for c in message.tool_calls]
        if message.usage_metadata:
            item["usage"] = dict(message.usage_metadata)
        if message.response_metadata.get("model_name"):
            item["model"] = message.response_metadata["model_name"]
    if isinstance(message, ToolMessage):
        item.update(tool_call_id=message.tool_call_id, name=message.name, status=message.status)
    return item


def export_trajectory(graph: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Messages and checkpoint timeline of one thread (JSON-serializable)."""
    state = graph.get_state(config)
    values = state.values or {}
    timeline = []
    for snapshot in reversed(list(graph.get_state_history(config))):
        metadata = snapshot.metadata or {}
        timeline.append({
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
            "step": metadata.get("step"),
            "source": metadata.get("source"),
            "wrote": sorted((metadata.get("writes") or {}).keys()) if isinstance(metadata.get("writes"), dict) else None,
            "next": list(snapshot.next),
            "messages": len((snapshot.values or {}).get("messages", [])),
            "stop_reason": (snapshot.values or {}).get("stop_reason"),
        })
    return {
        "thread_id": config["configurable"]["thread_id"],
        "messages": [message_to_dict(m) for m in values.get("messages", [])],
        "summary": values.get("summary"),
        "stop_reason": values.get("stop_reason"),
        "final_action": values.get("final_action"),
        "checkpoints": timeline,
    }


def checkpoint_before_last_model_call(graph: Any, config: dict[str, Any]) -> str | None:
    """Id of the latest checkpoint whose next node is ``model`` (replay the last decision)."""
    for snapshot in graph.get_state_history(config):  # newest first
        if "model" in snapshot.next:
            return snapshot.config["configurable"]["checkpoint_id"]
    return None


def replay(graph: Any, thread_id: str, checkpoint_id: str, **config_extra: Any) -> dict[str, Any]:
    """Re-run the thread from one checkpoint (a fork); returns the new final state."""
    config = {**config_extra, "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    return graph.invoke(None, config)
