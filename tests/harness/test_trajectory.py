"""Session A4: SQLite checkpoints, trajectory export, replay from a checkpoint. No network."""

from __future__ import annotations

import json

from fake_models import call, fake  # tests/harness is on sys.path
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from sellerops.harness.graph import build_agent_graph
from sellerops.harness.trajectory import (
    checkpoint_before_last_model_call,
    export_trajectory,
    open_sqlite_checkpointer,
    replay,
)


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


def test_thread_survives_a_new_checkpointer_on_the_same_file(tmp_path) -> None:
    db = tmp_path / "threads.sqlite"
    config = {"configurable": {"thread_id": "t1"}}
    graph = build_agent_graph(fake(AIMessage("halo")), [lookup], system_prompt="s",
                              checkpointer=open_sqlite_checkpointer(db))
    graph.invoke({"messages": [HumanMessage("hai")]}, config)

    reopened = build_agent_graph(fake(AIMessage("lagi")), [lookup], system_prompt="s",
                                 checkpointer=open_sqlite_checkpointer(db))
    state = reopened.invoke({"messages": [HumanMessage("dua")]}, config)
    assert [m.text for m in state["messages"]] == ["hai", "halo", "dua", "lagi"]


def test_export_and_replay_from_before_the_last_model_call(tmp_path) -> None:
    config = {"configurable": {"thread_id": "t2"}}
    first = fake(call("lookup", {"query": "omzet"}), AIMessage("Jawaban pertama."))
    graph = build_agent_graph(first, [lookup], system_prompt="s",
                              checkpointer=open_sqlite_checkpointer(tmp_path / "db.sqlite"))
    graph.invoke({"messages": [HumanMessage("Berapa omzet?")]}, config)

    trajectory = export_trajectory(graph, config)
    json.dumps(trajectory)
    assert [m["role"] for m in trajectory["messages"]] == ["human", "ai", "tool", "ai"]
    assert trajectory["messages"][1]["tool_calls"][0]["name"] == "lookup"
    assert trajectory["stop_reason"] == "answer"
    nodes = [c["next"] for c in trajectory["checkpoints"]]
    assert ["model"] in nodes and ["tools"] in nodes

    # Replay the last decision with a different model: same state up to the checkpoint, new answer.
    checkpoint = checkpoint_before_last_model_call(graph, config)
    second = fake(AIMessage("Jawaban kedua."))
    replayer = build_agent_graph(second, [lookup], system_prompt="s", checkpointer=graph.checkpointer)
    state = replay(replayer, "t2", checkpoint)
    assert state["messages"][-1].text == "Jawaban kedua."
    assert [m.type for m in second.calls[0][1:]] == ["human", "ai", "tool"]  # exact prefix was restored
    # The original branch is still in the history.
    texts = {m.text for s in graph.get_state_history(config) for m in s.values.get("messages", [])}
    assert {"Jawaban pertama.", "Jawaban kedua."} <= texts
