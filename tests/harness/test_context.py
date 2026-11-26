"""Session A4 context budget (summarize node) and runtime context injection. No network."""

from __future__ import annotations

from typing import Any

from fake_models import call, fake  # tests/harness is on sys.path
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from sellerops.harness.graph import build_agent_graph, summary_cut_index
from sellerops.harness.state import AgentContext


@tool
def whoami(topic: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Echo the store the request belongs to."""
    return f"{topic} for store {runtime.context.store_id} (user {runtime.context.user_id})"


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


def test_cut_index_never_orphans_tool_results() -> None:
    ai = AIMessage("", tool_calls=[{"name": "lookup", "args": {"query": "a"}, "id": "1"}])
    messages = [HumanMessage("h1"), ai, ToolMessage("r", tool_call_id="1"), AIMessage("a1"),
                HumanMessage("h2"), AIMessage("a2")]
    for keep in range(2, len(messages) + 1):
        cut = summary_cut_index(messages, keep)
        assert cut == 0 or not isinstance(messages[cut], ToolMessage)
        assert len(messages) - cut >= keep
    assert summary_cut_index(messages, 2) == 4 and summary_cut_index(messages, 6) == 0


def test_tool_gets_runtime_context_and_model_never_sees_it() -> None:
    model = fake(call("whoami", {"topic": "omzet"}), AIMessage("Selesai."))
    graph = build_agent_graph(model, [whoami], system_prompt="s")
    state = graph.invoke({"messages": [HumanMessage("siapa saya?")]},
                         context=AgentContext(user_id="u7", store_id="kala"))
    result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert result.content == "omzet for store kala (user u7)"
    assert state["stop_reason"] == "answer"
    assert "runtime" not in whoami.tool_call_schema.model_json_schema()["properties"]


def test_long_thread_is_summarized_and_continues() -> None:
    def big(name: str, args: dict[str, Any]) -> AIMessage:
        message = call(name, args)
        message.usage_metadata = {"input_tokens": 5000, "output_tokens": 10, "total_tokens": 5010}
        return message

    agent = fake(big("lookup", {"query": "omzet"}), AIMessage("Omzet Rp100.", usage_metadata={
                     "input_tokens": 5000, "output_tokens": 5, "total_tokens": 5005}),
                 AIMessage("Masih Rp100."))
    summarizer = fake(AIMessage("User tanya omzet; lookup: Rp100."))
    graph = build_agent_graph(agent, [lookup], system_prompt="s", checkpointer=InMemorySaver(),
                              max_context_tokens=4000, keep_messages=2, summary_model=summarizer)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("Berapa omzet?")]}, config)
    state = graph.invoke({"messages": [HumanMessage("Ulangi angkanya?")]}, config)

    assert len(summarizer.calls) == 1
    assert "lookup" in summarizer.calls[0][1].content  # old messages went to the summarizer
    assert state["summary"] == "User tanya omzet; lookup: Rp100."
    # Old messages were removed; the kept tail starts at a safe boundary.
    assert [type(m).__name__ for m in state["messages"]] == ["AIMessage", "HumanMessage", "AIMessage"]
    last_prompt = agent.calls[-1]
    assert "<conversation_summary>" in last_prompt[1].content
    assert not isinstance(last_prompt[2], ToolMessage)
    assert state["stop_reason"] == "answer"


def test_no_summary_below_the_budget() -> None:
    agent = fake(AIMessage("a"), AIMessage("b"))
    summarizer = fake(AIMessage("never"))
    graph = build_agent_graph(agent, [lookup], system_prompt="s", checkpointer=InMemorySaver(),
                              max_context_tokens=4000, summary_model=summarizer)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("1")]}, config)
    graph.invoke({"messages": [HumanMessage("2")]}, config)
    assert summarizer.calls == []
