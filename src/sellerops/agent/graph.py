"""Learning slice: general router -> SFT action -> tool -> general router."""

from functools import partial

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from sellerops.agent.nodes import call_model, router_node, tool_node
from sellerops.agent.state import AgentState
from sellerops.services.episode_service.episode_service import EpisodeService


def after_router(state: AgentState) -> str:
    """Routing functions choose an edge; they do not mutate state."""
    if state["status"] != "running":
        return "end"
    if state.get("pending_retry") == "router":
        return "retry"
    return "action"


def after_action(state: AgentState) -> str:
    if state["status"] == "failed":
        return "end"
    if state.get("pending_retry") == "action":
        return "retry"
    message = state["messages"][-1]
    if isinstance(message, AIMessage) and message.tool_calls:
        return "tools"
    # SFT may emit final or ask_user. The general model handles presentation.
    return "router"


def build_graph(service: EpisodeService):
    """Compile once; the caller creates episodes and supplies model aliases.

    Schema failures retry the same node within the shared episode retry budget.
    There is no checkpointer or inline outcome verifier. Model calls and tools
    are bounded independently; errors are never relabeled as successful answers.
    """
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("action", call_model)
    graph.add_node("tools", partial(tool_node, service=service))

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router,
                                {"action": "action", "retry": "router", "end": END})
    graph.add_conditional_edges(
        "action", after_action,
        {"tools": "tools", "router": "router", "retry": "action", "end": END}
    )
    graph.add_edge("tools", "router")
    return graph.compile()
