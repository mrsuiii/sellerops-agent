"""Entry points for LangGraph Studio (``make studio``; graphs listed in ``langgraph.json``).

Two graphs:

- ``caption_approval``: the approval graph as is. Input: ``{"proposal_id": "CP-0001"}``
  (a pending proposal that already exists in the Studio database).
- ``caption_approval_demo``: a wrapper for trying the flow from scratch. Its first
  node plays the chat agent and creates the proposal from the input
  (``product_id``, ``description`` and/or ``name`` / ``tags``, ``reason``; all
  optional, defaults to a long description for FS180). The approval graph then
  runs as a **subgraph**, so Studio shows both levels and the interrupt.

Studio runs the graph on its own dev server with its own persistence, so
these factories pass no checkpointer. The seller data lives in
``var/studio/seller.sqlite`` (``SELLEROPS_STUDIO_DB``); delete it to reset.
``SELLEROPS_STUDIO_FAKE=1`` swaps the LLM for a free fake that writes
numbered revisions.
"""

from __future__ import annotations

import itertools
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sellerops.seller import SellerService
from sellerops.settings import get_settings
from sellerops.workflows.approval.approval_graph import build_approval_graph
from sellerops.workflows.approval.approval_state import ApprovalState

DEMO_PRODUCT = "FS180"
DEMO_DESCRIPTION = (
    "The Dress Pea is a stylish and comfortable dress with a playful pea pattern. It is perfect for casual "
    "outings, weekend brunches, park walks and relaxed days in the city. The cheerful print adds a fun touch "
    "to your wardrobe, and the easy shape makes it a great choice for everyday wear in any season."
)


class FakeDraftModel(BaseChatModel):
    """Free stand-in for the LLM: every revision is a numbered placeholder description."""

    counter: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake-draft"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeDraftModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        if self.counter is None:
            self.counter = itertools.count(1)
        n = next(self.counter)
        draft = {"description": f"(fake revision {n}) A playful everyday dress.", "reason": "fake model"}
        message = AIMessage("", tool_calls=[{"name": "CaptionDraft", "args": draft, "id": f"fake{n}"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


@lru_cache(maxsize=1)
def studio_service() -> SellerService:
    settings = get_settings()
    db = Path(os.environ.get("SELLEROPS_STUDIO_DB") or settings.project_root / "var" / "studio" / "seller.sqlite")
    return SellerService.open(db, catalog_path=settings.catalog_path, sim_dir=settings.data_dir / "seller_sim")


@lru_cache(maxsize=1)
def studio_model() -> BaseChatModel | Any:
    if os.environ.get("SELLEROPS_STUDIO_FAKE") == "1":
        return FakeDraftModel()
    from sellerops.harness.models import get_chat_model

    return get_chat_model().with_config(tags=["studio"])


def caption_approval() -> CompiledStateGraph:
    return build_approval_graph(studio_service(), studio_model())


class DemoInput(TypedDict, total=False):
    product_id: str
    description: str
    name: str
    tags: list[str]
    reason: str


class DemoState(ApprovalState, DemoInput, total=False):
    pass


def caption_approval_demo() -> CompiledStateGraph:
    service = studio_service()

    def agent_proposes(state: DemoState) -> dict[str, Any]:
        """Stands in for the chat agent's ``propose_caption_update`` call."""
        changes = {key: state[key] for key in ("name", "description", "tags") if state.get(key)}
        proposal = service.propose_caption(
            state.get("product_id") or DEMO_PRODUCT, state.get("reason") or "make it more engaging",
            actor="agent", **(changes or {"description": DEMO_DESCRIPTION}))
        return {"proposal_id": proposal["proposal_id"]}

    builder = StateGraph(DemoState, input_schema=DemoInput)
    builder.add_node("agent_proposes", agent_proposes)
    builder.add_node("approval", build_approval_graph(service, studio_model()))  # subgraph
    builder.add_edge(START, "agent_proposes")
    builder.add_edge("agent_proposes", "approval")
    return builder.compile()
