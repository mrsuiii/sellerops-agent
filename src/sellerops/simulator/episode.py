"""One isolated seller-simulator episode: its own store state and tool history.

The dataset (orders, stock, reviews) is shared read-only between episodes;
what an episode can change (caption proposals) belongs to its own
``SellerStore``. Every tool call is recorded so the verifier can check that
the numbers in the final answer came from a tool ("grounding").

Observations follow one shape:

    {"ok": true,  "result": {...}}
    {"ok": false, "error": {"code": "product_not_found", "message": "..."}}

Only ``ToolError`` (a business failure) becomes ``ok: false``. Anything else
(a bug, an unknown tool name) raises; deciding what the graph does with an
exception is part of the tools node (A1/A2).
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
from typing import Any

from .store import SellerStore, SimData, ToolError, load_store


class SellerEpisode:
    def __init__(self, data: SimData | None = None, *, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.store: SellerStore = load_store(data)
        self.history: list[dict[str, Any]] = []
        self._dispatch: dict[str, Callable[..., dict[str, Any]]] = {
            "get_sales_summary": self.store.sales_summary,
            "get_product_performance": self.store.product_performance,
            "get_inventory_status": self.store.inventory_status,
            "get_reviews": self.store.reviews,
            "get_product": self.store.product,
            "propose_caption_update": self.store.propose_caption_update,
        }

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._dispatch)

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool and record it. Raises ``KeyError`` for an unknown tool."""
        handler = self._dispatch[tool]
        try:
            observation: dict[str, Any] = {"ok": True, "result": handler(**arguments)}
        except ToolError as error:
            observation = {"ok": False, "error": {"code": error.code, "message": error.message}}
        self.history.append({
            "step": len(self.history) + 1,
            "tool": tool,
            "arguments": copy.deepcopy(arguments),
            "observation": copy.deepcopy(observation),
        })
        return observation

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable view of the episode for the verifier and trajectory logs."""
        return {
            "session_id": self.session_id,
            "data_version": self.store.data.manifest.get("data_version"),
            "as_of": self.store.as_of.isoformat(),
            "history": copy.deepcopy(self.history),
            "proposals": copy.deepcopy(self.store.proposals),
        }
