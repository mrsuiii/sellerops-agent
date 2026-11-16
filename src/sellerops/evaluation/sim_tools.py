"""Simulator tools for evaluation (SellerOps tasks, ``EpisodeService``).

Eval-only: every call runs through ``EpisodeService.execute_tool`` so it is
recorded in the episode history that ``verify_outcome`` replays. The tools are
built per episode (closures over ``service`` and ``session_id``) so the model
never sees or chooses the session id. The harness never imports this module;
the eval target (or a test) passes these tools to ``build_agent_graph``.
Production tools live in ``sellerops.harness.tools`` (Track C / B).

What the framework does here: ``model.bind_tools(tools)`` turns each
``args_schema`` into the JSON schema sent to the API, and parses the reply
into ``AIMessage.tool_calls`` (``{"name", "args", "id"}``). ``tool.invoke(args)``
validates ``args`` against ``args_schema`` before the function runs (a
``ValidationError`` is raised on bad arguments).

Errors are not caught here. ``EpisodeService.execute_tool`` raises
``SchemaError`` (invalid arguments) and ``ValueError`` (episode terminal or its
own 8-step limit reached); business failures (unknown product, insufficient
stock, policy violation) come back as observations with ``ok: false``.
Deciding what the graph does with each is part of the tools node (A1) and
Session A2.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ..schemas import canonical_json
from ..simulator.episode_service import EpisodeService

# ---------------------------------------------------------------- arguments


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str | None = Field(default=None, description="Exact brand name, case-insensitive.")
    category: str | None = Field(default=None, description="Exact category.")
    color: str | None = Field(default=None, description="Exact color.")
    in_stock: bool | None = Field(default=None, description="true to keep only products with stock > 0.")
    max_price_idr: int | None = Field(default=None, ge=0, description="Maximum unit price in IDR.")


class SearchProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Text matched against name, category, brand and tags.")
    filters: SearchFilters = Field(default_factory=SearchFilters)


class ProductIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, description="Catalog id, e.g. P1008.")


class CalculateOfferArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, description="Catalog id, e.g. P1008.")
    discount_percent: int = Field(ge=0, le=100, description="Integer discount, at most the policy maximum.")
    quantity: int = Field(ge=1, description="Number of units.")


# ---------------------------------------------------------- simulator tools

_SIM_TOOL_SPECS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    (
        "search_products",
        "Search the catalog. Returns up to five products with product_id, name, "
        "price_idr and stock. An empty list means nothing matched.",
        SearchProductsArgs,
    ),
    (
        "get_inventory",
        "Read the current stock of one product. Fails with product_not_found for an unknown id.",
        ProductIdArgs,
    ),
    (
        "get_discount_policy",
        "Read the maximum allowed discount and the policy source for one product.",
        ProductIdArgs,
    ),
    (
        "calculate_offer",
        "Compute the discounted unit and total price. Call get_inventory and "
        "get_discount_policy first; fails on insufficient_stock or "
        "discount_policy_violation.",
        CalculateOfferArgs,
    ),
)
SIM_TOOL_NAMES = frozenset(name for name, _, _ in _SIM_TOOL_SPECS)


def _to_arguments(name: str, args: BaseModel) -> dict[str, Any]:
    arguments = args.model_dump()
    if name == "search_products":
        # The simulator only accepts filter keys that are actually set.
        arguments["filters"] = {k: v for k, v in arguments["filters"].items() if v is not None}
    return arguments


def make_sim_tools(service: EpisodeService, session_id: str) -> list[BaseTool]:
    """Build the four simulator tools bound to one episode."""

    def build(name: str, description: str, schema: type[BaseModel]) -> BaseTool:
        def run(**kwargs: Any) -> str:
            action = {"type": "tool_call", "tool": name, "arguments": _to_arguments(name, schema(**kwargs))}
            observation = service.execute_tool(session_id, action)
            return canonical_json(observation)

        return StructuredTool.from_function(run, name=name, description=description, args_schema=schema)

    return [build(*spec) for spec in _SIM_TOOL_SPECS]
