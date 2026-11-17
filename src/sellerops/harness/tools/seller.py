"""Seller agent tools: the six read/propose tools behind a ``SellerDataClient``.

One definition for eval and production, so the model sees the same names,
descriptions and argument schemas in both:

    model → tool (validate arguments, format the observation)
              → SellerDataClient.call(tool, arguments)
                  ├─ SellerEpisode     eval: simulator, records history for the verifier
                  ├─ in-process client production: the service layer FastAPI also uses (C2/C3)
                  └─ HTTP client       later, if agent and backend are deployed apart

The tools are thin adapters: business rules (revenue, restock, caption limits,
approval) live behind the client, never here. The client is bound per
store/user when the tools are built, so the model never picks a store or user
id. This module does not import ``sellerops.simulator`` or
``sellerops.evaluation``; ``SellerEpisode`` fits the Protocol structurally.

What the framework does here: ``model.bind_tools(tools)`` turns each
``args_schema`` into the JSON schema sent to the API and parses the reply into
``AIMessage.tool_calls``. ``tool.invoke(args)`` validates ``args`` against
``args_schema`` first and raises ``ValidationError`` on bad arguments (before
the client is called; the graph's ``tools`` node turns it into an error
``ToolMessage``). Business failures (unknown product, invalid caption) come
back from the client as ``{"ok": false, "error": ...}`` observations instead.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ...schemas import canonical_json

Period = Literal["last_7_days", "previous_7_days", "last_30_days", "previous_30_days", "last_90_days"]
Category = Literal["bags", "dresses", "fragrances", "jewellery", "shirts", "shoes", "sunglasses", "tops", "watches"]
Audience = Literal["men", "women", "unisex"]

_PERIOD_HELP = ("Fixed window ending today (the store's data date): last_7_days, previous_7_days "
                "(the 7 days before that), last_30_days, previous_30_days, last_90_days.")


class SellerDataClient(Protocol):
    """Where the seller tools get their data.

    ``call`` returns ``{"ok": True, "result": ...}`` or
    ``{"ok": False, "error": {"code": ..., "message": ...}}``; transport
    failures (timeouts, 5xx) should be retried or raised by the client, not
    returned as observations.
    """

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


_COMPARE_HELP = ("Also return the same numbers for the equal window right before (last_7_days vs "
                 "previous_7_days, last_30_days vs previous_30_days) with change percentages and a "
                 "trend (up/down/flat). Use it for any rise/fall question instead of computing "
                 "changes yourself.")


class SalesSummaryArgs(_Args):
    period: Period = Field(description=_PERIOD_HELP)
    group_by: Literal["category", "audience", "week"] | None = Field(
        default=None, description="Optional breakdown by product category, audience, or ISO week.")
    compare_to_previous: bool = Field(default=False, description=_COMPARE_HELP)


class ProductPerformanceArgs(_Args):
    period: Period = Field(description=_PERIOD_HELP)
    sort: Literal["units_desc", "units_asc", "revenue_desc", "revenue_asc"] = Field(
        default="units_desc", description="units_desc = best sellers first; units_asc = slowest first.")
    limit: int = Field(default=5, ge=1, le=54)
    category: Category | None = None
    audience: Audience | None = None
    compare_to_previous: bool = Field(default=False, description=_COMPARE_HELP)


class InventoryStatusArgs(_Args):
    category: Category | None = None
    target_days: int = Field(default=14, ge=1, le=90,
                             description="Days of demand the suggested restock should cover.")
    limit: int = Field(default=10, ge=1, le=54)


class ReviewsArgs(_Args):
    product_id: str | None = Field(default=None, description="Catalog id, e.g. FS188.")
    category: Category | None = None
    max_rating: int | None = Field(default=None, ge=1, le=5, description="e.g. 2 for complaints only.")
    period: Period = Field(default="last_90_days", description=_PERIOD_HELP)
    limit: int = Field(default=20, ge=1, le=50)


class ProductArgs(_Args):
    product_id: str = Field(min_length=1, description="Catalog id, e.g. FS180.")


class CaptionUpdateArgs(_Args):
    product_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, description="Why the change helps, for the seller reviewing it.")
    name: str | None = Field(default=None, description="New product name, at most 80 characters.")
    description: str | None = Field(default=None, description="New description, at most 600 characters.")
    tags: list[str] | None = Field(default=None, description="New tags, at most 8.")


SELLER_TOOL_SPECS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("get_sales_summary",
     "Revenue (IDR), number of orders, units and average order value for a period, "
     "optionally broken down by category, audience or week.",
     SalesSummaryArgs),
    ("get_product_performance",
     "Products ranked by units or revenue in a period (zero-sales products included), "
     "optionally filtered by category and audience. Each row has its current stock.",
     ProductPerformanceArgs),
    ("get_inventory_status",
     "Current stock with average daily units sold (last 30 days), days until stock runs out "
     "and a suggested restock quantity. Sorted by days_to_stockout, soonest first.",
     InventoryStatusArgs),
    ("get_reviews",
     "Buyer reviews (newest first) with a rating summary, filtered by product, category, "
     "maximum rating and period.",
     ReviewsArgs),
    ("get_product",
     "One product's catalog data: name, description, tags, prices, stock and caption_version.",
     ProductArgs),
    ("propose_caption_update",
     "Propose a new name, description and/or tags for a product. Creates a PENDING proposal for "
     "the seller to approve; it does not change the product. Only use facts from get_product.",
     CaptionUpdateArgs),
)
SELLER_TOOL_NAMES = frozenset(name for name, _, _ in SELLER_TOOL_SPECS)


def make_seller_tools(client: SellerDataClient) -> list[BaseTool]:
    """The six seller tools, all calling ``client``."""

    def build(name: str, description: str, schema: type[BaseModel]) -> BaseTool:
        def run(**kwargs: Any) -> str:
            # kwargs = the model's arguments with schema defaults filled in. Drop the None
            # optionals so the client gets e.g. no "category": None (defaults like period stay).
            arguments = schema(**kwargs).model_dump(exclude_none=True)
            return canonical_json(client.call(name, arguments))

        return StructuredTool.from_function(run, name=name, description=description, args_schema=schema)

    return [build(*spec) for spec in SELLER_TOOL_SPECS]
