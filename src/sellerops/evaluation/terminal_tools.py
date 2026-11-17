"""Terminal tools for seller eval tasks: ``final`` and ``ask_user``.

This is the evaluation contract: ``final.data`` is what the seller verifier
checks. The harness never imports this module; the caller (eval runner,
tests) passes ``TERMINAL_TOOLS`` to ``build_agent_graph``.

Protocol the graph relies on (generic, nothing seller-specific):

- a terminal tool is a Pydantic model class; its tool name is
  ``model_config["title"]`` (what ``bind_tools`` sends to the API);
- the model "calls" it to finish; the graph must NOT execute it, but run
  ``schema.model_validate(args).to_action()`` and end the run with the result;
- ``to_action()`` returns a JSON-serializable dict; ``model_validate`` raises
  ``pydantic.ValidationError`` when the arguments are invalid.

Multi-turn: ``ask_user`` ends the *turn*, not the conversation. The
conversation runner answers the pending ``ask_user`` call with a
``ToolMessage`` holding the seller's reply and invokes the graph again on the
same ``thread_id`` (see ``conversation.py``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FinalAnswer(BaseModel):
    """Finish the conversation with the answer for the seller. Use only numbers returned by tools.

    `data` holds the key facts as structured values, depending on the question:
    - sales totals: {"period", "revenue_idr", "orders"} (copy from get_sales_summary).
    - category with the highest revenue: {"period", "category"}.
    - best or slowest sellers: {"period", "product_ids": [...]} in ranking order.
    - what to restock: {"product_ids": [...]} for products that run out within 7 days
      (days_to_stockout <= 7), soonest first.
    - review complaints: {"product_id", "complaint"} (complaint = short description of the issue).
    - sales spike: {"period", "product_id"}.
    - sales drop: {"product_id", "cause"} where cause is one of
      "out_of_stock", "negative_reviews", "price", "seasonality", "other".
    - caption improvement: {"proposal_id", "product_id"} (from propose_caption_update).
    """

    model_config = ConfigDict(title="final")

    message: str = Field(min_length=1, description="The answer for the seller, in the seller's language.")
    data: dict[str, Any] = Field(description="Key facts of the answer as structured values (see above).")

    def to_action(self) -> dict[str, Any]:
        return {"type": "final", **self.model_dump()}


class AskUser(BaseModel):
    """Ask the seller a clarifying question when a needed detail is missing (period, category, product, ...).

    The seller's reply comes back as this tool's result.
    """

    model_config = ConfigDict(title="ask_user")

    question: str = Field(min_length=1)

    def to_action(self) -> dict[str, Any]:
        return {"type": "ask_user", **self.model_dump()}


TERMINAL_TOOLS: tuple[type[BaseModel], ...] = (FinalAnswer, AskUser)
TERMINAL_TOOL_NAMES = frozenset(schema.model_config["title"] for schema in TERMINAL_TOOLS)


def terminal_action(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``final`` / ``ask_user`` call and return its tagged action."""
    for schema in TERMINAL_TOOLS:
        if schema.model_config["title"] == name:
            return schema.model_validate(args).to_action()
    raise ValueError(f"{name!r} is not a terminal tool")
