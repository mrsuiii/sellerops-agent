"""Terminal tools for SellerOps eval tasks: ``final`` / ``ask_user``.

This is the evaluation contract: the shape of ``final.data`` is what
``verify_outcome`` checks. The harness never imports this module; the caller
(eval target, tests) passes ``TERMINAL_TOOLS`` to ``build_agent_graph``.

Protocol the graph relies on (generic, nothing SellerOps-specific):

- a terminal tool is a Pydantic model class; its tool name is
  ``model_config["title"]`` (what ``bind_tools`` sends to the API);
- the model "calls" it to finish; the graph must NOT execute it, but run
  ``schema.model_validate(args).to_action()`` and end the run with the result;
- ``to_action()`` returns a JSON-serializable dict or raises
  (``pydantic.ValidationError`` from ``model_validate``, ``SchemaError`` from
  ``to_action``) when the arguments are invalid.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import validate_action


class FinalAnswer(BaseModel):
    """Finish the task with a structured answer. Use only facts returned by tools.

    `data` must contain exactly these keys, depending on the task:
    - product search, found: {"products": [{"product_id", "name", "price_idr", "stock"}, ...]}
      with status "success" (copy the product objects from search_products).
    - product search, nothing matches: {"products": []} with status "not_found".
    - offer: {"product_id", "quantity", "discount_percent", "unit_price_idr", "total_idr"}
      with status "success" (copy from calculate_offer; cap the discount at the policy maximum).
    - stock too low for the requested quantity: {"product_id", "available_stock",
      "requested_quantity"} with status "unavailable" (do not calculate an offer).
    - discount policy question: {"product_id", "max_discount_percent", "source"}, plus
      {"proposed_discount_percent", "proposal_allowed"} when a discount was proposed, or
      {"discount_eligible"} when asked whether any discount is allowed; status "success".
    """

    model_config = ConfigDict(title="final")

    status: Literal["success", "unavailable", "not_found"]
    message: str = Field(min_length=1, description="Short answer for the seller, in the task's language.")
    data: dict[str, Any]

    def to_action(self) -> dict[str, Any]:
        action = {"type": "final", **self.model_dump()}
        validate_action(action)
        return action


class AskUser(BaseModel):
    """Ask the seller a clarifying question when a required detail is missing. Ends the turn."""

    model_config = ConfigDict(title="ask_user")

    question: str = Field(min_length=1)
    missing_fields: list[str] = Field(min_length=1, description="Names of the missing details.")

    def to_action(self) -> dict[str, Any]:
        action = {"type": "ask_user", **self.model_dump()}
        validate_action(action)
        return action


TERMINAL_TOOLS: tuple[type[BaseModel], ...] = (FinalAnswer, AskUser)
TERMINAL_TOOL_NAMES = frozenset(schema.model_config["title"] for schema in TERMINAL_TOOLS)


def terminal_action(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``final`` / ``ask_user`` call and return its tagged action."""
    for schema in TERMINAL_TOOLS:
        if schema.model_config["title"] == name:
            return schema.model_validate(args).to_action()
    raise ValueError(f"{name!r} is not a terminal tool")
