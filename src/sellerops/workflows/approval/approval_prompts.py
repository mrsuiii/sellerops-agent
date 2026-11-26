"""Prompt and output schema for ``draft_revision``, the only LLM call in the approval graph.

The prompt is rebuilt from the state and the service on every call (no
message history): the model sees the current caption, the product facts it
may use, which fields it may change, every rejected version with the
seller's reason, and the validation error of its previous attempt.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sellerops.seller.analytics import CAPTION_LIMITS

NO_REASON = "(no reason given)"

# Facts the caption may mention. Everything else (sales, stock, ids) is not caption material.
CAPTION_FACTS = ("brand", "category", "audience", "price_idr", "discount_percent", "final_price_idr",
                 "warranty", "shipping", "return_policy")


class CaptionDraft(BaseModel):
    """A revised caption. Leave a field null to keep it as it is."""

    name: str | None = Field(None, description=f"New product name (max {CAPTION_LIMITS['name']} characters).")
    description: str | None = Field(
        None, description=f"New description (max {CAPTION_LIMITS['description']} characters).")
    tags: list[str] | None = Field(None, description=f"New tags (1 to {CAPTION_LIMITS['tags']}).")
    reason: str = Field(description="For the seller: what changed and how it answers their feedback.")


REVISION_SYSTEM_PROMPT = f"""\
You revise a product caption for the seller of the Kala Studio fashion store. \
The seller rejected the previous version(s); write a new version they will accept.

Rules:
- Change only the fields listed as allowed; return null for every other field.
- Address every reason the seller gave. If a rejection has no reason, write a \
version that is clearly different from all rejected versions (angle, length, tone).
- Never repeat a rejected version.
- Only state facts present in the product data (material, size, warranty, \
shipping...); never invent claims.
- Write in the language of the current caption.
- Limits: name {CAPTION_LIMITS['name']} characters, description \
{CAPTION_LIMITS['description']} characters, 1 to {CAPTION_LIMITS['tags']} tags.
- In `reason`, tell the seller briefly what you changed and why.
"""


def _caption(caption: dict[str, Any]) -> dict[str, Any]:
    return {key: caption.get(key) for key in ("name", "description", "tags")}


def revision_messages(*, product: dict[str, Any], fields: list[str], rejected: list[dict[str, Any]],
                      draft_error: str | None) -> list[BaseMessage]:
    """``rejected``: oldest first, each ``{"version": {field: new value}, "feedback": str}``."""
    sections = [
        "Current caption (what the shop shows now):\n" + json.dumps(_caption(product), ensure_ascii=False, indent=2),
        "Product data (the only facts you may use):\n"
        + json.dumps({key: product.get(key) for key in CAPTION_FACTS}, ensure_ascii=False, indent=2),
        "Fields you may change: " + ", ".join(fields),
        "Rejected versions, oldest first:\n" + "\n".join(
            f"{i}. {json.dumps(item['version'], ensure_ascii=False)}\n   seller's reason: {item['feedback'] or NO_REASON}"
            for i, item in enumerate(rejected, 1)),
    ]
    if draft_error:
        sections.append(f"Your previous draft was refused: {draft_error}\nFix that in this draft.")
    return [SystemMessage(REVISION_SYSTEM_PROMPT), HumanMessage("\n\n".join(sections))]
