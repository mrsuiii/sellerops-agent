"""Prompt and output schemas for ``draft_caption``, the only LLM call in the approval graph.

The prompt is rebuilt from the state and the service on every call (no
message history). The model gets three sources of facts, kept apart so it
knows what each one can prove:

1. **Product data** (catalog): brand, category, price, warranty, shipping...
2. **Photos** (C3b, when the model has vision): only what is plainly visible,
   such as colour, pattern, cut. A photo does not prove material or size.
3. **Seller facts** (memory, ``product_facts``): what the seller answered
   earlier. ``null`` means the seller does not know.

Anything outside these three is not written. When the caption needs a fact
none of them holds, the model may call ``AskSeller`` once per version
(``can_ask``); if it cannot ask, or the seller does not know, it writes with
what it has and says in ``reason`` which data is still missing.

The model answers with exactly one tool call: ``CaptionDraft`` or ``AskSeller``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sellerops.seller.analytics import CAPTION_LIMITS
from sellerops.workflows.approval.questions import MAX_QUESTIONS, SellerQuestion

NO_REASON = "(no reason given)"

# Facts the caption may mention. Everything else (sales, stock, ids) is not caption material.
CAPTION_FACTS = ("brand", "category", "audience", "price_idr", "discount_percent", "final_price_idr",
                 "warranty", "shipping", "return_policy")


class CaptionDraft(BaseModel):
    """A caption draft. Leave a field null to keep it as it is."""

    name: str | None = Field(None, description=f"New product name (max {CAPTION_LIMITS['name']} characters).")
    description: str | None = Field(
        None, description=f"New description (max {CAPTION_LIMITS['description']} characters).")
    tags: list[str] | None = Field(None, description=f"New tags (1 to {CAPTION_LIMITS['tags']}).")
    reason: str = Field(description="For the seller, in Indonesian: what you wrote or changed and why; "
                                    "name any product data that is still missing.")


class AskSeller(BaseModel):
    """Ask the seller for facts the caption needs and no source holds. Do not draft in the same turn."""

    questions: list[SellerQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)


SYSTEM_PROMPT = f"""\
You write product captions for the seller of Kala Studio, a fashion store. \
A human seller approves every caption before the shop shows it.

Sources of facts. Use only these, and only for what each can prove:
1. Product data: the catalog fields given (brand, category, price, warranty, shipping, returns).
2. Photos (if attached): only what is plainly visible, such as colour, pattern, cut and style. \
A photo never proves material, size, fabric weight or care.
3. Seller facts: what the seller told you. A null value means the seller does not know.
Never state anything else (no invented material, size, origin, certification or awards).

Asking (only when the AskSeller tool is offered):
- Call AskSeller when the request or the seller's feedback needs a fact that none of the sources holds \
(for example the material when they ask you to mention it). Do not ask for facts you already have, for \
facts listed as unknown, or for things the caption does not need.
- At most {MAX_QUESTIONS} short questions in Indonesian. Each `key` is the fact's snake_case name \
(material, fit, size_range, care...).
- Either ask or draft in one turn, never both.

Drafting (CaptionDraft):
- Change only the fields listed as allowed; return null for every other field.
- If a needed fact is unknown or you could not ask, write the caption with what you have, leave the \
missing fact out, and say in `reason` that more product data is needed (name which).
- Write the caption in the language of the current caption. Write `reason` in Indonesian.
- Limits: name {CAPTION_LIMITS['name']} characters, description {CAPTION_LIMITS['description']} \
characters, 1 to {CAPTION_LIMITS['tags']} tags.
- On a revision: address every reason the seller gave and never repeat a rejected version. If a \
rejection has no reason, write a version clearly different from all rejected ones (angle, length, tone).
"""


def _caption(caption: dict[str, Any]) -> dict[str, Any]:
    return {key: caption.get(key) for key in ("name", "description", "tags")}


def draft_messages(*, product: dict[str, Any], fields: list[str], facts: dict[str, str | None],
                   rejected: list[dict[str, Any]], instruction: str = "", draft_error: str | None = None,
                   images: list[dict[str, Any]] | None = None, can_ask: bool = False) -> list[BaseMessage]:
    """The prompt for a first draft (``rejected=[]``) or a revision.

    ``rejected``: oldest first, each ``{"version": {field: new value}, "feedback": str}``.
    ``images``: LangChain image content blocks (``to_content_block``), already shrunk.
    """
    sections = [
        "Current caption (what the shop shows now):\n" + json.dumps(_caption(product), ensure_ascii=False, indent=2),
        "1. Product data:\n" + json.dumps({key: product.get(key) for key in CAPTION_FACTS},
                                          ensure_ascii=False, indent=2),
        "2. Photos: " + (f"{len(images)} attached below." if images else "none (describe no visual details "
                                                                           "that are not in the other sources)."),
        "3. Seller facts:\n" + (json.dumps(facts, ensure_ascii=False, indent=2) if facts else "(none yet)"),
        "Fields you may write: " + ", ".join(fields),
    ]
    if rejected:
        sections.append("Rejected versions, oldest first:\n" + "\n".join(
            f"{i}. {json.dumps(item['version'], ensure_ascii=False)}\n   seller's reason: {item['feedback'] or NO_REASON}"
            for i, item in enumerate(rejected, 1)))
        sections.append("Task: write a revision the seller will accept.")
    else:
        sections.append("Seller's request: " + (instruction or "(none: write a clear, appealing caption)"))
        sections.append("Task: write the first version of the caption.")
    sections.append("You may call AskSeller or CaptionDraft." if can_ask
                    else "Asking is not available now: call CaptionDraft.")
    if draft_error:
        sections.append(f"Your previous answer was refused: {draft_error}\nFix that in this answer.")
    text = "\n\n".join(sections)
    content: str | list[Any] = [{"type": "text", "text": text}, *images] if images else text
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(content)]
