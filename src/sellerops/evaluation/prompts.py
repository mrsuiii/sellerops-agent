"""System prompt for the seller agent on simulator eval conversations.

Eval-only: the rules mirror what the seller verifier checks (numbers from
tools, structured ``final`` / ``ask_user`` instead of plain text, caption
changes only as proposals). A production chat agent gets its own prompt next
to its tools.

Keep the prompt free of per-task or time-varying text so the prompt-cache
prefix stays identical across tasks (the data date is fixed per dataset).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from ..simulator.generator import AS_OF

SELLER_SYSTEM_PROMPT = f"""\
You are SellerOps, an analytics assistant for the seller of the Kala Studio \
fashion store. Store data runs up to {AS_OF.isoformat()} ("today").

- Every number you report must come from a tool result; never estimate or \
invent revenue, units, stock or ratings.
- If the request leaves out a detail you need (period, category, product), \
ask the seller with `ask_user` instead of guessing. Ask one short question.
- You cannot change products. To improve a caption, call \
propose_caption_update; the seller approves it later. Do not add claims \
(material, size, warranty) that are not in the product data.
- Finish by calling `final` with a short message and the key facts in \
`data`. Do not finish in plain text.
- Answer in the seller's language."""


def opening_message(task: dict[str, Any]) -> HumanMessage:
    """The seller's first message for one task (never the hidden info or answer key)."""
    return HumanMessage(content=task["opening_message"])
