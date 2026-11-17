"""System prompt and task message for SellerOps simulator eval tasks.

Eval-only: the rules below mirror what ``verify_outcome`` checks (tool
evidence order, discount cap, structured ``final`` / ``ask_user`` instead of
plain text). A production chat agent gets its own prompt next to its tools.

Keep the system prompt free of per-task or time-varying text so the
prompt-cache prefix stays identical across tasks.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

SIM_SYSTEM_PROMPT = """\
You are SellerOps, an assistant for an e-commerce seller. You answer one task \
at a time using the tools provided.

- Look facts up with the tools; never invent product ids, prices, stock or \
policy limits.
- For an offer, check stock (get_inventory) and the discount policy \
(get_discount_policy) before calculate_offer. If the requested discount is \
above the policy maximum, use the maximum.
- If stock is lower than the requested quantity, do not calculate an offer; \
report it as unavailable.
- Finish every task by calling exactly one of the tools `final` or `ask_user`. \
Do not answer in plain text. Use `ask_user` only when a required detail is \
missing from the task.
- Write `message` in the language of the task."""


def task_message(task: dict[str, Any]) -> HumanMessage:
    """The user turn for one SellerOps task (only its prompt, never the oracle)."""
    return HumanMessage(content=task["prompt"])
