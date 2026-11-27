"""Run the seller chat graph: one ``send`` is one agent turn in a thread.

- **Threads.** A conversation is a LangGraph thread named after the store and
  a random id the browser keeps. The checkpointer stores every message, so the
  agent sees the whole conversation (older turns are summarised once the
  prompt grows past ``MAX_CONTEXT_TOKENS``).
- **Reply.** The last plain-text AIMessage of the turn is the reply. Tool calls
  of the turn are reported by name and outcome; a successful
  ``propose_caption_update`` lists its proposal id, so the UI can link it.
- **Budget.** ``charge`` runs before the model is called (demo budget), so a
  refused message costs nothing.
- **Failures.** A model or network error becomes ``SellerError("agent_unavailable")``
  (HTTP 503); the thread keeps the messages up to the failure.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from sellerops.harness.graph import build_agent_graph
from sellerops.harness.state import AgentContext
from sellerops.harness.tools.seller import make_seller_tools
from sellerops.seller import SellerError, SellerService
from sellerops.seller.service import InProcessSellerClient
from sellerops.workflows.chat.prompts import CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

Charge = Callable[[], None]
TURN_SECONDS = 60.0
MAX_CONTEXT_TOKENS = 60_000
KEEP_MESSAGES = 10
BUDGET_STOPS = {"max_steps", "token_budget", "time_budget"}
PROPOSE = "propose_caption_update"


def _free() -> None:
    return None


@dataclass
class Reply:
    text: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)


def text_of(content: Any) -> str:
    """Plain text of a message: a string, or the text blocks of a content list (thinking is skipped)."""
    if isinstance(content, str):
        return content.strip()
    return "".join(block.get("text", "") for block in content
                   if isinstance(block, dict) and block.get("type") == "text").strip()


def _tool_result(content: Any) -> dict[str, Any]:
    try:
        data = json.loads(text_of(content) or "{}")
    except json.JSONDecodeError:
        return {"ok": False}
    return data if isinstance(data, dict) else {"ok": False}


def turn_reply(messages: list[AnyMessage], stop_reason: str | None) -> Reply:
    """What the latest turn (the messages after the last human message) says and did."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    turn = messages[last_human + 1:]
    names = {call["id"]: call["name"] for m in turn if isinstance(m, AIMessage) for call in m.tool_calls}
    reply = Reply(text="")
    for message in turn:
        if isinstance(message, ToolMessage):
            result = _tool_result(message.content)
            name = names.get(message.tool_call_id, message.name or "")
            reply.tools.append({"name": name, "ok": bool(result.get("ok"))})
            if name == PROPOSE and result.get("ok") and isinstance(result.get("result"), dict):
                reply.proposals.append(str(result["result"].get("proposal_id", "")))
        elif isinstance(message, AIMessage) and not message.tool_calls and text_of(message.content):
            reply.text = text_of(message.content)
    if not reply.text:
        reply.text = ("Maaf, pertanyaan ini terlalu panjang untuk saya jawab dalam satu giliran. "
                      "Coba persempit, misalnya dengan periode atau satu produk."
                      if stop_reason in BUDGET_STOPS else "Maaf, saya belum punya jawaban untuk itu.")
    return reply


class ChatRunner:
    """The seller chat graph of one store."""

    def __init__(self, service: SellerService, model: BaseChatModel, checkpointer: BaseCheckpointSaver, *,
                 graph: Any = None) -> None:
        self.service = service
        self.graph = graph if graph is not None else build_agent_graph(
            model,
            make_seller_tools(InProcessSellerClient(service, actor="agent:chat")),
            system_prompt=CHAT_SYSTEM_PROMPT,
            checkpointer=checkpointer,
            max_turn_seconds=TURN_SECONDS,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            keep_messages=KEEP_MESSAGES,
            summary_model=model,
        )

    def config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": f"{self.service.store_id}:seller-chat-{thread_id}"}}

    def send(self, thread_id: str, text: str, *, actor: str, charge: Charge = _free) -> Reply:
        charge()
        try:
            state = self.graph.invoke({"messages": [HumanMessage(text)]}, self.config(thread_id),
                                      context=AgentContext(user_id=actor, store_id=self.service.store_id))
        except Exception as error:
            logger.exception("seller chat turn failed", extra={"chat_thread": thread_id})
            raise SellerError("agent_unavailable", "Agen sedang tidak bisa menjawab. Coba lagi sebentar.") from error
        return turn_reply(state["messages"], state.get("stop_reason"))

    def history(self, thread_id: str) -> list[dict[str, str]]:
        snapshot = self.graph.get_state(self.config(thread_id))
        messages = snapshot.values.get("messages", []) if snapshot else []
        out: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                out.append({"role": "user", "text": text_of(message.content)})
            elif isinstance(message, AIMessage) and not message.tool_calls and text_of(message.content):
                out.append({"role": "agent", "text": text_of(message.content)})
        return out
