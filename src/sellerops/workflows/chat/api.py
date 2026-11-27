"""HTTP routes of the seller chat, next to ``/api/seller``.

    GET  /api/seller/chat/threads/{thread_id}            the conversation so far (user and agent texts)
    POST /api/seller/chat/threads/{thread_id}/messages   {"text": "..."} -> the agent's reply

``thread_id`` is 32 hex characters, chosen by the browser. A message takes one
step from the demo budget (``approval/budget.py``) before the model runs.
Mounted by the store app with the agent runtime; errors use the seller API's
``SellerError`` mapping (``agent_unavailable`` is 503).
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Path
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field

from sellerops.seller import api as seller_api
from sellerops.workflows.approval import api as approval_api
from sellerops.workflows.approval.budget import LLMBudget
from sellerops.workflows.chat.fake import FakeChatModel
from sellerops.workflows.chat.runner import ChatRunner

router = APIRouter(prefix="/api/seller/chat", tags=["seller: chat"])

ThreadId = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$", description="Random id the browser keeps.")]
User = Annotated[str, Depends(seller_api.seller_user)]
Budget = Annotated[LLMBudget, Depends(approval_api.default_budget)]
Client = Annotated[str, Depends(approval_api.client_key)]


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)


@lru_cache(maxsize=1)
def chat_model() -> BaseChatModel:
    """The chat model: real Claude (``SELLEROPS_LLM_*``), or the free fake under ``SELLEROPS_APPROVAL_FAKE=1``."""
    if os.environ.get("SELLEROPS_APPROVAL_FAKE") == "1":
        return FakeChatModel()
    from sellerops.harness.models import get_chat_model

    return get_chat_model().with_config(tags=["seller-chat"])  # type: ignore[return-value]


_runners: dict[str, ChatRunner] = {}
_guard = threading.Lock()


def get_chat_runner(service: seller_api.Service,
                    model: Annotated[BaseChatModel, Depends(chat_model)],
                    checkpointer: Annotated[BaseCheckpointSaver, Depends(approval_api.default_checkpointer)]) -> ChatRunner:
    """One runner (compiled graph) per store; tests override this dependency."""
    with _guard:
        runner = _runners.get(service.store_id)
        if runner is None or runner.service is not service:
            runner = _runners[service.store_id] = ChatRunner(service, model, checkpointer)
        return runner


Runner = Annotated[ChatRunner, Depends(get_chat_runner)]


@router.get("/threads/{thread_id}")
def history(thread_id: ThreadId, runner: Runner) -> dict[str, object]:
    return {"thread_id": thread_id, "messages": runner.history(thread_id)}


@router.post("/threads/{thread_id}/messages")
def send(thread_id: ThreadId, body: MessageIn, runner: Runner, user: User, budget: Budget,
         client: Client) -> dict[str, object]:
    reply = runner.send(thread_id, body.text, actor=f"seller:{user}", charge=lambda: budget.take(client))
    return {"thread_id": thread_id, "reply": reply.text, "tools": reply.tools,
            "proposals": [p for p in reply.proposals if p]}


def install(app: FastAPI) -> None:
    """Mount the chat routes on ``app`` (after ``seller_api.install``, which maps the errors)."""
    app.include_router(router)
