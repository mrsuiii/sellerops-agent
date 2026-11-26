"""HTTP routes of the caption approval workflow (C3), next to ``/api/seller``.

    GET  /api/seller/caption-proposals/{id}/approval            the thread: waiting / done / error
    POST /api/seller/caption-proposals/{id}/approval/decision   {"decision", "feedback", "add_fields"}
    POST /api/seller/caption-proposals/{id}/approval/retry      run the failed step again

``{id}`` is any proposal of the chain; a decision must name the version the
graph is waiting on. The seller comes from ``X-Seller-User`` (the seller API's
``seller_user`` dependency) and is passed to the graph as ``user``. A reject
drafts the revision with the LLM inside the request, so it takes a few
seconds; the response already holds the revision to decide on.

Mounted by the store app only when the agent runtime (``[agent]`` extra) is
installed. Business errors reuse the seller API's ``SellerError`` mapping.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from sellerops.harness.trajectory import open_sqlite_checkpointer
from sellerops.seller import api as seller_api
from sellerops.settings import get_settings
from sellerops.workflows.approval.approval_graph import DecisionIn
from sellerops.workflows.approval.runner import ApprovalRunner

router = APIRouter(prefix="/api/seller/caption-proposals", tags=["seller: caption approval"])


@lru_cache(maxsize=1)
def default_model() -> BaseChatModel:
    """The LLM that drafts revisions; ``SELLEROPS_APPROVAL_FAKE=1`` swaps in a free fake (``make store-fake``)."""
    if os.environ.get("SELLEROPS_APPROVAL_FAKE") == "1":
        from sellerops.workflows.approval.studio import FakeDraftModel

        return FakeDraftModel()
    from sellerops.harness.models import get_chat_model

    return get_chat_model().with_config(tags=["caption-approval"])  # type: ignore[return-value]


@lru_cache(maxsize=1)
def default_checkpointer() -> BaseCheckpointSaver:
    return open_sqlite_checkpointer(get_settings().approvals_db_path)


_runners: dict[str, ApprovalRunner] = {}
_runners_guard = threading.Lock()


def get_runner(service: seller_api.Service,
               model: Annotated[BaseChatModel, Depends(default_model)],
               checkpointer: Annotated[BaseCheckpointSaver, Depends(default_checkpointer)]) -> ApprovalRunner:
    """One runner (compiled graph) per store, built on first use; tests override this dependency."""
    with _runners_guard:
        runner = _runners.get(service.store_id)
        if runner is None or runner.service is not service:
            runner = _runners[service.store_id] = ApprovalRunner(service, model, checkpointer)
        return runner


Runner = Annotated[ApprovalRunner, Depends(get_runner)]
User = Annotated[str, Depends(seller_api.seller_user)]


@router.get("/{proposal_id}/approval")
def approval(runner: Runner, proposal_id: str) -> dict[str, Any]:
    return runner.status(proposal_id)


@router.post("/{proposal_id}/approval/decision")
def decide(runner: Runner, user: User, proposal_id: str, body: DecisionIn) -> dict[str, Any]:
    return runner.decide(proposal_id, body, user=user)


@router.post("/{proposal_id}/approval/retry")
def retry(runner: Runner, _: User, proposal_id: str) -> dict[str, Any]:
    return runner.retry(proposal_id)


def install(app: FastAPI) -> None:
    """Mount the approval routes on ``app`` (after ``seller_api.install``, which maps the errors)."""
    app.include_router(router)
