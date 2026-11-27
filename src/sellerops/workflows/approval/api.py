"""HTTP routes of the caption approval workflow (C3, C3b), next to ``/api/seller``.

    GET  /api/seller/caption-proposals/{id}/approval            the thread: waiting / done / error
    POST /api/seller/caption-proposals/{id}/approval/decision   {"decision", "feedback", "add_fields"}
    POST /api/seller/caption-proposals/{id}/approval/answer     {"answers": {"material": "katun"}}
    POST /api/seller/caption-proposals/{id}/approval/retry      run the failed step again

    POST /api/seller/caption-requests                           {"product_id", "fields", "instruction"}
    GET  /api/seller/caption-requests/{id}/approval
    POST /api/seller/caption-requests/{id}/approval/answer
    POST /api/seller/caption-requests/{id}/approval/retry

``{id}`` is any proposal of the chain, or the caption request; a decision
must name the version the graph is waiting on. ``waiting_for`` in the view
says whether the graph waits for a decision or for answers (``questions.py``). The seller comes from ``X-Seller-User`` (the seller API's
``seller_user`` dependency) and is passed to the graph as ``user``. A reject,
an answer or a new request runs the LLM inside the HTTP request, so it takes
a few seconds; the response already holds what to do next.

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
from pydantic import BaseModel, ConfigDict, Field

from sellerops.harness.trajectory import open_sqlite_checkpointer
from sellerops.seller import SellerError
from sellerops.seller import api as seller_api
from sellerops.settings import get_settings
from sellerops.workflows.approval.approval_graph import CaptionField, DecisionIn
from sellerops.workflows.approval.questions import AnswerIn
from sellerops.workflows.approval.runner import ApprovalRunner

router = APIRouter(prefix="/api/seller/caption-proposals", tags=["seller: caption approval"])
requests_router = APIRouter(prefix="/api/seller/caption-requests", tags=["seller: caption approval"])


class CaptionRequestIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1)
    fields: list[CaptionField] = Field(["description"], min_length=1, description="Fields the agent writes.")
    instruction: str = Field("", max_length=1000, description='What the seller wants, e.g. "fokus ke bahan".')


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
    return runner.status(_proposal_id(proposal_id))


@router.post("/{proposal_id}/approval/decision")
def decide(runner: Runner, user: User, proposal_id: str, body: DecisionIn) -> dict[str, Any]:
    return runner.decide(proposal_id, body, user=user)


@router.post("/{proposal_id}/approval/answer")
def answer(runner: Runner, user: User, proposal_id: str, body: AnswerIn) -> dict[str, Any]:
    return runner.answer(_proposal_id(proposal_id), body, user=user)


@router.post("/{proposal_id}/approval/retry")
def retry(runner: Runner, _: User, proposal_id: str) -> dict[str, Any]:
    return runner.retry(_proposal_id(proposal_id))


@requests_router.post("", status_code=201)
def create_request(runner: Runner, user: User, body: CaptionRequestIn) -> dict[str, Any]:
    return runner.start_request(body.product_id, user=user, fields=list(body.fields), instruction=body.instruction)


@requests_router.get("/{request_id}/approval")
def request_approval(runner: Runner, request_id: str) -> dict[str, Any]:
    return runner.status(_request_id(request_id))


@requests_router.post("/{request_id}/approval/answer")
def request_answer(runner: Runner, user: User, request_id: str, body: AnswerIn) -> dict[str, Any]:
    return runner.answer(_request_id(request_id), body, user=user)


@requests_router.post("/{request_id}/approval/retry")
def request_retry(runner: Runner, _: User, request_id: str) -> dict[str, Any]:
    return runner.retry(_request_id(request_id))


def _request_id(value: str) -> str:
    """Keep the two route families apart: a proposal id under /caption-requests is not found."""
    if not value.startswith("CR-"):
        raise SellerError("request_not_found", f"no caption request with id {value!r}")
    return value


def _proposal_id(value: str) -> str:
    if not value.startswith("CP-"):
        raise SellerError("proposal_not_found", f"no proposal with id {value!r}")
    return value


def install(app: FastAPI) -> None:
    """Mount the approval routes on ``app`` (after ``seller_api.install``, which maps the errors)."""
    app.include_router(router)
    app.include_router(requests_router)
