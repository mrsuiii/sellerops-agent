"""HTTP routes of the caption approval workflow (C3, C3b), next to ``/api/seller``.

    GET  /api/seller/caption-proposals/{id}/approval            the thread: waiting / done / error
    POST /api/seller/caption-proposals/{id}/approval/decision   {"decision", "feedback", "add_fields"}
    POST /api/seller/caption-proposals/{id}/approval/answer     {"answers": {"material": "katun"}}
    POST /api/seller/caption-proposals/{id}/approval/retry      run the failed step again

    POST /api/seller/caption-requests                           {"product_id", "fields", "instruction"}
    GET  /api/seller/caption-requests/{id}/approval
    POST /api/seller/caption-requests/{id}/approval/answer
    POST /api/seller/caption-requests/{id}/approval/retry

    GET  /api/seller/llm-budget                                 demo limits left for this client

Routes that run the LLM take one step from ``LLMBudget`` (demo mode only,
``budget.py``) once the request is valid, right before the LLM runs: a
refused request (stale version, wrong answers) and approve / cancel are free.

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

import importlib.util
import os
import threading
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Request
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field

from sellerops.harness.trajectory import open_sqlite_checkpointer
from sellerops.seller import SellerError
from sellerops.seller import api as seller_api
from sellerops.seller.images import LocalImageStore, ProductImages
from sellerops.settings import get_settings
from sellerops.workflows.approval.approval_graph import CaptionField, DecisionIn
from sellerops.workflows.approval.budget import LLMBudget
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
def default_budget() -> LLMBudget:
    settings = get_settings()
    return LLMBudget(per_client_hour=settings.demo_llm_per_client_hour, per_day=settings.demo_llm_per_day,
                     enabled=settings.demo_mode)


def client_key(request: Request) -> str:
    """Who the budget counts: the proxy's client-IP header when configured, else the socket peer."""
    header = get_settings().client_ip_header
    value = request.headers.get(header) if header else None
    return (value or (request.client.host if request.client else "") or "unknown").strip()


@lru_cache(maxsize=1)
def default_checkpointer() -> BaseCheckpointSaver:
    return open_sqlite_checkpointer(get_settings().approvals_db_path)


_runners: dict[str, ApprovalRunner] = {}
_runners_guard = threading.Lock()


def product_images(service: seller_api.Service) -> ProductImages | None:
    """Product photos for the draft (C3b); ``None`` without Pillow (the graph then drafts from text)."""
    if importlib.util.find_spec("PIL") is None:
        return None
    return ProductImages(LocalImageStore(get_settings().images_dir, store_id=service.store_id))


def get_runner(service: seller_api.Service,
               model: Annotated[BaseChatModel, Depends(default_model)],
               checkpointer: Annotated[BaseCheckpointSaver, Depends(default_checkpointer)]) -> ApprovalRunner:
    """One runner (compiled graph) per store, built on first use; tests override this dependency."""
    with _runners_guard:
        runner = _runners.get(service.store_id)
        if runner is None or runner.service is not service:
            runner = _runners[service.store_id] = ApprovalRunner(service, model, checkpointer,
                                                                 images=product_images(service))
        return runner


Runner = Annotated[ApprovalRunner, Depends(get_runner)]
User = Annotated[str, Depends(seller_api.seller_user)]
Budget = Annotated[LLMBudget, Depends(default_budget)]
Client = Annotated[str, Depends(client_key)]


@router.get("/{proposal_id}/approval")
def approval(runner: Runner, proposal_id: str) -> dict[str, Any]:
    return runner.status(_proposal_id(proposal_id))


@router.post("/{proposal_id}/approval/decision")
def decide(runner: Runner, user: User, budget: Budget, client: Client, proposal_id: str,
           body: DecisionIn) -> dict[str, Any]:
    return runner.decide(_proposal_id(proposal_id), body, user=user, charge=lambda: budget.take(client))


@router.post("/{proposal_id}/approval/answer")
def answer(runner: Runner, user: User, budget: Budget, client: Client, proposal_id: str,
           body: AnswerIn) -> dict[str, Any]:
    return runner.answer(_proposal_id(proposal_id), body, user=user, charge=lambda: budget.take(client))


@router.post("/{proposal_id}/approval/retry")
def retry(runner: Runner, _: User, budget: Budget, client: Client, proposal_id: str) -> dict[str, Any]:
    return runner.retry(_proposal_id(proposal_id), charge=lambda: budget.take(client))


@requests_router.post("", status_code=201)
def create_request(runner: Runner, user: User, budget: Budget, client: Client,
                   body: CaptionRequestIn) -> dict[str, Any]:
    return runner.start_request(body.product_id, user=user, fields=list(body.fields), instruction=body.instruction,
                                charge=lambda: budget.take(client))


@requests_router.get("/{request_id}/approval")
def request_approval(runner: Runner, request_id: str) -> dict[str, Any]:
    return runner.status(_request_id(request_id))


@requests_router.post("/{request_id}/approval/answer")
def request_answer(runner: Runner, user: User, budget: Budget, client: Client, request_id: str,
                   body: AnswerIn) -> dict[str, Any]:
    return runner.answer(_request_id(request_id), body, user=user, charge=lambda: budget.take(client))


@requests_router.post("/{request_id}/approval/retry")
def request_retry(runner: Runner, _: User, budget: Budget, client: Client, request_id: str) -> dict[str, Any]:
    return runner.retry(_request_id(request_id), charge=lambda: budget.take(client))


budget_router = APIRouter(prefix="/api/seller", tags=["seller: caption approval"])


@budget_router.get("/llm-budget")
def llm_budget(budget: Budget, client: Client) -> dict[str, Any]:
    return budget.remaining(client)


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
    app.include_router(budget_router)
