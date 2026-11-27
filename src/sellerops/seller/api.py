"""Seller HTTP API under ``/api/seller``: analytics for the seller UI and the caption approval flow.

    GET  /api/seller/sales-summary            ?period&group_by&compare_to_previous
    GET  /api/seller/product-performance      ?period&sort&limit&category&audience&compare_to_previous
    GET  /api/seller/inventory                ?category&target_days&limit
    GET  /api/seller/reviews                  ?product_id&category&max_rating&period&limit
    GET  /api/seller/products/{id}
    GET  /api/seller/products/{id}/caption-versions
    GET  /api/seller/caption-proposals        ?status&product_id&limit
    POST /api/seller/caption-proposals        create a pending proposal
    GET  /api/seller/caption-proposals/{id}
    POST /api/seller/caption-proposals/{id}/approve
    POST /api/seller/caption-proposals/{id}/reject
    POST /api/seller/caption-proposals/{id}/cancel
    POST /api/seller/caption-proposals/{id}/revert
    GET  /api/seller/caption-requests         ?status&product_id&limit
    GET  /api/seller/caption-requests/{id}
    POST /api/seller/caption-requests/{id}/cancel
    GET  /api/seller/products/{id}/facts      what the seller told the agent (memory)
    POST /api/seller/products/{id}/facts      {"facts": {"material": "katun", "fit": ""}} ("" = unknown)
    GET  /api/seller/audit-log                ?product_id&proposal_id&limit
    GET  /api/seller/images/{sha256}          an image the model saw (proposal ``inputs.images``)

Creating a caption request (``POST /caption-requests``) and answering the
agent's questions run the graph, so they live with the approval workflow.

The approval workflow adds ``/caption-proposals/{id}/approval`` routes when
the agent runtime is installed (``sellerops.workflows.approval.api``); the
approve / reject / cancel routes above are the plain service calls without
the graph (no revision is drafted).

Writes need an ``X-Seller-User`` header; it is recorded as the actor
(``seller:<user>``) in the audit log. **There is no authentication yet**: the
header is trusted, so this API must not be exposed beyond a demo until a real
login sits in front of it.

Who is calling and for which store is decided in exactly two dependencies,
``current_store_id`` and ``seller_user`` (``actor`` builds on it). Today the store is fixed (one demo store)
and the user comes from the header; with real auth both are read from the
token there, and the routes stay as they are.

Business failures (``SellerError``) map to HTTP status by code and keep the
code in the body (``{"detail": ..., "code": ...}``) so a UI can react to,
e.g., ``stale_proposal`` without parsing the message.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from ..settings import get_settings
from .analytics import SellerError
from .images import ImageStore, LocalImageStore, is_sha256
from .registry import SellerServices, SingleStoreServices
from .service import DEFAULT_STORE_ID, REQUEST_STATUSES, STATUSES, SellerService

Period = Literal["last_7_days", "previous_7_days", "last_30_days", "previous_30_days", "last_90_days"]
Status = Literal[STATUSES]  # type: ignore[valid-type]
RequestStatus = Literal[REQUEST_STATUSES]  # type: ignore[valid-type]

ERROR_STATUS = {
    "store_not_found": 404,
    "product_not_found": 404,
    "proposal_not_found": 404,
    "request_not_found": 404,
    "image_not_found": 404,
    "invalid_transition": 409,
    "stale_proposal": 409,
    "newer_change_exists": 409,
    "invalid_revision": 409,
    "max_rejections": 409,
    # caption approval workflow (sellerops.workflows.approval.api)
    "approval_not_found": 404,
    "approval_closed": 409,
    "stale_decision": 409,
    "approval_not_failed": 409,
    "not_waiting_for_answer": 409,
    "not_waiting_for_decision": 409,
    "llm_budget_exhausted": 429,
}  # every other code is a bad request (422)

router = APIRouter(prefix="/api/seller", tags=["seller"])


@lru_cache(maxsize=1)
def default_services() -> SellerServices:
    settings = get_settings()
    return SingleStoreServices(SellerService.open(
        settings.seller_db_path, store_id=DEFAULT_STORE_ID, catalog_path=settings.catalog_path,
        sim_dir=settings.data_dir / "seller_sim"))


def current_store_id() -> str:
    """The store this request acts on. One demo store for now; with auth, read it from the token."""
    return DEFAULT_STORE_ID


def get_service(store_id: Annotated[str, Depends(current_store_id)],
                services: Annotated[SellerServices, Depends(default_services)]) -> SellerService:
    """FastAPI dependency; tests override it (or ``default_services``) with a temporary database."""
    return services.for_store(store_id)


Service = Annotated[SellerService, Depends(get_service)]


def seller_user(x_seller_user: Annotated[str, Header(min_length=1, max_length=64, pattern=r"^[\w.@-]+$")]) -> str:
    """The seller's user name. Trusted header for now; with auth, read it from the token."""
    return x_seller_user


def actor(user: Annotated[str, Depends(seller_user)]) -> str:
    """Who acts, as recorded in the audit log."""
    return f"seller:{user}"


Actor = Annotated[str, Depends(actor)]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposalIn(_Body):
    product_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    revision_of: str | None = Field(None, description="Id of the rejected proposal this one revises.")


class DecisionIn(_Body):
    note: str | None = Field(None, max_length=500)


class FactsIn(_Body):
    facts: dict[str, str | None] = Field(description='Fact key → value; "" or null records "unknown".')


class RejectIn(_Body):
    feedback: str = Field("", max_length=1000, description="What the seller wants changed (used for a revision).")


# ------------------------------------------------------------------ analytics


@router.get("/sales-summary")
def sales_summary(service: Service, period: Period,
                  group_by: Literal["category", "audience", "week"] | None = None,
                  compare_to_previous: bool = False) -> dict[str, Any]:
    return service.analytics().sales_summary(period, group_by, compare_to_previous)


@router.get("/product-performance")
def product_performance(service: Service, period: Period,
                        sort: Literal["units_desc", "units_asc", "revenue_desc", "revenue_asc"] = "units_desc",
                        limit: int = Query(5, ge=1, le=54), category: str | None = None,
                        audience: Literal["men", "women", "unisex"] | None = None,
                        compare_to_previous: bool = False) -> dict[str, Any]:
    return service.analytics().product_performance(period, sort, limit, category, audience, compare_to_previous)


@router.get("/inventory")
def inventory(service: Service, category: str | None = None, target_days: int = Query(14, ge=1, le=90),
              limit: int = Query(10, ge=1, le=54)) -> dict[str, Any]:
    return service.analytics().inventory_status(category, target_days, limit)


@router.get("/reviews")
def reviews(service: Service, product_id: str | None = None, category: str | None = None,
            max_rating: int | None = Query(None, ge=1, le=5), period: Period = "last_90_days",
            limit: int = Query(20, ge=1, le=50)) -> dict[str, Any]:
    return service.analytics().reviews(product_id, category, max_rating, period, limit)


@router.get("/products/{product_id}")
def product(service: Service, product_id: str) -> dict[str, Any]:
    return service.analytics().product(product_id)


@router.get("/products/{product_id}/caption-versions")
def caption_versions(service: Service, product_id: str) -> dict[str, Any]:
    return {"product_id": product_id, "versions": service.caption_history(product_id)}


# ------------------------------------------------------------------ captions


@router.get("/caption-proposals")
def list_proposals(service: Service, status: Status | None = None, product_id: str | None = None,
                   limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    return {"proposals": service.list_proposals(status=status, product_id=product_id, limit=limit)}


@router.post("/caption-proposals", status_code=201)
def create_proposal(service: Service, who: Actor, body: ProposalIn) -> dict[str, Any]:
    return service.propose_caption(body.product_id, body.reason, actor=who, name=body.name,
                                   description=body.description, tags=body.tags, revision_of=body.revision_of)


@router.get("/caption-proposals/{proposal_id}")
def get_proposal(service: Service, proposal_id: str) -> dict[str, Any]:
    return service.get_proposal(proposal_id)


@router.post("/caption-proposals/{proposal_id}/approve")
def approve(service: Service, who: Actor, proposal_id: str, body: DecisionIn | None = None) -> dict[str, Any]:
    return service.approve(proposal_id, actor=who, note=body.note if body else None)


@router.post("/caption-proposals/{proposal_id}/reject")
def reject(service: Service, who: Actor, proposal_id: str, body: RejectIn | None = None) -> dict[str, Any]:
    return service.reject(proposal_id, actor=who, feedback=body.feedback if body else "")


@router.post("/caption-proposals/{proposal_id}/cancel")
def cancel(service: Service, who: Actor, proposal_id: str, body: DecisionIn | None = None) -> dict[str, Any]:
    return service.cancel(proposal_id, actor=who, note=body.note if body else None)


@router.post("/caption-proposals/{proposal_id}/revert")
def revert(service: Service, who: Actor, proposal_id: str, body: DecisionIn | None = None) -> dict[str, Any]:
    return service.revert(proposal_id, actor=who, note=body.note if body else None)


@router.get("/caption-requests")
def list_requests(service: Service, status: RequestStatus | None = None, product_id: str | None = None,
                  limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    return {"requests": service.list_requests(status=status, product_id=product_id, limit=limit)}


@router.get("/caption-requests/{request_id}")
def get_request(service: Service, request_id: str) -> dict[str, Any]:
    return service.get_request(request_id)


@router.post("/caption-requests/{request_id}/cancel")
def cancel_request(service: Service, who: Actor, request_id: str) -> dict[str, Any]:
    return service.cancel_request(request_id, actor=who)


@router.get("/products/{product_id}/facts")
def product_facts(service: Service, product_id: str) -> dict[str, Any]:
    return {"product_id": product_id, "facts": service.product_facts(product_id)}


@router.post("/products/{product_id}/facts")
def remember_facts(service: Service, who: Actor, product_id: str, body: FactsIn) -> dict[str, Any]:
    return {"product_id": product_id, "facts": service.remember_facts(product_id, body.facts, actor=who)}


@router.get("/audit-log")
def audit_log(service: Service, product_id: str | None = None, proposal_id: str | None = None,
              limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    return {"entries": service.audit_log(product_id=product_id, proposal_id=proposal_id, limit=limit)}


# -------------------------------------------------------------------- images


def get_image_store(service: Service) -> ImageStore:
    """The image store of the request's store; tests override it."""
    return LocalImageStore(get_settings().images_dir, store_id=service.store_id)


@router.get("/images/{sha256}", response_class=Response)
def image(store: Annotated[ImageStore, Depends(get_image_store)], sha256: str) -> Response:
    data = store.get(sha256) if is_sha256(sha256) else None
    if data is None:
        raise SellerError("image_not_found", f"no image {sha256}")
    # Content-addressed: the bytes behind a hash never change.
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=31536000, immutable"})


# ---------------------------------------------------------------------- wiring


async def seller_error_handler(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, SellerError)
    return JSONResponse(status_code=ERROR_STATUS.get(error.code, 422),
                        content={"detail": error.message, "code": error.code})


def install(app: FastAPI) -> None:
    """Mount the seller routes and their error mapping on ``app``."""
    app.include_router(router)
    app.add_exception_handler(SellerError, seller_error_handler)
