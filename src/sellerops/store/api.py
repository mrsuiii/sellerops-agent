"""FastAPI backend for the Kala Studio demo store.

Run from the project directory:

    make store            # or: uvicorn sellerops.store.api:app --reload

Then open http://127.0.0.1:8000 for the shop and http://127.0.0.1:8000/docs
for the API. The frontend in ``web/`` is plain HTML/JS served by this app.
Configuration comes from ``SELLEROPS_*`` environment variables
(see ``sellerops.settings``).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..logging_config import configure_logging
from ..settings import get_settings
from .catalog import ProductFilter, SortKey, load_catalog, summary

logger = logging.getLogger(__name__)

try:
    APP_VERSION = package_version("sellerops-agent")
except PackageNotFoundError:  # running from a source tree without installing
    APP_VERSION = "0.0.0+local"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load (and validate) the catalog at startup so a bad path fails fast,
    # not on the first customer request.
    catalog = load_catalog()
    logger.info("store started", extra={"products": len(catalog.products), "version": APP_VERSION})
    yield


settings = get_settings()
configure_logging()

app = FastAPI(title="Kala Studio API", version=APP_VERSION, lifespan=lifespan)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id, log one line per request, and hide internal errors."""

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["x-request-id"] = request_id
    logger.info(
        "%s %s %s",
        request.method,
        request.url.path,
        response.status_code,
        extra={"request_id": request_id, "status": response.status_code, "duration_ms": duration_ms},
    )
    return response


class RecommendRequest(BaseModel):
    """Body for ``POST /api/recommend``; mirrors the agent tool's arguments."""

    query: str = ""
    category: str | None = None
    audience: str | None = Field(None, pattern="^(men|women|unisex)$")
    brand: str | None = None
    min_price: int | None = Field(None, ge=0)
    max_price: int | None = Field(None, ge=0)
    min_rating: float | None = Field(None, ge=0, le=5)
    in_stock: bool = True
    similar_to: str | None = None
    exclude_ids: list[str] = []
    limit: int = Field(5, ge=1, le=20)


@app.get("/api/health")
def health() -> dict[str, Any]:
    catalog = load_catalog()
    return {"status": "ok", "version": APP_VERSION, "products": len(catalog.products), **catalog.meta}


@app.get("/api/categories")
def categories() -> dict[str, Any]:
    catalog = load_catalog()
    return {"categories": catalog.categories(), "brands": catalog.brands()}


@app.get("/api/products")
def list_products(
    q: str = "",
    category: str | None = None,
    audience: str | None = Query(None, pattern="^(men|women|unisex)$"),
    brand: str | None = None,
    min_price: int | None = Query(None, ge=0),
    max_price: int | None = Query(None, ge=0),
    min_rating: float | None = Query(None, ge=0, le=5),
    in_stock: bool = False,
    sort: SortKey = "relevance",
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=60),
) -> dict[str, Any]:
    filters = ProductFilter(
        category=category, audience=audience, brand=brand, min_price=min_price,
        max_price=max_price, min_rating=min_rating, in_stock=in_stock,
    )
    rows = load_catalog().search(q, filters, sort)
    start = (page - 1) * page_size
    return {
        "total": len(rows),
        "page": page,
        "page_size": page_size,
        "items": [summary(p) for p in rows[start : start + page_size]],
    }


@app.get("/api/products/{product_id}")
def get_product(product_id: str) -> dict[str, Any]:
    product = load_catalog().get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Unknown product_id {product_id}")
    return product


@app.post("/api/recommend")
def recommend(body: RecommendRequest) -> dict[str, Any]:
    filters = ProductFilter(
        category=body.category, audience=body.audience, brand=body.brand,
        min_price=body.min_price, max_price=body.max_price, min_rating=body.min_rating,
        in_stock=body.in_stock, exclude_ids=frozenset(body.exclude_ids),
    )
    items = load_catalog().recommend(body.query, filters, body.limit, body.similar_to)
    return {"items": items}


if settings.web_dir.is_dir():
    app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.web_dir / "index.html")
else:
    logger.warning("web dir not found; serving API only", extra={"web_dir": str(settings.web_dir)})
