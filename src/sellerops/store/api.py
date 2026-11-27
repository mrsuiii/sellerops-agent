"""FastAPI backend for the Kala Studio demo store.

Run from the project directory:

    make store            # or: uvicorn sellerops.store.api:app --reload

Then open http://127.0.0.1:8000 for the shop and http://127.0.0.1:8000/docs
for the API. The seller backend (analytics + caption approval, ``/api/seller``)
is mounted from ``sellerops.seller.api``; its SQLite database is created and
seeded from the demo data on first start (``SELLEROPS_SELLER_DB``). The frontend in ``web/`` is plain HTML/JS served by this app.
Configuration comes from ``SELLEROPS_*`` environment variables
(see ``sellerops.settings``).
"""

from __future__ import annotations

import json
import logging
import threading
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
from ..seller import SellerService
from ..seller import api as seller_api
from ..settings import get_settings
from .catalog import Catalog, ProductFilter, SortKey, load_catalog, summary

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
    # Opens (and on first start seeds) the seller database.
    seller = seller_api.default_services().for_store(seller_api.current_store_id())
    if get_settings().demo_mode:
        from ..seller.demo import seed_demo

        seed_demo(seller)
    logger.info("store started", extra={"products": len(catalog.products), "version": APP_VERSION,
                                        "seller_db": str(seller.db_path),
                                        "store_id": seller.store_id})
    yield


settings = get_settings()
configure_logging()

app = FastAPI(title="Kala Studio API", version=APP_VERSION, lifespan=lifespan)
seller_api.install(app)
try:  # the approval workflow needs the agent runtime ([agent] extra); the store image has only [store]
    from ..workflows.approval import api as approval_api
except ImportError as error:
    logger.info("caption approval workflow not mounted", extra={"missing": error.name})
else:
    approval_api.install(app)

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


_shop: dict[str, Any] = {"key": None, "base": None, "catalog": None}
_shop_lock = threading.Lock()


def shop_catalog(seller: SellerService) -> Catalog:
    """The catalog the shop shows: the snapshot with each product's current caption from the seller DB.

    An approved caption (or a revert) changes the seller database; the shop
    reads it from there, so it shows up on the next request. The search index
    is rebuilt only when a caption changed.
    """
    base = load_catalog()
    captions = seller.captions()
    key = json.dumps(captions, sort_keys=True)
    with _shop_lock:
        if _shop["key"] != key or _shop["base"] is not base:
            products = [{**p, **captions.get(p["product_id"], {})} for p in base.products]
            _shop.update(key=key, base=base, catalog=Catalog({**base.meta, "products": products}))
        return _shop["catalog"]


@app.get("/api/health")
def health() -> dict[str, Any]:
    catalog = load_catalog()
    return {"status": "ok", "version": APP_VERSION, "products": len(catalog.products), **catalog.meta}


@app.get("/api/categories")
def categories(seller: seller_api.Service) -> dict[str, Any]:
    catalog = shop_catalog(seller)
    return {"categories": catalog.categories(), "brands": catalog.brands()}


@app.get("/api/products")
def list_products(
    seller: seller_api.Service,
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
    rows = shop_catalog(seller).search(q, filters, sort)
    start = (page - 1) * page_size
    return {
        "total": len(rows),
        "page": page,
        "page_size": page_size,
        "items": [summary(p) for p in rows[start : start + page_size]],
    }


@app.get("/api/products/{product_id}")
def get_product(seller: seller_api.Service, product_id: str) -> dict[str, Any]:
    product = shop_catalog(seller).get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Unknown product_id {product_id}")
    return product


@app.post("/api/recommend")
def recommend(seller: seller_api.Service, body: RecommendRequest) -> dict[str, Any]:
    filters = ProductFilter(
        category=body.category, audience=body.audience, brand=body.brand,
        min_price=body.min_price, max_price=body.max_price, min_rating=body.min_rating,
        in_stock=body.in_stock, exclude_ids=frozenset(body.exclude_ids),
    )
    items = shop_catalog(seller).recommend(body.query, filters, body.limit, body.similar_to)
    return {"items": items}


if settings.web_dir.is_dir():
    app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.web_dir / "index.html")

    @app.get("/seller", include_in_schema=False)
    def seller_page() -> FileResponse:
        return FileResponse(settings.web_dir / "seller.html")
else:
    logger.warning("web dir not found; serving API only", extra={"web_dir": str(settings.web_dir)})
