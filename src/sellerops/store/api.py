"""FastAPI backend for the Kala Studio demo store.

Run from the project directory:

    uvicorn sellerops.store.api:app --reload --app-dir src

Then open http://127.0.0.1:8000 for the shop and http://127.0.0.1:8000/docs
for the API. The frontend in ``web/`` is plain HTML/JS served by this app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import ProductFilter, SortKey, load_catalog, summary


WEB_DIR = Path(__file__).resolve().parents[3] / "web"

app = FastAPI(title="Kala Studio API", version="0.1.0")


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
    return {"status": "ok", "products": len(catalog.products), **catalog.meta}


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


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
