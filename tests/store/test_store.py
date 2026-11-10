from __future__ import annotations

import pytest

from sellerops.store.catalog import BM25, ProductFilter, load_catalog, tokenize


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_snapshot_is_fashion_only_and_well_formed(catalog):
    assert len(catalog.products) == 54
    ids = [p["product_id"] for p in catalog.products]
    assert len(ids) == len(set(ids))
    for p in catalog.products:
        assert p["audience"] in {"men", "women", "unisex"}
        assert 0 < p["final_price_idr"] <= p["price_idr"]
        assert all("email" not in key for review in p["reviews"] for key in review)


def test_tokenize_strips_simple_plurals():
    assert tokenize("Running Shoes, Glasses!") == ["running", "shoe", "glass"]


def test_bm25_prefers_matching_document():
    bm25 = BM25(["red leather bag", "blue cotton shirt"])
    assert bm25.score("leather", 0) > 0
    assert bm25.score("leather", 1) == 0


def test_search_filters_and_sorts(catalog):
    rows = catalog.search("", ProductFilter(category="shoes", audience="women", max_price=2_000_000), "price_asc")
    assert rows
    assert all(p["category"] == "shoes" and p["audience"] in {"women", "unisex"} for p in rows)
    prices = [p["final_price_idr"] for p in rows]
    assert prices == sorted(prices) and prices[-1] <= 2_000_000


def test_search_keyword_only_returns_matches(catalog):
    rows = catalog.search("watch")
    assert rows and all(p["category"] == "watches" for p in rows)


def test_recommend_respects_hard_filters_and_explains(catalog):
    items = catalog.recommend("dress", ProductFilter(in_stock=True, max_price=1_500_000), limit=3)
    assert 0 < len(items) <= 3
    for item in items:
        assert item["stock"] > 0 and item["final_price_idr"] <= 1_500_000
        assert isinstance(item["reasons"], list)
    assert [i["score"] for i in items] == sorted((i["score"] for i in items), reverse=True)


def test_recommend_similar_excludes_anchor(catalog):
    anchor = catalog.search("", ProductFilter(category="sunglasses"))[0]
    items = catalog.recommend(similar_to=anchor["product_id"], limit=4)
    assert items and anchor["product_id"] not in {i["product_id"] for i in items}
    assert items[0]["category"] == "sunglasses"


def test_recommend_is_deterministic(catalog):
    first = catalog.recommend("black shirt", limit=5)
    assert first == catalog.recommend("black shirt", limit=5)


def test_api_endpoints():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from sellerops.store.api import app

    client = TestClient(app)
    assert client.get("/api/health").json()["products"] == 54
    page = client.get("/api/products", params={"category": "bags", "page_size": 2}).json()
    assert page["total"] == 5 and len(page["items"]) == 2
    pid = page["items"][0]["product_id"]
    assert client.get(f"/api/products/{pid}").json()["product_id"] == pid
    assert client.get("/api/products/NOPE").status_code == 404
    rec = client.post("/api/recommend", json={"query": "gold watch", "audience": "women", "limit": 2}).json()
    assert len(rec["items"]) <= 2
    assert client.get("/").status_code == 200


def test_api_request_id_and_internal_errors_are_hidden(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from sellerops.store import api

    client = TestClient(api.app, raise_server_exceptions=False)
    response = client.get("/api/health", headers={"x-request-id": "req-1"})
    assert response.headers["x-request-id"] == "req-1"
    assert response.json()["version"]

    def boom(*_args, **_kwargs):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(api, "load_catalog", boom)
    response = client.get("/api/categories")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert "secret" not in response.text
    assert response.headers["x-request-id"] == body["request_id"]
