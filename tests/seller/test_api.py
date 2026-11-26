"""Seller HTTP API: routes, actor header, error codes → HTTP status."""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from sellerops.seller import api as seller_api
from sellerops.store.api import app

HEADERS = {"X-Seller-User": "seller-1"}


@pytest.fixture
def client(service):
    app.dependency_overrides[seller_api.get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_analytics_routes(client, service) -> None:
    body = client.get("/api/seller/sales-summary", params={"period": "last_7_days", "compare_to_previous": True}).json()
    assert body == service.analytics().sales_summary("last_7_days", compare_to_previous=True)
    rows = client.get("/api/seller/product-performance", params={"period": "last_30_days", "limit": 3}).json()
    assert len(rows["products"]) == 3
    assert client.get("/api/seller/inventory", params={"category": "dresses"}).json()["total_matching"] == 5
    assert client.get("/api/seller/reviews", params={"max_rating": 2}).json()["summary"]["count"] > 0
    assert client.get("/api/seller/products/FS180").json()["caption_version"]


def test_caption_flow_over_http(client) -> None:
    created = client.post("/api/seller/caption-proposals", headers=HEADERS,
                          json={"product_id": "FS180", "reason": "clearer", "description": "Clearer."})
    assert created.status_code == 201
    pid = created.json()["proposal_id"]
    assert created.json()["created_by"] == "seller:seller-1"
    assert client.get("/api/seller/caption-proposals", params={"status": "pending"}).json()["proposals"][0][
        "proposal_id"] == pid

    approved = client.post(f"/api/seller/caption-proposals/{pid}/approve", headers=HEADERS, json={"note": "ok"})
    assert approved.status_code == 200 and approved.json()["status"] == "applied"
    assert client.get("/api/seller/products/FS180").json()["description"] == "Clearer."

    again = client.post(f"/api/seller/caption-proposals/{pid}/approve", headers=HEADERS)
    assert again.status_code == 409 and again.json()["code"] == "invalid_transition"

    reverted = client.post(f"/api/seller/caption-proposals/{pid}/revert", headers=HEADERS)
    assert reverted.json()["status"] == "reverted"
    versions = client.get("/api/seller/products/FS180/caption-versions").json()["versions"]
    assert [v["source"] for v in versions] == ["seed", "proposal", "revert"]
    actions = [e["action"] for e in client.get("/api/seller/audit-log", params={"proposal_id": pid}).json()["entries"]]
    assert actions == ["reverted", "approved", "proposed"]


def test_reject_and_revise_over_http(client) -> None:
    pid = client.post("/api/seller/caption-proposals", headers=HEADERS,
                      json={"product_id": "FS180", "reason": "r", "name": "Long Name"}).json()["proposal_id"]
    rejected = client.post(f"/api/seller/caption-proposals/{pid}/reject", headers=HEADERS,
                           json={"feedback": "keep the name"})
    assert rejected.json()["feedback"] == "keep the name"
    revision = client.post("/api/seller/caption-proposals", headers=HEADERS,
                           json={"product_id": "FS180", "reason": "r", "tags": ["dress"], "revision_of": pid})
    assert revision.json()["revision_of"] == pid


def test_cancel_over_http(client) -> None:
    pid = client.post("/api/seller/caption-proposals", headers=HEADERS,
                      json={"product_id": "FS180", "reason": "r", "name": "Other"}).json()["proposal_id"]
    assert client.post(f"/api/seller/caption-proposals/{pid}/cancel", headers=HEADERS).json()["status"] == "cancelled"


@pytest.mark.parametrize(("method", "path", "kwargs", "status", "code"), [
    ("get", "/api/seller/products/NOPE", {}, 404, "product_not_found"),
    ("get", "/api/seller/caption-proposals/CP-0404", {}, 404, "proposal_not_found"),
    ("post", "/api/seller/caption-proposals/CP-0404/approve", {"headers": HEADERS}, 404, "proposal_not_found"),
    ("get", "/api/seller/inventory", {"params": {"category": "hats"}}, 422, "unknown_category"),
    ("get", "/api/seller/sales-summary", {"params": {"period": "last_90_days", "compare_to_previous": True}},
     422, "invalid_compare"),
    ("post", "/api/seller/caption-proposals",
     {"headers": HEADERS, "json": {"product_id": "FS180", "reason": "r", "name": "Dress Pea"}}, 422, "no_changes"),
])
def test_business_errors_keep_their_code(client, method, path, kwargs, status, code) -> None:
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == status and response.json()["code"] == code


@pytest.mark.parametrize("headers", [{}, {"X-Seller-User": "bad user!"}])
def test_writes_need_a_valid_actor_header(client, headers) -> None:
    response = client.post("/api/seller/caption-proposals", headers=headers,
                           json={"product_id": "FS180", "reason": "r", "name": "New"})
    assert response.status_code == 422


def test_request_bodies_reject_unknown_fields(client) -> None:
    response = client.post("/api/seller/caption-proposals", headers=HEADERS,
                           json={"product_id": "FS180", "reason": "r", "name": "New", "status": "applied"})
    assert response.status_code == 422
