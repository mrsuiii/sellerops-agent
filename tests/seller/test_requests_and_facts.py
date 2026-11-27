"""C3b backend: caption requests (caption from scratch) and product facts (the seller's answers, memory)."""

from __future__ import annotations

import sqlite3

import pytest

from sellerops.seller import SellerError


def test_request_lifecycle(service) -> None:
    request = service.request_caption("FS180", actor="seller:s1", fields=["description", "tags"],
                                      instruction="  fokus ke bahan ")
    assert request["request_id"] == "CR-0001" and request["status"] == "open"
    assert (request["fields"], request["instruction"], request["proposal_id"]) == (
        ["description", "tags"], "fokus ke bahan", None)
    assert service.list_requests(status="open") == [request]

    proposal = service.propose_caption("FS180", "first draft", actor="agent:caption-approval",
                                       description="Dress biru.", request_id="CR-0001",
                                       inputs={"seller_facts": {"material": "katun"}})
    assert proposal["request_id"] == "CR-0001" and proposal["inputs"]["seller_facts"] == {"material": "katun"}
    closed = service.get_request("CR-0001")
    assert (closed["status"], closed["proposal_id"], closed["closed_by"]) == (
        "proposed", proposal["proposal_id"], "agent:caption-approval")

    with pytest.raises(SellerError) as error:           # a request gets one first version
        service.propose_caption("FS180", "again", actor="agent", description="x", request_id="CR-0001")
    assert error.value.code == "invalid_transition"
    actions = [e["action"] for e in service.audit_log(product_id="FS180")]
    assert actions[:2] == ["proposed", "caption_requested"]


def test_request_rules(service) -> None:
    for bad in ({"fields": ["price"]}, {"fields": []}, {"instruction": "x" * 1001}):
        with pytest.raises(SellerError):
            service.request_caption("FS180", actor="seller:s1", **({"fields": ["description"]} | bad))
    with pytest.raises(SellerError) as error:
        service.request_caption("NOPE", actor="seller:s1")
    assert error.value.code == "product_not_found"

    service.request_caption("FS180", actor="seller:s1", fields=["description"])
    with pytest.raises(SellerError) as error:            # only the fields the seller asked for
        service.propose_caption("FS180", "r", actor="agent", name="New name", request_id="CR-0001")
    assert error.value.code == "field_not_allowed"
    with pytest.raises(SellerError) as error:            # same product
        service.propose_caption("FS181", "r", actor="agent", description="x", request_id="CR-0001")
    assert error.value.code == "invalid_request"

    cancelled = service.cancel_request("CR-0001", actor="seller:s1")
    assert cancelled["status"] == "cancelled"
    with pytest.raises(SellerError) as error:
        service.propose_caption("FS180", "r", actor="agent", description="x", request_id="CR-0001")
    assert error.value.code == "invalid_transition"
    with pytest.raises(SellerError) as error:
        service.get_request("CP-0001")
    assert error.value.code == "request_not_found"


def test_facts_are_append_only_and_latest_wins(service) -> None:
    assert service.product_facts("FS180") == {}
    facts = service.remember_facts("FS180", {"material": " katun ", "fit": ""}, actor="seller:s1", source="CR-0001")
    assert facts["material"]["value"] == "katun" and facts["fit"]["value"] is None   # "" = does not know
    assert facts["material"]["source"] == "CR-0001" and facts["material"]["by"] == "seller:s1"

    service.remember_facts("FS180", {"fit": "regular fit"}, actor="seller:s2")
    facts = service.product_facts("FS180")
    assert facts["fit"] == {"value": "regular fit", "source": None, "at": facts["fit"]["at"], "by": "seller:s2"}
    assert facts["material"]["value"] == "katun"
    assert service.product_facts("FS181") == {}

    for bad in ({}, {"Material": "x"}, {"a" * 41: "x"}, {"care": "x" * 301}):
        with pytest.raises(SellerError) as error:
            service.remember_facts("FS180", bad, actor="seller:s1")
        assert error.value.code == "invalid_facts"

    with sqlite3.connect(service.db_path) as conn, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE product_facts SET value = 'linen'")
    assert service.audit_log(product_id="FS180")[0]["action"] == "facts_recorded"


def test_seller_api_requests_and_facts(service) -> None:
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from sellerops.seller import api as seller_api
    from sellerops.store.api import app

    app.dependency_overrides[seller_api.get_service] = lambda: service
    try:
        client = TestClient(app)
        headers = {"X-Seller-User": "s1"}
        saved = client.post("/api/seller/products/FS180/facts", headers=headers,
                            json={"facts": {"material": "katun", "care": None}})
        assert saved.status_code == 200 and saved.json()["facts"]["care"]["value"] is None
        assert client.get("/api/seller/products/FS180/facts").json()["facts"]["material"]["by"] == "seller:s1"
        assert client.post("/api/seller/products/FS180/facts", json={"facts": {"a": "b"}}).status_code == 422

        service.request_caption("FS180", actor="seller:s1")
        assert [r["request_id"] for r in client.get("/api/seller/caption-requests?status=open").json()["requests"]] \
            == ["CR-0001"]
        assert client.post("/api/seller/caption-requests/CR-0001/cancel", headers=headers).json()["status"] \
            == "cancelled"
        missing = client.get("/api/seller/caption-requests/CR-0099")
        assert missing.status_code == 404 and missing.json()["code"] == "request_not_found"
    finally:
        app.dependency_overrides.clear()


def test_demo_seed_only_fills_an_untouched_store(service) -> None:
    from sellerops.seller.demo import seed_demo

    assert seed_demo(service) == 2
    pending = service.list_proposals(status="pending")
    assert {p["product_id"] for p in pending} == {"FS180", "FS083"} and all(p["created_by"] == "agent" for p in pending)
    assert seed_demo(service) == 0                 # never twice, never on a store with work in it
