"""Store seam: a service is bound to one store, and store ids resolve in one place."""

from __future__ import annotations

import pytest

from sellerops.seller import DEFAULT_STORE_ID, SellerError, SellerService, SingleStoreServices


def test_service_is_bound_to_one_store(service) -> None:
    assert service.store_id == DEFAULT_STORE_ID
    assert SellerService(service.db_path).store_id == DEFAULT_STORE_ID  # reopening is fine
    with pytest.raises(RuntimeError, match="belongs to store"):
        SellerService(service.db_path, store_id="other-store")


def test_registry_resolves_only_known_stores(service) -> None:
    services = SingleStoreServices(service)
    assert services.for_store(DEFAULT_STORE_ID) is service
    with pytest.raises(SellerError) as error:
        services.for_store("other-store")
    assert error.value.code == "store_not_found"


def test_api_resolves_the_store_through_one_dependency(service) -> None:
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from sellerops.seller import api as seller_api
    from sellerops.store.api import app

    app.dependency_overrides[seller_api.default_services] = lambda: SingleStoreServices(service)
    try:
        client = TestClient(app)
        created = client.post("/api/seller/caption-proposals", headers={"X-Seller-User": "seller-1"},
                              json={"product_id": "FS180", "reason": "r", "description": "Clearer."})
        assert service.get_proposal(created.json()["proposal_id"])["status"] == "pending"  # the override's db
        app.dependency_overrides[seller_api.current_store_id] = lambda: "other-store"
        response = client.get("/api/seller/products/FS180")
        assert (response.status_code, response.json()["code"]) == (404, "store_not_found")
    finally:
        app.dependency_overrides.clear()
