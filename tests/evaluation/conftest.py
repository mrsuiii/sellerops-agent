"""Fixtures for seller eval tests: tasks and a reference solver (no model, no network)."""

from __future__ import annotations

from typing import Any

import pytest

from sellerops.evaluation.seller_tasks import load_tasks
from sellerops.simulator import SellerEpisode


def reference_final(task: dict[str, Any], episode: SellerEpisode) -> dict[str, Any]:
    """Solve a task with the tools (as a correct agent would) and return its ``final`` action."""
    e = task["expected"]

    def call(tool: str, **arguments: Any) -> dict[str, Any]:
        observation = episode.call(tool, {k: v for k, v in arguments.items() if v is not None})
        assert observation["ok"], observation
        return observation["result"]

    family = task["family"]
    if family == "sales_summary":
        r = call("get_sales_summary", period=e["period"])
        data = {"period": e["period"], "revenue_idr": r["revenue_idr"], "orders": r["orders"]}
    elif family == "top_category":
        r = call("get_sales_summary", period=e["period"], group_by="category")
        data = {"period": e["period"], "category": r["groups"][0]["key"]}
    elif family == "top_products":
        r = call("get_product_performance", period=e["period"], sort="units_desc",
                 limit=len(e["product_ids"]), category=e["category"])
        data = {"period": e["period"], "product_ids": [p["product_id"] for p in r["products"]]}
    elif family == "slow_movers":
        r = call("get_product_performance", period=e["period"], sort="units_asc", limit=54, category=e["category"])
        data = {"period": e["period"], "product_ids": [p["product_id"] for p in r["products"] if p["units"] == 0]}
    elif family == "restock":
        r = call("get_inventory_status", category=e["category"], limit=54)
        data = {"product_ids": [p["product_id"] for p in r["products"]
                                if p["days_to_stockout"] is not None and p["days_to_stockout"] <= 7]}
    elif family == "review_insight":
        r = call("get_reviews", category=e["category"], max_rating=2, period=e["period"], limit=50)
        counts: dict[str, int] = {}
        for review in r["reviews"]:
            counts[review["product_id"]] = counts.get(review["product_id"], 0) + 1
        data = {"product_id": max(sorted(counts), key=counts.get), "complaint": "Ukurannya kekecilan."}
    elif family == "sales_spike":
        now = call("get_product_performance", period="last_7_days", limit=54)["products"]
        before = {p["product_id"]: p["units"] for p in
                  call("get_product_performance", period="previous_7_days", limit=54)["products"]}
        best = max(sorted(now, key=lambda p: p["product_id"]), key=lambda p: p["units"] - before[p["product_id"]])
        data = {"period": "last_7_days", "product_id": best["product_id"]}
    elif family == "sales_drop":
        now = call("get_product_performance", period="last_30_days", sort="revenue_desc", limit=54,
                   category=e["category"], audience=e["audience"])["products"]
        before = {p["product_id"]: p["revenue_idr"] for p in call(
            "get_product_performance", period="previous_30_days", sort="revenue_desc", limit=54,
            category=e["category"], audience=e["audience"])["products"]}
        worst = min(now, key=lambda p: p["revenue_idr"] - before[p["product_id"]])
        data = {"product_id": worst["product_id"], "cause": "out_of_stock" if worst["stock"] == 0 else "other"}
    elif family == "caption":
        product = call("get_product", product_id=e["product_id"])
        changes = {"description": product["description"] + " Cocok untuk dipakai sehari-hari."}
        if "tags" in e["required_fields"]:
            changes["tags"] = [*product["tags"], "kala-studio"]
        r = call("propose_caption_update", product_id=e["product_id"], reason="Lebih jelas.", **changes)
        data = {"proposal_id": r["proposal_id"], "product_id": e["product_id"]}
    else:
        raise AssertionError(family)
    return {"type": "final", "message": "Jawaban.", "data": data}


@pytest.fixture(scope="session")
def all_tasks() -> list[dict[str, Any]]:
    return load_tasks("dev") + load_tasks("test")


@pytest.fixture(scope="session")
def dev_tasks() -> dict[str, dict[str, Any]]:
    return {t["task_id"]: t for t in load_tasks("dev")}


@pytest.fixture
def solve():
    return reference_final
