"""Seller simulator dataset: determinism, manifest, stock consistency and planted patterns."""

from __future__ import annotations

import json
from datetime import date

import pytest

from sellerops.settings import get_settings
from sellerops.simulator.generator import AS_OF, DAYS, SCENARIOS, START, dataset_files, generate
from sellerops.simulator.store import SellerStore, default_sim_dir, load_sim_data


@pytest.fixture(scope="module")
def catalog():
    return json.loads(get_settings().catalog_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store():
    return SellerStore(load_sim_data())


def test_generation_is_deterministic_and_matches_committed_files(catalog) -> None:
    files = dataset_files(catalog)
    assert files == dataset_files(catalog)
    for name, text in files.items():
        assert (default_sim_dir() / name).read_text(encoding="utf-8") == text, f"{name} is stale; regenerate"


def test_other_seed_gives_other_orders(catalog) -> None:
    assert generate(catalog, seed=1)["orders"] != generate(catalog, seed=2)["orders"]


def test_orders_cover_the_window_and_totals_add_up(store) -> None:
    orders = store.data.orders
    assert min(o["date"] for o in orders) == START.isoformat()
    assert max(o["date"] for o in orders) <= AS_OF.isoformat()
    assert (AS_OF - START).days == DAYS - 1
    for order in orders:
        assert order["total_idr"] == sum(i["quantity"] * i["unit_price_idr"] for i in order["items"])
        assert len({i["product_id"] for i in order["items"]}) == len(order["items"])


def test_stock_is_consistent_with_sales(store) -> None:
    sold: dict[str, int] = {}
    for order in store.data.orders:
        for item in order["items"]:
            sold[item["product_id"]] = sold.get(item["product_id"], 0) + item["quantity"]
    for pid, row in store.data.inventory.items():
        assert row["units_sold"] == sold.get(pid, 0)
        assert row["initial_stock"] - row["units_sold"] == row["stock"] >= 0


def test_spike(store) -> None:
    pid = SCENARIOS["spike"]["product_id"]
    now, before = store.units_by_product("last_7_days"), store.units_by_product("previous_7_days")
    assert now[pid] >= 2.5 * before[pid]
    assert max(now, key=lambda p: now[p] - before[p]) == pid


def test_stockout_drives_the_category_drop(store) -> None:
    pid = SCENARIOS["stockout"]["product_id"]
    days = SCENARIOS["stockout"]["out_of_stock_last_days"]
    last_sale = max(o["date"] for o in store.data.orders if any(i["product_id"] == pid for i in o["items"]))
    assert (AS_OF - date.fromisoformat(last_sale)).days >= days
    assert store.data.inventory[pid]["stock"] == 0

    def revenue(period: str) -> int:
        rows = store.product_performance(period, "revenue_desc", 54, "shoes", "women")["products"]
        return sum(r["revenue_idr"] for r in rows)

    assert revenue("last_30_days") < 0.7 * revenue("previous_30_days")


def test_low_stock_and_slow_movers(store) -> None:
    rows = {r["product_id"]: r for r in store.inventory_rows()}
    low = rows[SCENARIOS["low_stock"]["product_id"]]
    assert low["stock"] == SCENARIOS["low_stock"]["end_stock"]
    assert low["days_to_stockout"] is not None and low["days_to_stockout"] < 3
    units = store.units_by_product("last_30_days")
    for pid in SCENARIOS["slow_movers"]["product_ids"]:
        assert units[pid] == 0 and store.data.inventory[pid]["stock"] > 0


def test_recurring_complaint_is_only_on_the_planted_product(store) -> None:
    planted = SCENARIOS["negative_reviews"]
    complaints = [r for r in store.data.reviews if "kecil" in r["comment"].lower() or "sempit" in r["comment"].lower()]
    assert {r["product_id"] for r in complaints} == {planted["product_id"]}
    assert len(complaints) == planted["count"]
    assert all(r["rating"] <= 2 for r in complaints)
