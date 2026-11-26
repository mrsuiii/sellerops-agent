"""SellerStore queries, tool errors, caption proposals and episode isolation."""

from __future__ import annotations

import math

import pytest

from sellerops.simulator import SellerEpisode, ToolError, load_store


@pytest.fixture
def store():
    return load_store()


def test_sales_summary_groups_add_up(store) -> None:
    total = store.sales_summary("last_30_days")
    by_category = store.sales_summary("last_30_days", "category")
    assert sum(g["revenue_idr"] for g in by_category["groups"]) == total["revenue_idr"]
    assert sum(g["units"] for g in by_category["groups"]) == total["units"]
    weeks = store.sales_summary("last_90_days", "week")["groups"]
    assert [g["key"] for g in weeks] == sorted(g["key"] for g in weeks)
    assert total["period"] == {"name": "last_30_days", "start": "2026-10-17", "end": "2026-11-15"}


def test_periods_do_not_overlap(store) -> None:
    last, previous = store.sales_summary("last_7_days"), store.sales_summary("previous_7_days")
    assert previous["period"]["end"] < last["period"]["start"]
    assert store.sales_summary("last_90_days")["orders"] == len(store.data.orders)


def test_product_performance_includes_zero_sales(store) -> None:
    rows = store.product_performance("last_30_days", "units_asc", 54)["products"]
    assert len(rows) == 54 and rows[0]["units"] == 0
    top = store.product_performance("last_30_days", "revenue_desc", 3, category="bags")["products"]
    assert [r["category"] for r in top] == ["bags"] * 3
    assert top[0]["revenue_idr"] >= top[1]["revenue_idr"] >= top[2]["revenue_idr"]


def test_inventory_status_formula(store) -> None:
    rows = store.inventory_status(limit=54, target_days=14)["products"]
    units = store.units_by_product("last_30_days")
    for row in rows:
        velocity = units[row["product_id"]] / 30
        if velocity == 0:
            assert row["days_to_stockout"] is None and row["suggested_restock_units"] == 0
        else:
            assert row["days_to_stockout"] == round(row["stock"] / velocity, 1)
            assert row["suggested_restock_units"] == max(0, math.ceil(velocity * 14) - row["stock"])
    known = [r["days_to_stockout"] for r in rows if r["days_to_stockout"] is not None]
    assert known == sorted(known)


def test_reviews_filters(store) -> None:
    result = store.reviews(category="shoes", max_rating=2, period="last_30_days")
    assert result["summary"]["count"] == len(result["reviews"])
    assert all(r["rating"] <= 2 for r in result["reviews"])
    dates = [r["date"] for r in result["reviews"]]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.parametrize(("call", "code"), [
    (lambda s: s.product("NOPE"), "product_not_found"),
    (lambda s: s.sales_summary("yesterday"), "invalid_period"),
    (lambda s: s.inventory_status(category="hats"), "unknown_category"),
    (lambda s: s.propose_caption_update("FS180", reason="x", description="a" * 601), "field_too_long"),
    (lambda s: s.propose_caption_update("FS180", reason="x", tags=[f"t{i}" for i in range(9)]), "invalid_tags"),
    (lambda s: s.propose_caption_update("FS180", reason="x"), "no_changes"),
])
def test_tool_errors(store, call, code) -> None:
    with pytest.raises(ToolError) as info:
        call(store)
    assert info.value.code == code


def test_caption_proposal_never_changes_the_product(store) -> None:
    before = store.product("FS180")
    proposal = store.propose_caption_update("FS180", reason="Lebih jelas", description="Dress motif polkadot.")
    assert proposal["status"] == "pending"
    assert set(proposal["changes"]) == {"description"}
    assert proposal["changes"]["description"]["old"] == before["description"]
    assert store.product("FS180") == before


def test_episodes_are_isolated() -> None:
    a, b = SellerEpisode(), SellerEpisode()
    a.call("propose_caption_update", {"product_id": "FS180", "reason": "x", "description": "Baru."})
    assert len(a.snapshot()["proposals"]) == 1
    assert b.snapshot()["proposals"] == [] and b.snapshot()["history"] == []


def test_episode_records_errors_as_observations() -> None:
    episode = SellerEpisode()
    observation = episode.call("get_product", {"product_id": "NOPE"})
    assert observation == {"ok": False, "error": {"code": "product_not_found", "message": "no product with id 'NOPE'"}}
    assert episode.snapshot()["history"][0]["tool"] == "get_product"
    with pytest.raises(KeyError):
        episode.call("delete_everything", {})


def test_compare_to_previous_adds_changes_computed_from_both_windows() -> None:
    store = load_store()
    summary = store.sales_summary("last_7_days", compare_to_previous=True)
    before = store.sales_summary("previous_7_days")
    assert summary["compared_with"]["name"] == "previous_7_days"
    assert summary["previous_revenue_idr"] == before["revenue_idr"]
    expected = round((summary["revenue_idr"] - before["revenue_idr"]) / before["revenue_idr"] * 100, 1)
    assert summary["revenue_change_pct"] == expected
    assert summary["trend"] == ("down" if expected < 0 else "up")

    rows = store.product_performance("last_30_days", category="shoes", audience="women", limit=54,
                                     compare_to_previous=True)["products"]
    previous = store.units_by_product("previous_30_days")
    for row in rows:
        assert row["previous_units"] == previous[row["product_id"]]
        assert row["units_change"] == row["units"] - row["previous_units"]
    stockout = next(r for r in rows if r["product_id"] == "FS186")  # planted stockout scenario
    assert stockout["trend"] == "down" and stockout["stock"] == 0


def test_compare_to_previous_rejects_windows_without_a_previous_one() -> None:
    import pytest

    from sellerops.simulator import ToolError

    with pytest.raises(ToolError, match="compare_to_previous"):
        load_store().sales_summary("last_90_days", compare_to_previous=True)
