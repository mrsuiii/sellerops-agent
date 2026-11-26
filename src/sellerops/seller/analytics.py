"""Seller analytics: the numbers behind the seller tools, shared by production and eval.

``SellerAnalytics`` is a pure read model over plain rows (products, orders,
reviews, stock). It does not know where the rows come from:

    SellerService (production)   rows from SQLite (``seller/db.py``)
    SellerStore   (simulator)    rows from the committed JSON dataset

so the answer keys of the eval and the numbers a real seller sees come from
the same code. Every method returns JSON-serializable data or raises
``SellerError`` (a business failure the agent or API client should see, e.g.
an unknown product). Numbers are computed here, never by the model.

Periods are fixed windows ending at ``as_of`` (the data date, "today"):

    last_7_days       as_of-6  .. as_of
    previous_7_days   as_of-13 .. as_of-7
    last_30_days      as_of-29 .. as_of
    previous_30_days  as_of-59 .. as_of-30
    last_90_days      as_of-89 .. as_of
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any, Literal

from ..schemas import canonical_json

Period = Literal["last_7_days", "previous_7_days", "last_30_days", "previous_30_days", "last_90_days"]
GroupBy = Literal["category", "audience", "week"]
PerformanceSort = Literal["units_desc", "units_asc", "revenue_desc", "revenue_asc"]

PERIODS: dict[str, tuple[int, int]] = {  # (days before as_of of the first day, of the last day)
    "last_7_days": (6, 0),
    "previous_7_days": (13, 7),
    "last_30_days": (29, 0),
    "previous_30_days": (59, 30),
    "last_90_days": (89, 0),
}
PREVIOUS_PERIOD = {"last_7_days": "previous_7_days", "last_30_days": "previous_30_days"}
VELOCITY_WINDOW_DAYS = 30
DEFAULT_RESTOCK_TARGET_DAYS = 14
CAPTION_FIELDS = ("name", "description", "tags")
CAPTION_LIMITS = {"name": 80, "description": 600, "tags": 8}
PRODUCT_KEYS = ("product_id", "sku", "name", "brand", "category", "audience", "description", "tags",
                "price_idr", "discount_percent", "final_price_idr", "rating", "warranty", "shipping",
                "return_policy")


class SellerError(Exception):
    """A business failure: returned to the agent as ``{"ok": false, "error": ...}``,
    to API clients as a 4xx with the same ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def period_bounds(period: str, as_of: date) -> tuple[date, date]:
    if period not in PERIODS:
        raise SellerError("invalid_period", f"period must be one of {sorted(PERIODS)}")
    first, last = PERIODS[period]
    return as_of - timedelta(days=first), as_of - timedelta(days=last)


def previous_period(period: str) -> str:
    """The equal-length window right before ``period`` (for ``compare_to_previous``)."""
    if period not in PREVIOUS_PERIOD:
        raise SellerError("invalid_compare", "compare_to_previous works with last_7_days or last_30_days")
    return PREVIOUS_PERIOD[period]


def change_pct(current: int, previous: int) -> float | None:
    """Percent change rounded to 0.1; None when the previous value is 0."""
    return None if previous == 0 else round((current - previous) / previous * 100, 1)


def trend(current: int, previous: int) -> str:
    return "up" if current > previous else "down" if current < previous else "flat"


def caption_version(product: Mapping[str, Any]) -> str:
    """Hash of the editable caption fields; a proposal is stale once it changes."""
    text = canonical_json({field: product[field] for field in CAPTION_FIELDS})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def caption_changes(product: Mapping[str, Any], name: str | None = None, description: str | None = None,
                    tags: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Validate a caption edit and return ``{field: {"old", "new"}}`` for the fields that change.

    ``None`` means "leave this field alone". Raises ``SellerError`` for a field
    outside its limits or when nothing would change.
    """
    changes: dict[str, dict[str, Any]] = {}
    for field, value in (("name", name), ("description", description), ("tags", tags)):
        if value is None:
            continue
        if field == "tags":
            value = [t.strip() for t in value if t.strip()]
            if not value or len(value) > CAPTION_LIMITS["tags"]:
                raise SellerError("invalid_tags", f"tags must have 1 to {CAPTION_LIMITS['tags']} entries")
        else:
            value = value.strip()
            if not value or len(value) > CAPTION_LIMITS[field]:
                raise SellerError("field_too_long", f"{field} must have 1 to {CAPTION_LIMITS[field]} characters")
        if value != product[field]:
            changes[field] = {"old": product[field], "new": value}
    if not changes:
        raise SellerError("no_changes", "propose at least one changed field: name, description or tags")
    return changes


class SellerAnalytics:
    """Queries over one store's rows. ``inventory`` maps product_id → ``{"stock": int, ...}``."""

    def __init__(self, products: Iterable[Mapping[str, Any]], orders: Iterable[Mapping[str, Any]],
                 reviews: Iterable[Mapping[str, Any]], inventory: Mapping[str, Mapping[str, Any]],
                 as_of: date) -> None:
        self.as_of = as_of
        self._products: dict[str, Mapping[str, Any]] = {p["product_id"]: p for p in products}
        self._orders = tuple(orders)
        self._reviews = tuple(reviews)
        self._inventory = inventory

    # ------------------------------------------------------------- helpers

    def _product(self, product_id: str) -> Mapping[str, Any]:
        product = self._products.get(product_id)
        if product is None:
            raise SellerError("product_not_found", f"no product with id {product_id!r}")
        return product

    def _orders_in(self, period: str) -> list[Mapping[str, Any]]:
        start, end = period_bounds(period, self.as_of)
        lo, hi = start.isoformat(), end.isoformat()
        return [o for o in self._orders if lo <= o["date"] <= hi]

    def _period(self, period: str) -> dict[str, str]:
        start, end = period_bounds(period, self.as_of)
        return {"name": period, "start": start.isoformat(), "end": end.isoformat()}

    def _check_category(self, category: str | None) -> None:
        categories = {p["category"] for p in self._products.values()}
        if category is not None and category not in categories:
            raise SellerError("unknown_category", f"category must be one of {sorted(categories)}")

    def _stock(self, product_id: str) -> int:
        return self._inventory[product_id]["stock"]

    def units_by_product(self, period: str) -> dict[str, int]:
        units: Counter[str] = Counter()
        for order in self._orders_in(period):
            for item in order["items"]:
                units[item["product_id"]] += item["quantity"]
        return {pid: units.get(pid, 0) for pid in self._products}

    # --------------------------------------------------------------- tools

    def sales_summary(self, period: str, group_by: str | None = None,
                      compare_to_previous: bool = False) -> dict[str, Any]:
        result = self._sales_summary(period, group_by)
        if not compare_to_previous:
            return result
        before = self._sales_summary(previous_period(period), None)
        result["compared_with"] = before["period"]
        for key in ("revenue_idr", "orders", "units"):
            result[f"previous_{key}"] = before[key]
            result[f"{key.removesuffix('_idr')}_change_pct"] = change_pct(result[key], before[key])
        result["trend"] = trend(result["revenue_idr"], before["revenue_idr"])
        return result

    def _sales_summary(self, period: str, group_by: str | None) -> dict[str, Any]:
        orders = self._orders_in(period)
        revenue = sum(o["total_idr"] for o in orders)
        units = sum(i["quantity"] for o in orders for i in o["items"])
        result: dict[str, Any] = {
            "period": self._period(period),
            "revenue_idr": revenue,
            "orders": len(orders),
            "units": units,
            "avg_order_value_idr": round(revenue / len(orders)) if orders else 0,
        }
        if group_by is None:
            return result
        if group_by not in ("category", "audience", "week"):
            raise SellerError("invalid_group_by", "group_by must be category, audience or week")
        groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"revenue_idr": 0, "units": 0, "orders": set()})
        for order in orders:
            for item in order["items"]:
                if group_by == "week":
                    day = date.fromisoformat(order["date"])
                    key = (day - timedelta(days=day.weekday())).isoformat()
                else:
                    key = self._products[item["product_id"]][group_by]
                group = groups[key]
                group["revenue_idr"] += item["quantity"] * item["unit_price_idr"]
                group["units"] += item["quantity"]
                group["orders"].add(order["order_id"])
        rows = [{"key": key, "revenue_idr": g["revenue_idr"], "units": g["units"], "orders": len(g["orders"])}
                for key, g in groups.items()]
        if group_by == "week":
            rows.sort(key=lambda r: r["key"])
        else:
            rows.sort(key=lambda r: (-r["revenue_idr"], r["key"]))
        result["group_by"] = group_by
        result["groups"] = rows
        return result

    def product_performance(self, period: str, sort: str = "units_desc", limit: int = 5,
                            category: str | None = None, audience: str | None = None,
                            compare_to_previous: bool = False) -> dict[str, Any]:
        if sort not in ("units_desc", "units_asc", "revenue_desc", "revenue_asc"):
            raise SellerError("invalid_sort", "sort must be units_desc, units_asc, revenue_desc or revenue_asc")
        if not 1 <= limit <= 54:
            raise SellerError("invalid_limit", "limit must be between 1 and 54")
        self._check_category(category)
        units = self.units_by_product(period)
        compared = previous_period(period) if compare_to_previous else None
        before = self.units_by_product(compared) if compared else {}
        rows = []
        for pid, product in self._products.items():
            if category and product["category"] != category:
                continue
            if audience and product["audience"] != audience:
                continue
            rows.append({
                "product_id": pid,
                "name": product["name"],
                "category": product["category"],
                "audience": product["audience"],
                "units": units[pid],
                "revenue_idr": units[pid] * product["final_price_idr"],
                "stock": self._stock(pid),
            })
            if compared:
                previous_units = before[pid]
                rows[-1].update(previous_units=previous_units, units_change=units[pid] - previous_units,
                                previous_revenue_idr=previous_units * product["final_price_idr"],
                                revenue_change_pct=change_pct(units[pid], previous_units),
                                trend=trend(units[pid], previous_units))
        metric, direction = sort.rsplit("_", 1)
        metric = "units" if metric == "units" else "revenue_idr"
        sign = -1 if direction == "desc" else 1
        rows.sort(key=lambda r: (sign * r[metric], r["product_id"]))
        result = {"period": self._period(period), "sort": sort, "total_matching": len(rows), "products": rows[:limit]}
        if compared:
            result["compared_with"] = self._period(compared)
        return result

    def inventory_rows(self, target_days: int = DEFAULT_RESTOCK_TARGET_DAYS) -> list[dict[str, Any]]:
        units = self.units_by_product("last_30_days")
        rows = []
        for pid, product in self._products.items():
            stock = self._stock(pid)
            velocity = units[pid] / VELOCITY_WINDOW_DAYS
            days = None if velocity == 0 else round(stock / velocity, 1)
            suggestion = 0 if velocity == 0 else max(0, math.ceil(velocity * target_days) - stock)
            rows.append({
                "product_id": pid,
                "name": product["name"],
                "category": product["category"],
                "stock": stock,
                "avg_daily_units_30d": round(velocity, 2),
                "days_to_stockout": days,
                "suggested_restock_units": suggestion,
            })
        rows.sort(key=lambda r: (r["days_to_stockout"] is None, r["days_to_stockout"] or 0, r["product_id"]))
        return rows

    def inventory_status(self, category: str | None = None, target_days: int = DEFAULT_RESTOCK_TARGET_DAYS,
                         limit: int = 10) -> dict[str, Any]:
        self._check_category(category)
        if not 1 <= target_days <= 90:
            raise SellerError("invalid_target_days", "target_days must be between 1 and 90")
        if not 1 <= limit <= 54:
            raise SellerError("invalid_limit", "limit must be between 1 and 54")
        rows = [r for r in self.inventory_rows(target_days) if not category or r["category"] == category]
        return {
            "as_of": self.as_of.isoformat(),
            "velocity_window_days": VELOCITY_WINDOW_DAYS,
            "target_days": target_days,
            "note": "days_to_stockout = stock / avg_daily_units_30d; null when nothing sold in 30 days.",
            "total_matching": len(rows),
            "products": rows[:limit],
        }

    def reviews(self, product_id: str | None = None, category: str | None = None,
                max_rating: int | None = None, period: str = "last_90_days", limit: int = 20) -> dict[str, Any]:
        if product_id is not None:
            self._product(product_id)
        self._check_category(category)
        if max_rating is not None and not 1 <= max_rating <= 5:
            raise SellerError("invalid_rating", "max_rating must be between 1 and 5")
        if not 1 <= limit <= 50:
            raise SellerError("invalid_limit", "limit must be between 1 and 50")
        start, end = period_bounds(period, self.as_of)
        lo, hi = start.isoformat(), end.isoformat()
        rows = [
            r for r in self._reviews
            if lo <= r["date"] <= hi
            and (product_id is None or r["product_id"] == product_id)
            and (category is None or self._products[r["product_id"]]["category"] == category)
            and (max_rating is None or r["rating"] <= max_rating)
        ]
        rows.sort(key=lambda r: (r["date"], r["review_id"]), reverse=True)
        counts = Counter(r["rating"] for r in rows)
        return {
            "period": self._period(period),
            "summary": {
                "count": len(rows),
                "avg_rating": round(sum(r["rating"] for r in rows) / len(rows), 2) if rows else None,
                "rating_counts": {str(k): counts.get(k, 0) for k in range(1, 6)},
            },
            "reviews": [
                {**r, "product_name": self._products[r["product_id"]]["name"]}
                for r in rows[:limit]
            ],
        }

    def product(self, product_id: str) -> dict[str, Any]:
        product = self._product(product_id)
        return {
            **{key: product[key] for key in PRODUCT_KEYS},
            "stock": self._stock(product_id),
            "caption_version": caption_version(product),
        }
