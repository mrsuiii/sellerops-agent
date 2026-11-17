"""Read model over the seller simulator data: the logic behind the seller tools.

Every method returns plain JSON-serializable data or raises ``ToolError``
(a business failure the agent should see, e.g. an unknown product). Numbers
are computed here, never by the model ("angka berasal dari tool").

Periods are fixed windows ending at ``AS_OF`` (the simulator's today):

    last_7_days       AS_OF-6  .. AS_OF
    previous_7_days   AS_OF-13 .. AS_OF-7
    last_30_days      AS_OF-29 .. AS_OF
    previous_30_days  AS_OF-59 .. AS_OF-30
    last_90_days      AS_OF-89 .. AS_OF

Caption proposals are the only write, and they never change a product: they
are stored as ``pending`` for a human to approve (Session C3).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ..schemas import canonical_json
from ..settings import get_settings
from .generator import AS_OF

Period = Literal["last_7_days", "previous_7_days", "last_30_days", "previous_30_days", "last_90_days"]
GroupBy = Literal["category", "audience", "week"]
PerformanceSort = Literal["units_desc", "units_asc", "revenue_desc", "revenue_asc"]

PERIODS: dict[str, tuple[int, int]] = {  # (days before AS_OF of the first day, of the last day)
    "last_7_days": (6, 0),
    "previous_7_days": (13, 7),
    "last_30_days": (29, 0),
    "previous_30_days": (59, 30),
    "last_90_days": (89, 0),
}
VELOCITY_WINDOW_DAYS = 30
DEFAULT_RESTOCK_TARGET_DAYS = 14
CAPTION_FIELDS = ("name", "description", "tags")
CAPTION_LIMITS = {"name": 80, "description": 600, "tags": 8}


class ToolError(Exception):
    """A business failure returned to the agent as ``{"ok": false, "error": ...}``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SimData:
    catalog: dict[str, Any]
    orders: tuple[dict[str, Any], ...]
    reviews: tuple[dict[str, Any], ...]
    inventory: dict[str, dict[str, Any]]
    manifest: dict[str, Any]


def period_bounds(period: str) -> tuple[date, date]:
    if period not in PERIODS:
        raise ToolError("invalid_period", f"period must be one of {sorted(PERIODS)}")
    first, last = PERIODS[period]
    return AS_OF - timedelta(days=first), AS_OF - timedelta(days=last)


PREVIOUS_PERIOD = {"last_7_days": "previous_7_days", "last_30_days": "previous_30_days"}


def previous_period(period: str) -> str:
    """The equal-length window right before ``period`` (for ``compare_to_previous``)."""
    if period not in PREVIOUS_PERIOD:
        raise ToolError("invalid_compare", "compare_to_previous works with last_7_days or last_30_days")
    return PREVIOUS_PERIOD[period]


def change_pct(current: int, previous: int) -> float | None:
    """Percent change rounded to 0.1; None when the previous value is 0."""
    return None if previous == 0 else round((current - previous) / previous * 100, 1)


def trend(current: int, previous: int) -> str:
    return "up" if current > previous else "down" if current < previous else "flat"


def caption_version(product: dict[str, Any]) -> str:
    """Hash of the editable caption fields; a proposal is stale once it changes."""
    text = canonical_json({field: product[field] for field in CAPTION_FIELDS})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class SellerStore:
    """Queries over one simulator dataset. Proposals live on the instance."""

    def __init__(self, data: SimData) -> None:
        self.data = data
        self.as_of = AS_OF
        self._products = {p["product_id"]: p for p in data.catalog["products"]}
        self.proposals: list[dict[str, Any]] = []

    # ------------------------------------------------------------- helpers

    def _product(self, product_id: str) -> dict[str, Any]:
        product = self._products.get(product_id)
        if product is None:
            raise ToolError("product_not_found", f"no product with id {product_id!r}")
        return product

    def _orders_in(self, period: str) -> list[dict[str, Any]]:
        start, end = period_bounds(period)
        lo, hi = start.isoformat(), end.isoformat()
        return [o for o in self.data.orders if lo <= o["date"] <= hi]

    def _period(self, period: str) -> dict[str, str]:
        start, end = period_bounds(period)
        return {"name": period, "start": start.isoformat(), "end": end.isoformat()}

    def _check_category(self, category: str | None) -> None:
        if category is not None and category not in {p["category"] for p in self._products.values()}:
            categories = sorted({p["category"] for p in self._products.values()})
            raise ToolError("unknown_category", f"category must be one of {categories}")

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
            raise ToolError("invalid_group_by", "group_by must be category, audience or week")
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
            raise ToolError("invalid_sort", "sort must be units_desc, units_asc, revenue_desc or revenue_asc")
        if not 1 <= limit <= 54:
            raise ToolError("invalid_limit", "limit must be between 1 and 54")
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
                "stock": self.data.inventory[pid]["stock"],
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
            stock = self.data.inventory[pid]["stock"]
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
            raise ToolError("invalid_target_days", "target_days must be between 1 and 90")
        if not 1 <= limit <= 54:
            raise ToolError("invalid_limit", "limit must be between 1 and 54")
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
            raise ToolError("invalid_rating", "max_rating must be between 1 and 5")
        if not 1 <= limit <= 50:
            raise ToolError("invalid_limit", "limit must be between 1 and 50")
        start, end = period_bounds(period)
        lo, hi = start.isoformat(), end.isoformat()
        rows = [
            r for r in self.data.reviews
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
        keys = ("product_id", "sku", "name", "brand", "category", "audience", "description", "tags",
                "price_idr", "discount_percent", "final_price_idr", "rating", "warranty", "shipping",
                "return_policy")
        return {
            **{key: product[key] for key in keys},
            "stock": self.data.inventory[product_id]["stock"],
            "caption_version": caption_version(product),
        }

    def propose_caption_update(self, product_id: str, reason: str, name: str | None = None,
                               description: str | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        product = self._product(product_id)
        if not reason.strip():
            raise ToolError("missing_reason", "reason is required")
        proposed = {"name": name, "description": description, "tags": tags}
        changes: dict[str, dict[str, Any]] = {}
        for field, value in proposed.items():
            if value is None:
                continue
            if field == "tags":
                value = [t.strip() for t in value if t.strip()]
                if not value or len(value) > CAPTION_LIMITS["tags"]:
                    raise ToolError("invalid_tags", f"tags must have 1 to {CAPTION_LIMITS['tags']} entries")
            else:
                value = value.strip()
                if not value or len(value) > CAPTION_LIMITS[field]:
                    raise ToolError("field_too_long",
                                    f"{field} must have 1 to {CAPTION_LIMITS[field]} characters")
            if value != product[field]:
                changes[field] = {"old": product[field], "new": value}
        if not changes:
            raise ToolError("no_changes", "propose at least one changed field: name, description or tags")
        proposal = {
            "proposal_id": f"CP-{len(self.proposals) + 1:04d}",
            "product_id": product_id,
            "status": "pending",
            "changes": changes,
            "reason": reason.strip(),
            "base_caption_version": caption_version(product),
        }
        self.proposals.append(proposal)
        return {**proposal, "note": "Pending seller approval. The product is unchanged."}


# ------------------------------------------------------------------ loading


def default_sim_dir() -> Path:
    return get_settings().data_dir / "seller_sim"


def load_sim_data(sim_dir: Path | None = None, catalog_path: Path | None = None) -> SimData:
    sim_dir = sim_dir or default_sim_dir()
    catalog_path = catalog_path or get_settings().catalog_path

    def jsonl(name: str) -> tuple[dict[str, Any], ...]:
        lines = (sim_dir / name).read_text(encoding="utf-8").splitlines()
        return tuple(json.loads(line) for line in lines if line)

    inventory = json.loads((sim_dir / "inventory.json").read_text(encoding="utf-8"))
    return SimData(
        catalog=json.loads(catalog_path.read_text(encoding="utf-8")),
        orders=jsonl("orders.jsonl"),
        reviews=jsonl("reviews.jsonl"),
        inventory={row["product_id"]: row for row in inventory},
        manifest=json.loads((sim_dir / "manifest.json").read_text(encoding="utf-8")),
    )


@lru_cache(maxsize=1)
def _default_data() -> SimData:
    return load_sim_data()


def load_store(data: SimData | None = None) -> SellerStore:
    """A fresh store (no proposals) over the committed dataset, or over ``data``."""
    return SellerStore(data or _default_data())
