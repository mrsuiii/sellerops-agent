"""Deterministic synthetic sales data for the Kala Studio catalog.

One fixed seed produces 90 days of orders, the stock left at the end, and
buyer reviews. A few scenarios are planted on purpose so that eval questions
have one verifiable answer:

- ``spike``: one product sells ~4x faster in the last 7 days;
- ``stockout``: a best-selling product runs out 20 days before ``AS_OF``, so
  its category's revenue drops in the last 30 days;
- ``low_stock``: a fast seller with only a few units left;
- ``slow_movers``: products with stock but no sale in the last 45 days;
- ``negative_reviews``: one product gets recurring complaints about the same
  issue in the last three weeks.

Stock is consistent by construction: ``initial_stock = units sold + stock``.
Nothing here reads the clock; ``AS_OF`` is the simulator's "today".
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ..schemas import canonical_json

DATA_VERSION = "seller-sim-v1"
DEFAULT_SEED = 20261117
AS_OF = date(2026, 11, 15)
DAYS = 90
START = AS_OF - timedelta(days=DAYS - 1)
BASE_ORDERS_PER_DAY = 22

SCENARIOS: dict[str, dict[str, Any]] = {
    "spike": {"product_id": "FS163", "multiplier": 4.0, "last_days": 7},
    "stockout": {"product_id": "FS186", "boost": 8.0, "out_of_stock_last_days": 20},
    "low_stock": {"product_id": "FS180", "multiplier": 2.5, "end_stock": 6},
    "slow_movers": {"product_ids": ["FS174", "FS157"], "no_sales_last_days": 45},
    "negative_reviews": {
        "product_id": "FS188",
        "issue": "size_runs_small",
        "count": 6,
        "last_days": 21,
    },
}

# Review comments by rating band (synthetic, Indonesian). The planted complaint
# uses its own pool so no other product gets the same recurring issue.
_COMMENTS = {
    "positive": [
        "Barangnya bagus, sesuai foto.",
        "Pengiriman cepat, packing rapi.",
        "Kualitas oke untuk harganya.",
        "Suka banget, bakal beli lagi.",
        "Bahannya nyaman dipakai.",
    ],
    "neutral": [
        "Lumayan, sesuai harga.",
        "Biasa saja, tidak ada yang istimewa.",
        "Warnanya sedikit beda dari foto.",
    ],
    "negative": [
        "Pengiriman lama sekali.",
        "Packing kurang rapi, kotaknya penyok.",
        "Kualitas tidak sesuai harapan.",
    ],
}
_SIZE_COMPLAINTS = [
    "Ukurannya kekecilan, padahal sudah pesan ukuran biasa.",
    "Kekecilan satu nomor, harus tukar ukuran.",
    "Ukuran kecil dari tabel size, sempit di kaki.",
    "Sepatunya sempit, ukurannya kecil sekali.",
    "Size-nya kekecilan, kecewa.",
    "Kekecilan, saran naik satu ukuran.",
]


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth; lam is small (< 40) here.
    threshold, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


def _base_weight(product: dict[str, Any]) -> float:
    price_factor = min(3.0, max(0.01, (500_000 / product["final_price_idr"]) ** 0.6))
    return price_factor * (0.5 + product["rating"] / 5)


def _day_factor(day: date) -> float:
    weekend = 1.3 if day.weekday() >= 5 else 1.0
    payday = 1.2 if day.day >= 25 or day.day == 1 else 1.0
    return weekend * payday


def _weight_on(product: dict[str, Any], base: float, day: date) -> float:
    pid, days_left = product["product_id"], (AS_OF - day).days  # 0 on AS_OF
    weight = base
    spike, stockout = SCENARIOS["spike"], SCENARIOS["stockout"]
    if pid == spike["product_id"] and days_left < spike["last_days"]:
        weight *= spike["multiplier"]
    if pid == stockout["product_id"]:
        if days_left < stockout["out_of_stock_last_days"]:
            return 0.0
        weight *= stockout["boost"]
    if pid == SCENARIOS["low_stock"]["product_id"]:
        weight *= SCENARIOS["low_stock"]["multiplier"]
    slow = SCENARIOS["slow_movers"]
    if pid in slow["product_ids"] and days_left < slow["no_sales_last_days"]:
        return 0.0
    return weight


def _review_comment(rng: random.Random, rating: int) -> str:
    band = "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"
    return rng.choice(_COMMENTS[band])


def _review_rating(rng: random.Random, product_rating: float) -> int:
    value = round(rng.gauss(product_rating, 0.9))
    return min(5, max(1, value))


def generate(catalog: dict[str, Any], seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Return ``{"orders", "reviews", "inventory"}`` for the catalog (pure function)."""

    rng = random.Random(seed)
    products = sorted(catalog["products"], key=lambda p: p["product_id"])
    base = {p["product_id"]: _base_weight(p) for p in products}
    sold = dict.fromkeys(base, 0)
    orders: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    negative = SCENARIOS["negative_reviews"]

    for offset in range(DAYS):
        day = START + timedelta(days=offset)
        weights = [_weight_on(p, base[p["product_id"]], day) for p in products]
        for _ in range(_poisson(rng, BASE_ORDERS_PER_DAY * _day_factor(day))):
            n_items = rng.choice((1, 1, 1, 2, 2, 3))
            picked: list[dict[str, Any]] = []
            pool = list(zip(products, weights, strict=True))
            for _ in range(n_items):
                candidates = [(p, w) for p, w in pool if w > 0]
                if not candidates:
                    break
                product = rng.choices([p for p, _ in candidates], [w for _, w in candidates])[0]
                pool = [(p, w) for p, w in pool if p is not product]
                picked.append(product)
            if not picked:
                continue
            order_id = f"O{len(orders) + 1:05d}"
            items = []
            for product in picked:
                quantity = 1 if rng.random() < 0.85 else 2
                sold[product["product_id"]] += quantity
                items.append({
                    "product_id": product["product_id"],
                    "quantity": quantity,
                    "unit_price_idr": product["final_price_idr"],
                })
                is_planted = (product["product_id"] == negative["product_id"]
                              and (AS_OF - day).days < negative["last_days"])
                if not is_planted and rng.random() < 0.08:
                    rating = _review_rating(rng, product["rating"])
                    review_day = min(AS_OF, day + timedelta(days=rng.randint(2, 7)))
                    reviews.append({
                        "product_id": product["product_id"],
                        "order_id": order_id,
                        "rating": rating,
                        "comment": _review_comment(rng, rating),
                        "date": review_day.isoformat(),
                    })
            orders.append({
                "order_id": order_id,
                "date": day.isoformat(),
                "customer_id": f"C{rng.randint(1, 900):04d}",
                "items": items,
                "total_idr": sum(i["quantity"] * i["unit_price_idr"] for i in items),
            })

    # Planted complaint: attach to that product's orders in the last weeks.
    planted_orders = [
        o for o in orders
        if (AS_OF - date.fromisoformat(o["date"])).days < negative["last_days"]
        and any(i["product_id"] == negative["product_id"] for i in o["items"])
    ]
    if len(planted_orders) < negative["count"]:
        raise RuntimeError("not enough orders to plant the negative-review scenario")
    for order, comment in zip(planted_orders[: negative["count"]], _SIZE_COMPLAINTS, strict=True):
        review_day = min(AS_OF, date.fromisoformat(order["date"]) + timedelta(days=2))
        reviews.append({
            "product_id": negative["product_id"],
            "order_id": order["order_id"],
            "rating": rng.choice((1, 2)),
            "comment": comment,
            "date": review_day.isoformat(),
        })

    reviews.sort(key=lambda r: (r["date"], r["order_id"], r["product_id"]))
    for index, review in enumerate(reviews, start=1):
        review["review_id"] = f"R{index:05d}"

    end_stock = {p["product_id"]: p["stock"] for p in products}
    end_stock[SCENARIOS["stockout"]["product_id"]] = 0
    end_stock[SCENARIOS["low_stock"]["product_id"]] = SCENARIOS["low_stock"]["end_stock"]
    inventory = [
        {"product_id": pid, "initial_stock": sold[pid] + end_stock[pid], "units_sold": sold[pid],
         "stock": end_stock[pid]}
        for pid in sorted(base)
    ]
    return {"orders": orders, "reviews": reviews, "inventory": inventory}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def dataset_files(catalog: dict[str, Any], seed: int = DEFAULT_SEED) -> dict[str, str]:
    """File name → text for ``data/seller_sim/`` (manifest last, with checksums)."""

    data = generate(catalog, seed)
    files = {
        "orders.jsonl": _jsonl(data["orders"]),
        "reviews.jsonl": _jsonl(data["reviews"]),
        "inventory.json": json.dumps(data["inventory"], ensure_ascii=False, indent=1) + "\n",
    }
    manifest = {
        "data_version": DATA_VERSION,
        "seed": seed,
        "as_of": AS_OF.isoformat(),
        "start": START.isoformat(),
        "days": DAYS,
        "catalog_sha256": _sha256(canonical_json(catalog["products"])),
        "counts": {"orders": len(data["orders"]), "reviews": len(data["reviews"]),
                   "products": len(data["inventory"])},
        "scenarios": SCENARIOS,
        "files": {name: _sha256(text) for name, text in sorted(files.items())},
    }
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
    return files


def write_dataset(catalog: dict[str, Any], out_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = dataset_files(catalog, seed)
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    return json.loads(files["manifest.json"])
