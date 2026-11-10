"""Build the fashion store catalog snapshot from DummyJSON's public dataset.

DummyJSON (MIT licensed, https://github.com/Ovi/DummyJSON) ships a fixed
product database. We download it once, keep only the fashion categories,
normalize the fields we need, convert prices to IDR, and commit the result to
``data/store/fashion_catalog.json``. The store and agent then read only that
local snapshot, so runs stay deterministic and offline.

Usage (from the project directory):

    python scripts/import_fashion_catalog.py
    python scripts/import_fashion_catalog.py --source path/to/products.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_URL = "https://raw.githubusercontent.com/Ovi/DummyJSON/master/database/products.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "store" / "fashion_catalog.json"

USD_TO_IDR = 16_000

# DummyJSON category -> (store category, audience)
CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "mens-shirts": ("shirts", "men"),
    "mens-shoes": ("shoes", "men"),
    "mens-watches": ("watches", "men"),
    "womens-dresses": ("dresses", "women"),
    "tops": ("tops", "women"),
    "womens-shoes": ("shoes", "women"),
    "womens-bags": ("bags", "women"),
    "womens-watches": ("watches", "women"),
    "womens-jewellery": ("jewellery", "women"),
    "sunglasses": ("sunglasses", "unisex"),
    "fragrances": ("fragrances", "unisex"),
}


def to_idr(price_usd: float) -> int:
    """Convert a USD price to a shop-style IDR price ending in 9,000."""

    rounded = round(price_usd * USD_TO_IDR, -3)
    return max(int(rounded) - 1_000, 9_000)


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Map one DummyJSON product to the store schema (no reviewer emails)."""

    category, audience = CATEGORY_MAP[raw["category"]]
    price = to_idr(raw["price"])
    discount = round(float(raw.get("discountPercentage", 0.0)))
    return {
        "product_id": f"FS{raw['id']:03d}",
        "sku": raw.get("sku"),
        "name": raw["title"],
        "brand": raw.get("brand") or "Kala Studio",
        "category": category,
        "audience": audience,
        "description": raw["description"],
        "tags": sorted({tag.lower() for tag in raw.get("tags", [])}),
        "price_idr": price,
        "discount_percent": discount,
        "final_price_idr": int(round(price * (100 - discount) / 100, -3)),
        "rating": round(float(raw.get("rating", 0.0)), 2),
        "stock": int(raw.get("stock", 0)),
        "warranty": raw.get("warrantyInformation"),
        "shipping": raw.get("shippingInformation"),
        "return_policy": raw.get("returnPolicy"),
        "thumbnail": raw.get("thumbnail"),
        "images": raw.get("images", []),
        "reviews": [
            {
                "rating": review["rating"],
                "comment": review["comment"],
                "reviewer": review["reviewerName"],
                "date": review["date"][:10],
            }
            for review in raw.get("reviews", [])
        ],
        "source": f"dummyjson#{raw['id']}",
    }


def build_catalog(raw_products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return normalized fashion products sorted by product id."""

    products = [normalize(p) for p in raw_products if p["category"] in CATEGORY_MAP]
    return sorted(products, key=lambda p: p["product_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="Local DummyJSON products.json instead of downloading.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.source:
        raw = json.loads(args.source.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))

    catalog = {
        "store": "Kala Studio",
        "currency": "IDR",
        "source": "DummyJSON products database (MIT), fashion categories only",
        "source_url": SOURCE_URL,
        "usd_to_idr": USD_TO_IDR,
        "products": build_catalog(raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(catalog['products'])} products to {args.output}")


if __name__ == "__main__":
    main()
