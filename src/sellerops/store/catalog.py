"""Fashion store catalog: loading, filtering, keyword search, recommendations.

Everything here is deterministic and dependency-free so the same functions can
back the web API and, later, the agent's ``search``/``recommend_products``
tools. The recommender is intentionally small: hard filters, BM25 over the
product text, then a fixed weighted score. Swap in embeddings later if needed;
the agent loop is the interesting part of the project, not this file.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ..settings import get_settings

SortKey = Literal["relevance", "price_asc", "price_desc", "rating", "discount"]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with a naive plural strip ("shoes" -> "shoe")."""

    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token.endswith("sses"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def product_text(product: dict[str, Any]) -> str:
    """Searchable text; name/category/tags are repeated to weigh them more."""

    head = " ".join([product["name"], product["category"], product["brand"], *product["tags"]])
    return f"{head} {head} {product['audience']} {product['description']}"


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, docs: Iterable[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [Counter(tokenize(doc)) for doc in docs]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.k1, self.b = k1, b

    def score(self, query: str, index: int) -> float:
        doc, length = self.docs[index], self.lengths[index]
        total = 0.0
        for term in set(tokenize(query)):
            tf = doc.get(term, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf[term] * tf * (self.k1 + 1) / norm
        return total


@dataclass(frozen=True)
class ProductFilter:
    """Hard constraints. ``None`` means "no constraint"."""

    category: str | None = None
    audience: str | None = None  # men | women | unisex
    brand: str | None = None
    min_price: int | None = None  # compared against final_price_idr
    max_price: int | None = None
    min_rating: float | None = None
    in_stock: bool = False
    exclude_ids: frozenset[str] = field(default_factory=frozenset)

    def matches(self, product: dict[str, Any]) -> bool:
        price = product["final_price_idr"]
        if self.category and product["category"] != self.category:
            return False
        # "men" also accepts unisex items, same for "women".
        if self.audience and product["audience"] not in {self.audience, "unisex"}:
            return False
        if self.brand and product["brand"].lower() != self.brand.lower():
            return False
        if self.min_price is not None and price < self.min_price:
            return False
        if self.max_price is not None and price > self.max_price:
            return False
        if self.min_rating is not None and product["rating"] < self.min_rating:
            return False
        if self.in_stock and product["stock"] <= 0:
            return False
        return product["product_id"] not in self.exclude_ids


class Catalog:
    """In-memory read-only catalog with search and recommendation."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.meta = {k: v for k, v in data.items() if k != "products"}
        self.products: list[dict[str, Any]] = data["products"]
        self._by_id = {p["product_id"]: p for p in self.products}
        self._bm25 = BM25(product_text(p) for p in self.products)

    @classmethod
    def from_file(cls, path: Path | None = None) -> Catalog:
        """Load a catalog JSON file; defaults to ``SELLEROPS_CATALOG_PATH``."""

        path = path or get_settings().catalog_path
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def get(self, product_id: str) -> dict[str, Any] | None:
        return self._by_id.get(product_id)

    def categories(self) -> list[dict[str, Any]]:
        counts = Counter(p["category"] for p in self.products)
        return [{"category": name, "count": counts[name]} for name in sorted(counts)]

    def brands(self) -> list[str]:
        return sorted({p["brand"] for p in self.products})

    def search(
        self,
        query: str = "",
        filters: ProductFilter = ProductFilter(),
        sort: SortKey = "relevance",
    ) -> list[dict[str, Any]]:
        """Filter, then keep keyword matches (if a query is given), then sort."""

        rows = []
        for index, product in enumerate(self.products):
            if not filters.matches(product):
                continue
            relevance = self._bm25.score(query, index) if query.strip() else 0.0
            if query.strip() and relevance <= 0:
                continue
            rows.append((relevance, product))

        sorters = {
            "relevance": lambda r: (-r[0], -r[1]["rating"]),
            "price_asc": lambda r: r[1]["final_price_idr"],
            "price_desc": lambda r: -r[1]["final_price_idr"],
            "rating": lambda r: -r[1]["rating"],
            "discount": lambda r: -r[1]["discount_percent"],
        }
        rows.sort(key=lambda r: (sorters[sort](r), r[1]["product_id"]))
        return [product for _, product in rows]

    def recommend(
        self,
        query: str = "",
        filters: ProductFilter = ProductFilter(in_stock=True),
        limit: int = 5,
        similar_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rank candidates that pass ``filters``.

        score = 0.6 * relevance (BM25, scaled to 0..1 within candidates)
              + 0.3 * rating / 5
              + 0.1 * min(discount, 50) / 50

        ``similar_to`` uses that product's name + category + tags as the query
        and excludes the product itself. Each result carries ``score`` and a
        short ``reasons`` list so the agent can explain the pick.
        """

        anchor = self.get(similar_to) if similar_to else None
        if anchor:
            query = f"{query} {anchor['name']} {anchor['category']} {' '.join(anchor['tags'])}"
            filters = ProductFilter(**{**filters.__dict__, "exclude_ids": filters.exclude_ids | {anchor["product_id"]}})

        candidates = [(i, p) for i, p in enumerate(self.products) if filters.matches(p)]
        if not candidates:
            return []
        raw = [self._bm25.score(query, i) if query.strip() else 0.0 for i, _ in candidates]
        top = max(raw) or 1.0

        results = []
        for (_, product), relevance in zip(candidates, raw, strict=True):
            rel = relevance / top
            if query.strip() and rel == 0:
                continue
            score = 0.6 * rel + 0.3 * product["rating"] / 5 + 0.1 * min(product["discount_percent"], 50) / 50
            reasons = []
            if rel >= 0.5:
                reasons.append("cocok dengan kata kunci")
            if product["rating"] >= 4.5:
                reasons.append(f"rating tinggi ({product['rating']})")
            if product["discount_percent"] >= 10:
                reasons.append(f"diskon {product['discount_percent']}%")
            if 0 < product["stock"] <= 5:
                reasons.append(f"stok tinggal {product['stock']}")
            results.append({**summary(product), "score": round(score, 4), "reasons": reasons})

        results.sort(key=lambda r: (-r["score"], r["product_id"]))
        return results[:limit]


def summary(product: dict[str, Any]) -> dict[str, Any]:
    """The compact fields used in lists and tool outputs (no reviews/images)."""

    keys = (
        "product_id", "name", "brand", "category", "audience", "price_idr",
        "discount_percent", "final_price_idr", "rating", "stock", "thumbnail",
    )
    return {key: product[key] for key in keys}


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    """Process-wide catalog loaded from the committed snapshot."""

    return Catalog.from_file()
