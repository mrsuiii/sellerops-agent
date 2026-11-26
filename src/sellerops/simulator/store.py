"""Seller simulator store: the shared analytics over the committed JSON dataset.

The numbers come from ``sellerops.seller.analytics`` (the same code the
production ``SellerService`` runs over SQLite), so eval answer keys and what
a real seller sees cannot drift apart. What is simulator-specific lives here:
loading the dataset from ``data/seller_sim/`` and keeping caption proposals
in memory, per episode.

Caption proposals are the only write, and they never change a product: they
are stored as ``pending`` for a human to approve. The approve / apply /
revert flow is production-only (``sellerops.seller.captions``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..seller.analytics import (  # noqa: F401  (re-exported for simulator users)
    CAPTION_FIELDS,
    CAPTION_LIMITS,
    DEFAULT_RESTOCK_TARGET_DAYS,
    PERIODS,
    PREVIOUS_PERIOD,
    VELOCITY_WINDOW_DAYS,
    GroupBy,
    PerformanceSort,
    Period,
    SellerAnalytics,
    caption_changes,
    caption_version,
    change_pct,
    previous_period,
    trend,
)
from ..seller.analytics import SellerError as ToolError
from ..settings import get_settings
from .generator import AS_OF


@dataclass(frozen=True)
class SimData:
    catalog: dict[str, Any]
    orders: tuple[dict[str, Any], ...]
    reviews: tuple[dict[str, Any], ...]
    inventory: dict[str, dict[str, Any]]
    manifest: dict[str, Any]


class SellerStore(SellerAnalytics):
    """Analytics over one simulator dataset. Proposals live on the instance."""

    def __init__(self, data: SimData) -> None:
        super().__init__(data.catalog["products"], data.orders, data.reviews, data.inventory, AS_OF)
        self.data = data
        self.proposals: list[dict[str, Any]] = []

    def propose_caption_update(self, product_id: str, reason: str, name: str | None = None,
                               description: str | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        product = self._product(product_id)
        if not reason.strip():
            raise ToolError("missing_reason", "reason is required")
        changes = caption_changes(product, name, description, tags)
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
