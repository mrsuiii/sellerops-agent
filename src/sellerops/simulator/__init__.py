"""Seller simulator: synthetic orders, stock and reviews for Kala Studio (eval only).

- ``generator``: deterministic 90-day dataset with planted scenarios
  (``python scripts/generate_seller_sim.py`` writes ``data/seller_sim/``).
- ``store``: read model over that dataset (sales, performance, stock, reviews,
  products) plus caption proposals; the logic behind the seller tools.
- ``episode``: one isolated episode (own proposals, tool history) per eval run.

The harness never imports this package; eval code passes tools built on it.
"""

from .episode import SellerEpisode
from .store import SellerStore, ToolError, load_store

__all__ = ["SellerEpisode", "SellerStore", "ToolError", "load_store"]
