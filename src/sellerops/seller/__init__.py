"""Seller backend (production): analytics, SQLite storage, caption approval service, HTTP API.

- ``analytics``: the numbers behind the seller tools (shared with the simulator).
- ``db``: SQLite schema and seeding from the demo data.
- ``service``: ``SellerService`` (reads + caption state machine + audit log) and
  ``InProcessSellerClient`` (the ``SellerDataClient`` the seller agent's tools use).
- ``registry``: ``SellerServices.for_store(store_id)``, the one place a store id
  becomes a service (single store today; the seam for multi-store later).
- ``api``: FastAPI router under ``/api/seller`` (needs the ``store`` extra).

This package never imports ``harness``, ``evaluation`` or ``simulator``.
"""

from .analytics import SellerAnalytics, SellerError
from .registry import SellerServices, SingleStoreServices
from .service import DEFAULT_STORE_ID, MAX_REJECTIONS, InProcessSellerClient, SellerService

__all__ = ["DEFAULT_STORE_ID", "MAX_REJECTIONS", "InProcessSellerClient", "SellerAnalytics", "SellerError",
           "SellerService", "SellerServices", "SingleStoreServices"]
