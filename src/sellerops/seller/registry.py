"""Which ``SellerService`` serves a store: the seam for going multi-store later.

Callers (the API, the agent's client, the approval wiring) never build a
service themselves; they ask ``SellerServices.for_store(store_id)`` with the
store id resolved at the request boundary. Today there is one store and one
database (``SingleStoreServices``). A multi-store version implements the same
``for_store`` (e.g. one database per store, or one shared database with a
``store_id`` column), and nothing above it changes.
"""

from __future__ import annotations

from typing import Protocol

from .analytics import SellerError
from .service import SellerService


class SellerServices(Protocol):
    def for_store(self, store_id: str) -> SellerService: ...


class SingleStoreServices:
    """The demo: one store, one service; any other store id is unknown."""

    def __init__(self, service: SellerService) -> None:
        self._service = service

    def for_store(self, store_id: str) -> SellerService:
        if store_id != self._service.store_id:
            # Same answer as a missing store: do not reveal which ids exist.
            raise SellerError("store_not_found", f"no store with id {store_id!r}")
        return self._service
