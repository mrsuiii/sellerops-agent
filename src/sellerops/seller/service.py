"""Seller service: the one place production reads store data and changes captions.

Callers::

    FastAPI routes (seller/api.py)          seller UI: analytics, approve/reject/revert
    InProcessSellerClient → seller tools    the seller agent (read + propose only)
    C3 approval graph (workflows/, Nigka)   propose → interrupt → approve/reject → revise

Caption proposal state machine (one transaction per transition, each one
writes an ``audit_log`` row)::

                 approve (base version still current)
      pending ─────────────────────────────────────────► applied ──revert──► reverted
         │  │                                              │
         │  ├── reject(feedback) ──► rejected ──► (revision: new pending, revision_of=…)
         │  └── cancel ──► cancelled   (seller stops the flow: no approve, no revision)
         │
         └── another proposal for the product is applied ──► stale

- Approve applies in the same transaction (no "approved but not applied"
  state to recover from): a new ``caption_versions`` row, the product's
  caption, the proposal status and the audit row commit together or not at
  all.
- A proposal is checked against the caption it was written for
  (``base_caption_version``). If the caption changed since, approving it
  would overwrite someone else's edit, so it becomes ``stale`` instead.
- Revert restores the caption from before the proposal as a *new* version
  (history is never rewritten) and only while the product still shows that
  proposal's caption; otherwise a later change would be silently undone.
- A rejected proposal can be revised once (``revision_of``); a chain stops
  after ``MAX_REJECTIONS`` rejections (the approval graph stops there too,
  with ``stop_reason="max_rejections"``).
- The agent's client has no approve/reject/revert: only a human (API) or the
  approval graph acting on a human's ``resume`` can apply a caption.

Orders, reviews and stock are imported facts, read-only here; they are read
once per service and cached. Products (whose captions change) are read on
every call.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

from .analytics import SellerAnalytics, SellerError, caption_changes, caption_version
from .db import connect, dumps, init_db, is_seeded, seed_from_files, transaction, utc_now

MAX_REJECTIONS = 5
# The demo has one store. Every service is bound to a store id anyway, so a
# multi-store setup only changes how ``SellerServices`` maps ids to databases
# (see ``registry.py``), not the callers.
DEFAULT_STORE_ID = "kala-studio"
STATUSES = ("pending", "applied", "rejected", "stale", "reverted", "cancelled")
AGENT_PROPOSAL_KEYS = ("proposal_id", "product_id", "status", "changes", "reason", "base_caption_version")


def format_proposal_id(row_id: int) -> str:
    return f"CP-{row_id:04d}"


def parse_proposal_id(proposal_id: str) -> int:
    prefix, _, number = proposal_id.partition("-")
    if prefix != "CP" or not number.isdigit():
        raise SellerError("proposal_not_found", f"no proposal with id {proposal_id!r}")
    return int(number)


class SellerService:
    """All reads and caption changes of **one store** (``store_id``), on that store's database."""

    def __init__(self, db_path: Path, *, store_id: str = DEFAULT_STORE_ID,
                 clock: Callable[[], str] = utc_now) -> None:
        self.db_path = db_path
        self.store_id = store_id
        self._clock = clock
        with self._connect() as conn:
            init_db(conn)
            # The database remembers which store it belongs to, so opening it
            # for another store fails loudly instead of mixing their data.
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('store_id', ?)", (store_id,))
            owner = conn.execute("SELECT value FROM meta WHERE key = 'store_id'").fetchone()["value"]
        if owner != store_id:
            raise RuntimeError(f"{db_path} belongs to store {owner!r}, not {store_id!r}")

    @classmethod
    def open(cls, db_path: Path, *, catalog_path: Path, sim_dir: Path, **kwargs: Any) -> SellerService:
        """Open (and on first use create and seed from the demo data) the database."""
        service = cls(db_path, **kwargs)
        with service._connect() as conn:
            if not is_seeded(conn):
                seed_from_files(conn, catalog_path, sim_dir)
        return service

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------ analytics

    @cached_property
    def _facts(self) -> tuple[date, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'as_of'").fetchone()
            if row is None:
                raise RuntimeError("seller database is not seeded")
            orders = tuple(json.loads(r["data"]) for r in conn.execute("SELECT data FROM orders ORDER BY order_id"))
            reviews = tuple(json.loads(r["data"]) for r in conn.execute("SELECT data FROM reviews ORDER BY review_id"))
        return date.fromisoformat(row["value"]), orders, reviews

    def _products(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute("SELECT * FROM products ORDER BY product_id").fetchall()
        return [_product_from_row(r) for r in rows]

    def analytics(self) -> SellerAnalytics:
        """A read model over the current products (fresh captions) and the cached facts."""
        as_of, orders, reviews = self._facts
        with self._connect() as conn:
            products = self._products(conn)
            stock = {r["product_id"]: {"stock": r["stock"]} for r in conn.execute("SELECT * FROM inventory")}
        return SellerAnalytics(products, orders, reviews, stock, as_of)

    # ------------------------------------------------------- caption reads

    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            return self._proposal(conn, parse_proposal_id(proposal_id))

    def list_proposals(self, *, status: str | None = None, product_id: str | None = None,
                       limit: int = 50) -> list[dict[str, Any]]:
        if status is not None and status not in STATUSES:
            raise SellerError("invalid_status", f"status must be one of {list(STATUSES)}")
        sql, args = "SELECT id FROM caption_proposals WHERE 1 = 1", []
        if status:
            sql, args = sql + " AND status = ?", [*args, status]
        if product_id:
            sql, args = sql + " AND product_id = ?", [*args, product_id]
        with self._connect() as conn:
            ids = [r["id"] for r in conn.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))]
            return [self._proposal(conn, i) for i in ids]

    def product_images(self, product_id: str) -> list[str]:
        """The product's catalog image URLs, thumbnail first (raises ``product_not_found``)."""
        with self._connect() as conn:
            data = json.loads(self._product_row(conn, product_id)["data"])
        urls = [data.get("thumbnail"), *data.get("images", [])]
        return list(dict.fromkeys(url for url in urls if url))

    def captions(self) -> dict[str, dict[str, Any]]:
        """The caption every product shows now (``name``, ``description``, ``tags``), by product id."""
        with self._connect() as conn:
            rows = conn.execute("SELECT product_id, name, description, tags FROM products").fetchall()
        return {r["product_id"]: {"name": r["name"], "description": r["description"],
                                  "tags": json.loads(r["tags"])} for r in rows}

    def caption_history(self, product_id: str) -> list[dict[str, Any]]:
        """Every caption the product had, oldest first; the last one is current."""
        with self._connect() as conn:
            self._product_row(conn, product_id)
            rows = conn.execute("SELECT * FROM caption_versions WHERE product_id = ? ORDER BY version_id",
                                (product_id,)).fetchall()
        return [{
            "version_id": r["version_id"], "caption_version": r["caption_version"], "name": r["name"],
            "description": r["description"], "tags": json.loads(r["tags"]), "source": r["source"],
            "proposal_id": format_proposal_id(r["proposal_id"]) if r["proposal_id"] else None,
            "created_at": r["created_at"], "created_by": r["created_by"],
        } for r in rows]

    def audit_log(self, *, product_id: str | None = None, proposal_id: str | None = None,
                  limit: int = 100) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM audit_log WHERE 1 = 1", []
        if product_id:
            sql, args = sql + " AND product_id = ?", [*args, product_id]
        if proposal_id:
            sql, args = sql + " AND proposal_id = ?", [*args, parse_proposal_id(proposal_id)]
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [{
            "id": r["id"], "at": r["at"], "actor": r["actor"], "action": r["action"],
            "product_id": r["product_id"],
            "proposal_id": format_proposal_id(r["proposal_id"]) if r["proposal_id"] else None,
            "detail": json.loads(r["detail"]),
        } for r in rows]

    # ------------------------------------------------- caption transitions

    def propose_caption(self, product_id: str, reason: str, *, actor: str, name: str | None = None,
                        description: str | None = None, tags: list[str] | None = None,
                        revision_of: str | None = None, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a ``pending`` proposal. The product does not change.

        ``inputs`` records what the author used besides the caption and product
        data, e.g. ``{"images": [ImageRef as dict, ...]}`` for the images the
        model saw (C3b); it is stored as given and shown with the proposal.
        """
        if not reason.strip():
            raise SellerError("missing_reason", "reason is required")
        now = self._clock()
        with self._connect() as conn, transaction(conn):
            product = _product_from_row(self._product_row(conn, product_id))
            changes = caption_changes(product, name, description, tags)
            previous = None
            if revision_of is not None:
                previous = self._revisable(conn, parse_proposal_id(revision_of), product_id)
            cur = conn.execute(
                "INSERT INTO caption_proposals (product_id, status, changes, reason, base_caption_version, "
                "revision_of, created_at, created_by, inputs) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?)",
                (product_id, dumps(changes), reason.strip(), caption_version(product), previous, now, actor,
                 dumps(inputs) if inputs else None))
            self._audit(conn, now, actor, "proposed", product_id, cur.lastrowid,
                        {"changes": changes, "reason": reason.strip(),
                         "revision_of": format_proposal_id(previous) if previous else None})
            return self._proposal(conn, cur.lastrowid)

    def approve(self, proposal_id: str, *, actor: str, note: str | None = None) -> dict[str, Any]:
        """Apply a pending proposal: new caption version + product caption, atomically."""
        pid = parse_proposal_id(proposal_id)
        now = self._clock()
        with self._connect() as conn:
            with transaction(conn):
                row = self._pending(conn, pid, "approve")
                product_row = self._product_row(conn, row["product_id"])
                stale = product_row["caption_version"] != row["base_caption_version"]
                if stale:
                    self._mark_stale(conn, now, actor, row, product_row["caption_version"])
                else:
                    self._apply(conn, now, actor, row, product_row, note)
            if stale:  # raised after COMMIT so the stale status and its audit row are kept
                raise SellerError("stale_proposal", f"{proposal_id} was written for an older caption; "
                                  "it is now stale. Propose again from the current caption.")
            return self._proposal(conn, pid)

    def reject(self, proposal_id: str, *, actor: str, feedback: str = "") -> dict[str, Any]:
        pid = parse_proposal_id(proposal_id)
        now = self._clock()
        with self._connect() as conn, transaction(conn):
            row = self._pending(conn, pid, "reject")
            conn.execute("UPDATE caption_proposals SET status = 'rejected', decided_at = ?, decided_by = ?, "
                         "feedback = ? WHERE id = ?", (now, actor, feedback.strip(), pid))
            self._audit(conn, now, actor, "rejected", row["product_id"], pid, {"feedback": feedback.strip()})
            return self._proposal(conn, pid)

    def cancel(self, proposal_id: str, *, actor: str, note: str | None = None) -> dict[str, Any]:
        """Close a pending proposal without a decision on its content (not revisable)."""
        pid = parse_proposal_id(proposal_id)
        now = self._clock()
        with self._connect() as conn, transaction(conn):
            row = self._pending(conn, pid, "cancel")
            conn.execute("UPDATE caption_proposals SET status = 'cancelled', decided_at = ?, decided_by = ?, "
                         "feedback = ? WHERE id = ?", (now, actor, (note or "").strip(), pid))
            self._audit(conn, now, actor, "cancelled", row["product_id"], pid, {"note": note})
            return self._proposal(conn, pid)

    def revert(self, proposal_id: str, *, actor: str, note: str | None = None) -> dict[str, Any]:
        """Undo an applied proposal by restoring the caption before it, as a new version."""
        pid = parse_proposal_id(proposal_id)
        now = self._clock()
        with self._connect() as conn, transaction(conn):
            row = self._row(conn, pid)
            if row["status"] != "applied":
                raise SellerError("invalid_transition", f"cannot revert a {row['status']} proposal")
            product_row = self._product_row(conn, row["product_id"])
            applied = conn.execute("SELECT caption_version FROM caption_versions WHERE version_id = ?",
                                   (row["applied_version_id"],)).fetchone()
            # Compare content, not version ids: after a newer change is itself reverted, the
            # caption is this proposal's again and reverting it undoes nothing else.
            if product_row["caption_version"] != applied["caption_version"]:
                raise SellerError("newer_change_exists", "the caption changed after this proposal; "
                                  "revert the newer change first")
            before = conn.execute(
                "SELECT * FROM caption_versions WHERE product_id = ? AND version_id < ? "
                "ORDER BY version_id DESC LIMIT 1", (row["product_id"], row["applied_version_id"])).fetchone()
            caption = {"name": before["name"], "description": before["description"],
                       "tags": json.loads(before["tags"])}
            version_id = self._write_caption(conn, now, actor, row["product_id"], caption, "revert", pid)
            conn.execute("UPDATE caption_proposals SET status = 'reverted', reverted_at = ?, reverted_by = ? "
                         "WHERE id = ?", (now, actor, pid))
            self._audit(conn, now, actor, "reverted", row["product_id"], pid,
                        {"from_version": product_row["caption_version"], "to_version": caption_version(caption),
                         "restored_version_id": before["version_id"], "new_version_id": version_id,
                         "note": note})
            return self._proposal(conn, pid)

    # ------------------------------------------------------------- helpers

    def _product_row(self, conn: sqlite3.Connection, product_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
        if row is None:
            raise SellerError("product_not_found", f"no product with id {product_id!r}")
        return row

    def _row(self, conn: sqlite3.Connection, pid: int) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM caption_proposals WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise SellerError("proposal_not_found", f"no proposal with id {format_proposal_id(pid)!r}")
        return row

    def _pending(self, conn: sqlite3.Connection, pid: int, action: str) -> sqlite3.Row:
        row = self._row(conn, pid)
        if row["status"] != "pending":
            raise SellerError("invalid_transition", f"cannot {action} a {row['status']} proposal")
        return row

    def _chain(self, conn: sqlite3.Connection, pid: int) -> list[sqlite3.Row]:
        """The proposal and everything it revises, newest first."""
        chain, current = [], pid
        while current is not None:
            row = self._row(conn, current)
            chain.append(row)
            current = row["revision_of"]
        return chain

    def _revisable(self, conn: sqlite3.Connection, pid: int, product_id: str) -> int:
        row = self._row(conn, pid)
        if row["product_id"] != product_id:
            raise SellerError("invalid_revision", "a revision must be for the same product")
        if row["status"] != "rejected":
            raise SellerError("invalid_revision", f"only a rejected proposal can be revised (it is {row['status']})")
        if conn.execute("SELECT 1 FROM caption_proposals WHERE revision_of = ?", (pid,)).fetchone():
            raise SellerError("invalid_revision", f"{format_proposal_id(pid)} was already revised")
        rejections = sum(r["status"] == "rejected" for r in self._chain(conn, pid))
        if rejections >= MAX_REJECTIONS:
            raise SellerError("max_rejections", f"the seller rejected {rejections} versions; stop revising")
        return pid

    def _write_caption(self, conn: sqlite3.Connection, now: str, actor: str, product_id: str,
                       caption: dict[str, Any], source: str, pid: int) -> int:
        version = caption_version(caption)
        cur = conn.execute(
            "INSERT INTO caption_versions (product_id, caption_version, name, description, tags, source, "
            "proposal_id, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (product_id, version, caption["name"], caption["description"], dumps(caption["tags"]), source,
             pid, now, actor))
        conn.execute("UPDATE products SET name = ?, description = ?, tags = ?, caption_version = ?, "
                     "current_version_id = ? WHERE product_id = ?",
                     (caption["name"], caption["description"], dumps(caption["tags"]), version, cur.lastrowid,
                      product_id))
        self._mark_others_stale(conn, now, product_id, version)
        return cur.lastrowid

    def _apply(self, conn: sqlite3.Connection, now: str, actor: str, row: sqlite3.Row,
               product_row: sqlite3.Row, note: str | None) -> None:
        current = _product_from_row(product_row)
        changes = json.loads(row["changes"])
        caption = {field: changes[field]["new"] if field in changes else current[field]
                   for field in ("name", "description", "tags")}
        conn.execute("UPDATE caption_proposals SET status = 'applied', decided_at = ?, decided_by = ? "
                     "WHERE id = ?", (now, actor, row["id"]))
        version_id = self._write_caption(conn, now, actor, row["product_id"], caption, "proposal", row["id"])
        conn.execute("UPDATE caption_proposals SET applied_version_id = ? WHERE id = ?", (version_id, row["id"]))
        self._audit(conn, now, actor, "approved", row["product_id"], row["id"],
                    {"changes": changes, "from_version": product_row["caption_version"],
                     "to_version": caption_version(caption), "version_id": version_id, "note": note})

    def _mark_stale(self, conn: sqlite3.Connection, now: str, actor: str, row: sqlite3.Row, current: str) -> None:
        conn.execute("UPDATE caption_proposals SET status = 'stale', decided_at = ?, decided_by = ? WHERE id = ?",
                     (now, actor, row["id"]))
        self._audit(conn, now, actor, "stale", row["product_id"], row["id"],
                    {"base_caption_version": row["base_caption_version"], "current_caption_version": current})

    def _mark_others_stale(self, conn: sqlite3.Connection, now: str, product_id: str, version: str) -> None:
        rows = conn.execute("SELECT * FROM caption_proposals WHERE product_id = ? AND status = 'pending' "
                            "AND base_caption_version != ?", (product_id, version)).fetchall()
        for row in rows:
            self._mark_stale(conn, now, "system", row, version)

    def _audit(self, conn: sqlite3.Connection, now: str, actor: str, action: str, product_id: str | None,
               pid: int | None, detail: dict[str, Any]) -> None:
        conn.execute("INSERT INTO audit_log (at, actor, action, product_id, proposal_id, detail) "
                     "VALUES (?, ?, ?, ?, ?, ?)", (now, actor, action, product_id, pid, dumps(detail)))

    def _proposal(self, conn: sqlite3.Connection, pid: int) -> dict[str, Any]:
        row = self._row(conn, pid)
        chain = self._chain(conn, pid)
        return {
            "proposal_id": format_proposal_id(row["id"]),
            "product_id": row["product_id"],
            "status": row["status"],
            "changes": json.loads(row["changes"]),
            "reason": row["reason"],
            "base_caption_version": row["base_caption_version"],
            "revision_of": format_proposal_id(row["revision_of"]) if row["revision_of"] else None,
            "rejections_in_chain": sum(r["status"] == "rejected" for r in chain),
            "created_at": row["created_at"], "created_by": row["created_by"],
            "decided_at": row["decided_at"], "decided_by": row["decided_by"],
            "feedback": row["feedback"],
            "applied_version_id": row["applied_version_id"],
            "reverted_at": row["reverted_at"], "reverted_by": row["reverted_by"],
            "inputs": json.loads(row["inputs"]) if row["inputs"] else None,
        }


def _product_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """Catalog data with the current caption columns on top."""
    return {**json.loads(row["data"]), "name": row["name"], "description": row["description"],
            "tags": json.loads(row["tags"])}


class InProcessSellerClient:
    """``SellerDataClient`` for production: the seller tools call the service directly.

    Same observation shape as the simulator's ``SellerEpisode``
    (``{"ok": true, "result": ...}`` / ``{"ok": false, "error": ...}``), so the
    agent sees identical tool results in eval and production. Only read tools
    and ``propose_caption_update`` exist here: the agent cannot approve.
    """

    def __init__(self, service: SellerService, *, actor: str = "agent") -> None:
        self.service = service
        self.actor = actor

    def _propose(self, **arguments: Any) -> dict[str, Any]:
        proposal = self.service.propose_caption(actor=self.actor, **arguments)
        return {**{k: proposal[k] for k in AGENT_PROPOSAL_KEYS},
                "note": "Pending seller approval. The product is unchanged."}

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool. Raises ``KeyError`` for an unknown tool (a bug, not an observation)."""
        readers = {
            "get_sales_summary": "sales_summary",
            "get_product_performance": "product_performance",
            "get_inventory_status": "inventory_status",
            "get_reviews": "reviews",
            "get_product": "product",
        }
        if tool == "propose_caption_update":
            handler: Callable[..., dict[str, Any]] = self._propose
        else:
            handler = getattr(self.service.analytics(), readers[tool])
        try:
            return {"ok": True, "result": handler(**arguments)}
        except SellerError as error:
            return {"ok": False, "error": {"code": error.code, "message": error.message}}
