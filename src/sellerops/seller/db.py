"""SQLite storage for the seller backend: store data, caption versions, proposals, audit log.

The database is the source of truth for captions (C3 decision, 17 Nov): the
approval graph and the HTTP API only call ``SellerService``, which reads and
writes these tables in one transaction per operation.

Tables::

    meta              key/value: as_of (data date), data_version, seeded_at
    products          catalog row (JSON) + current caption columns + current_version_id
    inventory         stock per product
    orders, reviews   imported facts (JSON rows), read-only in this demo
    caption_versions  every caption a product ever had; append-only
    caption_proposals one row per proposal and its decision (state machine in service.py)
    audit_log         who did what, when; append-only

Invariants the schema enforces itself (triggers), so a bug in the service
cannot silently break them:

- ``audit_log`` and ``caption_versions`` are append-only (no UPDATE/DELETE).
- A product's caption (name, description, tags) only changes together with
  ``current_version_id``, and always equals that version row: every caption
  the shop shows has a version (and so an author) behind it.

What SQLite does here: ``isolation_level=None`` turns off the sqlite3 module's
implicit transactions, so ``transaction()`` controls them explicitly.
``BEGIN IMMEDIATE`` takes the write lock at the start, so two concurrent
approvals of one proposal run one after the other (the second sees the new
status) instead of both reading "pending". WAL mode lets readers keep reading
while a write is in progress; ``busy_timeout`` makes a writer wait for the
lock instead of failing at once.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .analytics import caption_version

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id         TEXT PRIMARY KEY,
    data               TEXT NOT NULL,
    name               TEXT NOT NULL,
    description        TEXT NOT NULL,
    tags               TEXT NOT NULL,
    caption_version    TEXT NOT NULL,
    current_version_id INTEGER
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id TEXT PRIMARY KEY REFERENCES products(product_id),
    stock      INTEGER NOT NULL CHECK (stock >= 0)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    date     TEXT NOT NULL,
    data     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_date ON orders(date);

CREATE TABLE IF NOT EXISTS reviews (
    review_id  TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    date       TEXT NOT NULL,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS caption_proposals (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id           TEXT NOT NULL REFERENCES products(product_id),
    status               TEXT NOT NULL
                         CHECK (status IN ('pending', 'applied', 'rejected', 'stale', 'reverted', 'cancelled')),
    changes              TEXT NOT NULL,
    reason               TEXT NOT NULL,
    base_caption_version TEXT NOT NULL,
    revision_of          INTEGER REFERENCES caption_proposals(id),
    created_at           TEXT NOT NULL,
    created_by           TEXT NOT NULL,
    decided_at           TEXT,
    decided_by           TEXT,
    feedback             TEXT,
    applied_version_id   INTEGER REFERENCES caption_versions(version_id),
    reverted_at          TEXT,
    reverted_by          TEXT
);
CREATE INDEX IF NOT EXISTS proposals_product ON caption_proposals(product_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS proposals_one_revision ON caption_proposals(revision_of)
    WHERE revision_of IS NOT NULL;

CREATE TABLE IF NOT EXISTS caption_versions (
    version_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id      TEXT NOT NULL REFERENCES products(product_id),
    caption_version TEXT NOT NULL,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL,
    tags            TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('seed', 'proposal', 'revert')),
    proposal_id     INTEGER REFERENCES caption_proposals(id),
    created_at      TEXT NOT NULL,
    created_by      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_product ON caption_versions(product_id, version_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    product_id  TEXT,
    proposal_id INTEGER,
    detail      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_product ON audit_log(product_id, id);

CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS caption_versions_no_update BEFORE UPDATE ON caption_versions
BEGIN SELECT RAISE(ABORT, 'caption_versions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS caption_versions_no_delete BEFORE DELETE ON caption_versions
BEGIN SELECT RAISE(ABORT, 'caption_versions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS products_caption_needs_version
BEFORE UPDATE OF name, description, tags, current_version_id ON products
WHEN ((NEW.name IS NOT OLD.name OR NEW.description IS NOT OLD.description OR NEW.tags IS NOT OLD.tags)
      AND NEW.current_version_id IS OLD.current_version_id)
  OR NOT EXISTS (SELECT 1 FROM caption_versions v
                 WHERE v.version_id = NEW.current_version_id AND v.product_id = NEW.product_id
                   AND v.name = NEW.name AND v.description = NEW.description AND v.tags = NEW.tags)
BEGIN SELECT RAISE(ABORT, 'a caption change needs a matching new caption_versions row'); END;
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def connect(path: Path) -> sqlite3.Connection:
    """One connection (open one per operation or per thread; they are cheap)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` … ``COMMIT``; rolls back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def is_seeded(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM meta WHERE key = 'as_of'").fetchone() is not None


def seed(conn: sqlite3.Connection, *, products: Iterable[Mapping[str, Any]], orders: Iterable[Mapping[str, Any]],
         reviews: Iterable[Mapping[str, Any]], stock: Mapping[str, int], as_of: str, data_version: str,
         actor: str = "system:seed") -> None:
    """Load one store's data into an empty database, with a ``seed`` caption version per product."""
    now = utc_now()
    with transaction(conn):
        if is_seeded(conn):
            raise RuntimeError("database is already seeded")
        for product in products:
            pid = product["product_id"]
            version = caption_version(product)
            conn.execute(
                "INSERT INTO products (product_id, data, name, description, tags, caption_version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (pid, dumps(dict(product)), product["name"], product["description"], dumps(product["tags"]),
                 version))
            cur = conn.execute(
                "INSERT INTO caption_versions (product_id, caption_version, name, description, tags, source, "
                "created_at, created_by) VALUES (?, ?, ?, ?, ?, 'seed', ?, ?)",
                (pid, version, product["name"], product["description"], dumps(product["tags"]), now, actor))
            # Not a caption change (same name/description/tags), so the trigger allows it.
            conn.execute("UPDATE products SET current_version_id = ? WHERE product_id = ?", (cur.lastrowid, pid))
            conn.execute("INSERT INTO inventory (product_id, stock) VALUES (?, ?)", (pid, stock[pid]))
        conn.executemany("INSERT INTO orders (order_id, date, data) VALUES (?, ?, ?)",
                         [(o["order_id"], o["date"], dumps(dict(o))) for o in orders])
        conn.executemany("INSERT INTO reviews (review_id, product_id, date, data) VALUES (?, ?, ?, ?)",
                         [(r["review_id"], r["product_id"], r["date"], dumps(dict(r))) for r in reviews])
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                         [("as_of", as_of), ("data_version", data_version), ("seeded_at", now)])
        conn.execute("INSERT INTO audit_log (at, actor, action, detail) VALUES (?, ?, 'seeded', ?)",
                     (now, actor, dumps({"data_version": data_version, "as_of": as_of})))


def seed_from_files(conn: sqlite3.Connection, catalog_path: Path, sim_dir: Path) -> None:
    """Seed from the committed demo data (``data/store`` catalog + ``data/seller_sim``)."""

    def jsonl(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (sim_dir / name).read_text(encoding="utf-8").splitlines() if line]

    manifest = json.loads((sim_dir / "manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((sim_dir / "inventory.json").read_text(encoding="utf-8"))
    seed(conn, products=json.loads(catalog_path.read_text(encoding="utf-8"))["products"],
         orders=jsonl("orders.jsonl"), reviews=jsonl("reviews.jsonl"),
         stock={row["product_id"]: row["stock"] for row in inventory},
         as_of=manifest["as_of"], data_version=manifest["data_version"])
