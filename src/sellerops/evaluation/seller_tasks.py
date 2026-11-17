"""Multi-turn seller eval tasks (τ-bench style), with answer keys computed from the data.

A task is what the *user simulator* knows plus what the *verifier* checks:

- ``opening_message``: the seller's first message (all the agent sees);
- ``persona`` and ``goal``: how the simulated seller behaves and what it wants;
- ``hidden_info``: details the seller only gives when asked (``fact`` text plus
  ``keywords`` the scripted user matches against the agent's question);
- ``expected``: the answer key, computed from ``SellerStore`` so it
  is always consistent with the dataset (never typed by hand); it also keeps
  the parameters used (period, category, ...) for reference solutions;
- ``needs_clarification``: the opening message leaves out a needed detail, so a
  good agent asks before answering (reported, not a pass/fail condition).

Splits: ``dev`` for iterating on prompts and the graph, ``test`` frozen (its
checksum is in ``tasks.manifest.json``; do not tune on it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..schemas import canonical_json
from ..simulator.store import SellerStore, SimData

TASKS_VERSION = "seller-tasks-v1"
RESTOCK_DAYS = 7
NEGATIVE_MAX_RATING = 2
COMPLAINT_KEYWORDS = {
    "size_runs_small": ["ukuran", "kecil", "kekecilan", "sempit", "size", "small", "tight"],
}

PERSONA = ("Pemilik toko Kala Studio. Santai, menulis singkat dalam Bahasa Indonesia, "
           "tidak paham istilah teknis. Hanya menjawab yang ditanya.")

_PERIOD_KEYWORDS = ["periode", "kapan", "rentang", "jangka", "waktu", "hari", "minggu", "bulan", "period"]
_CATEGORY_KEYWORDS = ["kategori", "jenis", "produk apa", "category", "tipe"]
_FIELD_KEYWORDS = ["bagian", "field", "yang mana", "apa saja", "deskripsi", "nama", "tag", "ubah"]
_PERIOD_TEXT = {
    "last_7_days": "7 hari terakhir",
    "previous_7_days": "7 hari sebelum minggu ini",
    "last_30_days": "30 hari terakhir",
    "previous_30_days": "30 hari sebelum itu",
    "last_90_days": "90 hari terakhir",
}


def _hidden_period(period: str) -> dict[str, Any]:
    return {"fact": f"Periode yang dimaksud: {_PERIOD_TEXT[period]}.", "keywords": _PERIOD_KEYWORDS}


def _hidden_category(label: str) -> dict[str, Any]:
    return {"fact": f"Kategorinya {label}.", "keywords": _CATEGORY_KEYWORDS}


# ------------------------------------------------------------ answer keys


def _sales_totals(store: SellerStore, period: str) -> dict[str, Any]:
    summary = store.sales_summary(period)
    return {"period": period, "revenue_idr": summary["revenue_idr"], "orders": summary["orders"]}


def _top_category(store: SellerStore, period: str) -> dict[str, Any]:
    groups = store.sales_summary(period, "category")["groups"]
    if len(groups) > 1 and groups[0]["revenue_idr"] == groups[1]["revenue_idr"]:
        raise ValueError("tie for the top category")
    return {"period": period, "category": groups[0]["key"]}


def _ranking(store: SellerStore, period: str, n: int, category: str | None = None) -> dict[str, Any]:
    rows = store.product_performance(period, "units_desc", 54, category)["products"]
    if rows[n - 1]["units"] == rows[n]["units"]:
        raise ValueError(f"tie at rank {n} ({category}, {period})")
    return {"period": period, "category": category, "product_ids": [r["product_id"] for r in rows[:n]]}


def _no_sales(store: SellerStore, period: str, category: str) -> dict[str, Any]:
    rows = store.product_performance(period, "units_asc", 54, category)["products"]
    ids = sorted(r["product_id"] for r in rows if r["units"] == 0)
    if not ids:
        raise ValueError(f"no zero-sales product in {category}")
    return {"period": period, "category": category, "product_ids": ids}


def _restock(store: SellerStore, category: str | None = None) -> dict[str, Any]:
    rows = store.inventory_status(category=category, limit=54)["products"]
    ids = [r["product_id"] for r in rows if r["days_to_stockout"] is not None and r["days_to_stockout"] <= RESTOCK_DAYS]
    return {"category": category, "max_days_to_stockout": RESTOCK_DAYS, "product_ids": sorted(ids)}


def _complaint(store: SellerStore, period: str, category: str | None, issue: str) -> dict[str, Any]:
    reviews = store.reviews(category=category, max_rating=NEGATIVE_MAX_RATING, period=period, limit=50)["reviews"]
    counts: dict[str, int] = {}
    for review in reviews:
        counts[review["product_id"]] = counts.get(review["product_id"], 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError("tie for the most complained-about product")
    return {"period": period, "category": category, "product_id": ranked[0][0], "issue": issue,
            "complaint_keywords": COMPLAINT_KEYWORDS[issue]}


def _spike(store: SellerStore, category: str | None = None) -> dict[str, Any]:
    now, before = store.units_by_product("last_7_days"), store.units_by_product("previous_7_days")
    rows = [(now[pid] - before[pid], pid) for pid in now
            if category is None or store.product(pid)["category"] == category]
    rows.sort(key=lambda r: (-r[0], r[1]))
    if rows[0][0] == rows[1][0]:
        raise ValueError("tie for the biggest spike")
    return {"period": "last_7_days", "product_id": rows[0][1]}


def _drop(store: SellerStore, category: str, audience: str | None) -> dict[str, Any]:
    now = store.product_performance("last_30_days", "revenue_desc", 54, category, audience)["products"]
    before = {r["product_id"]: r["revenue_idr"] for r in
              store.product_performance("previous_30_days", "revenue_desc", 54, category, audience)["products"]}
    rows = sorted((r["revenue_idr"] - before[r["product_id"]], r["product_id"], r["stock"]) for r in now)
    delta, product_id, stock = rows[0]
    if delta >= 0:
        raise ValueError("no revenue drop")
    return {"category": category, "audience": audience, "product_id": product_id,
            "cause": "out_of_stock" if stock == 0 else "other"}


def _caption(store: SellerStore, product_name: str, required: list[str]) -> dict[str, Any]:
    matches = [p for p in store.data.catalog["products"] if p["name"] == product_name]
    if len(matches) != 1:
        raise ValueError(f"product name {product_name!r} is not unique")
    return {"product_id": matches[0]["product_id"], "required_fields": required,
            "allowed_fields": ["name", "description", "tags"]}


# ----------------------------------------------------------------- tasks


def build_tasks(data: SimData) -> list[dict[str, Any]]:
    """All dev and test tasks for one dataset (deterministic)."""
    store = SellerStore(data)
    specs: list[dict[str, Any]] = [
        # ---- dev
        dict(split="dev", family="sales_summary", slug="toko-tanpa-periode",
             opening_message="Gimana penjualan toko?",
             hidden_info=[_hidden_period("last_7_days")],
             goal="Tahu omzet dan jumlah order 7 hari terakhir.",
             expected=_sales_totals(store, "last_7_days"), needs_clarification=True),
        dict(split="dev", family="sales_summary", slug="omzet-30-hari",
             opening_message="Berapa omzet dan jumlah order 30 hari terakhir?",
             hidden_info=[], goal="Tahu omzet dan jumlah order 30 hari terakhir.",
             expected=_sales_totals(store, "last_30_days"), needs_clarification=False),
        dict(split="dev", family="top_category", slug="kategori-omzet-terbesar",
             opening_message="Kategori apa yang omzetnya paling besar 30 hari terakhir?",
             hidden_info=[], goal="Tahu kategori dengan omzet terbesar.",
             expected=_top_category(store, "last_30_days"), needs_clarification=False),
        dict(split="dev", family="top_products", slug="sepatu-terlaris",
             opening_message="Tiga sepatu apa yang paling laku?",
             hidden_info=[_hidden_period("last_30_days")],
             goal="Tahu tiga sepatu terlaris (berdasarkan unit terjual) 30 hari terakhir.",
             expected=_ranking(store, "last_30_days", 3, "shoes"), needs_clarification=True),
        dict(split="dev", family="slow_movers", slug="tas-tidak-laku",
             opening_message="Produk mana yang nggak laku sama sekali?",
             hidden_info=[_hidden_category("tas (bags)"), _hidden_period("last_30_days")],
             goal="Tahu produk tas yang tidak terjual sama sekali 30 hari terakhir.",
             expected=_no_sales(store, "last_30_days", "bags"), needs_clarification=True),
        dict(split="dev", family="restock", slug="restock-semua",
             opening_message="Apa saja yang harus segera di-restock (habis dalam seminggu)?",
             hidden_info=[], goal="Tahu produk yang stoknya habis dalam 7 hari.",
             expected=_restock(store), needs_clarification=False),
        dict(split="dev", family="review_insight", slug="keluhan-sepatu",
             opening_message="Pembeli ngeluh apa soal sepatu bulan ini?",
             hidden_info=[], goal="Tahu sepatu mana yang paling banyak dikeluhkan dan keluhannya apa.",
             expected=_complaint(store, "last_30_days", "shoes", "size_runs_small"),
             needs_clarification=False),
        dict(split="dev", family="sales_spike", slug="lonjakan-minggu-ini",
             opening_message="Ada produk yang tiba-tiba laku banget minggu ini?",
             hidden_info=[], goal="Tahu produk dengan kenaikan unit terbesar dibanding minggu sebelumnya.",
             expected=_spike(store), needs_clarification=False),
        dict(split="dev", family="sales_drop", slug="sepatu-wanita-turun",
             opening_message="Kenapa penjualan sepatu wanita turun bulan ini?",
             hidden_info=[], goal="Tahu produk penyebab penurunan dan alasannya.",
             expected=_drop(store, "shoes", "women"), needs_clarification=False),
        dict(split="dev", family="caption", slug="caption-dress-pea",
             opening_message="Perbaiki caption Dress Pea dong.",
             hidden_info=[{"fact": "Yang diperbaiki deskripsinya saja, nama dan tag jangan diubah.",
                           "keywords": _FIELD_KEYWORDS}],
             goal="Dapat usulan deskripsi baru untuk Dress Pea yang menunggu persetujuan.",
             expected=_caption(store, "Dress Pea", ["description"]), needs_clarification=True),
        # ---- test (frozen)
        dict(split="test", family="sales_summary", slug="rekap-tanpa-periode",
             opening_message="Tolong rekap penjualan.",
             hidden_info=[_hidden_period("last_30_days")],
             goal="Tahu omzet dan jumlah order 30 hari terakhir.",
             expected=_sales_totals(store, "last_30_days"), needs_clarification=True),
        dict(split="test", family="sales_summary", slug="omzet-7-hari",
             opening_message="Omzet 7 hari terakhir berapa? Ordernya berapa?",
             hidden_info=[], goal="Tahu omzet dan jumlah order 7 hari terakhir.",
             expected=_sales_totals(store, "last_7_days"), needs_clarification=False),
        dict(split="test", family="top_category", slug="kategori-90-hari",
             opening_message="Kategori mana yang paling banyak menyumbang omzet?",
             hidden_info=[_hidden_period("last_90_days")], goal="Tahu kategori dengan omzet terbesar.",
             expected=_top_category(store, "last_90_days"), needs_clarification=True),
        dict(split="test", family="top_products", slug="dress-terlaris",
             opening_message="Dua dress terlaris apa saja?",
             hidden_info=[_hidden_period("last_30_days")],
             goal="Tahu dua dress terlaris (unit terjual) 30 hari terakhir.",
             expected=_ranking(store, "last_30_days", 2, "dresses"), needs_clarification=True),
        dict(split="test", family="slow_movers", slug="kacamata-tidak-laku",
             opening_message="Kacamata apa yang nggak laku sama sekali 30 hari terakhir?",
             hidden_info=[], goal="Tahu kacamata yang tidak terjual 30 hari terakhir.",
             expected=_no_sales(store, "last_30_days", "sunglasses"), needs_clarification=False),
        dict(split="test", family="restock", slug="restock-dress",
             opening_message="Produk mana yang perlu restock segera?",
             hidden_info=[_hidden_category("dress (dresses)")],
             goal="Tahu dress yang stoknya habis dalam 7 hari.",
             expected=_restock(store, "dresses"), needs_clarification=True),
        dict(split="test", family="review_insight", slug="keluhan-berulang",
             opening_message="Ada keluhan yang berulang di review 30 hari terakhir?",
             hidden_info=[], goal="Tahu produk dengan keluhan berulang dan isi keluhannya.",
             expected=_complaint(store, "last_30_days", None, "size_runs_small"),
             needs_clarification=False),
        dict(split="test", family="sales_drop", slug="sepatu-turun",
             opening_message="Omzet sepatu wanita kok turun ya? Kenapa?",
             hidden_info=[], goal="Tahu produk penyebab penurunan dan alasannya.",
             expected=_drop(store, "shoes", "women"), needs_clarification=False),
        dict(split="test", family="caption", slug="caption-gray-dress",
             opening_message="Tolong rapikan caption Gray Dress.",
             hidden_info=[{"fact": "Yang diubah deskripsi dan tag-nya; namanya biarkan.",
                           "keywords": _FIELD_KEYWORDS}],
             goal="Dapat usulan deskripsi dan tag baru untuk Gray Dress yang menunggu persetujuan.",
             expected=_caption(store, "Gray Dress", ["description", "tags"]), needs_clarification=True),
    ]
    tasks = []
    for index, spec in enumerate(specs):
        slug = spec.pop("slug")
        tasks.append({
            "task_id": f"seller-{spec['split']}-{spec['family'].replace('_', '-')}-{slug}",
            "index": index,
            "persona": PERSONA,
            "data_version": data.manifest.get("data_version"),
            **spec,
        })
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task ids")
    return tasks


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_files(data: SimData) -> dict[str, str]:
    tasks = build_tasks(data)
    files: dict[str, str] = {}
    for split in ("dev", "test"):
        files[f"tasks.{split}.jsonl"] = "".join(canonical_json(t) + "\n" for t in tasks if t["split"] == split)
    manifest = {
        "tasks_version": TASKS_VERSION,
        "data_version": data.manifest.get("data_version"),
        "data_files": data.manifest.get("files"),
        "counts": {split: sum(t["split"] == split for t in tasks) for split in ("dev", "test")},
        "families": sorted({t["family"] for t in tasks}),
        "files": {name: _sha256(text) for name, text in sorted(files.items())},
        "frozen": ["tasks.test.jsonl"],
    }
    files["tasks.manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
    return files


def write_tasks(data: SimData, out_dir: Path) -> dict[str, Any]:
    files = task_files(data)
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    return json.loads(files["tasks.manifest.json"])


def load_tasks(split: str = "dev", sim_dir: Path | None = None) -> list[dict[str, Any]]:
    from ..simulator.store import default_sim_dir

    path = (sim_dir or default_sim_dir()) / f"tasks.{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
