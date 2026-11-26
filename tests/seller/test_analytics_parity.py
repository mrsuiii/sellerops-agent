"""Production (SQLite) and the simulator (JSON) run the same analytics: same answers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sellerops.simulator import load_store

CASES = [
    ("sales_summary", {"period": "last_30_days"}),
    ("sales_summary", {"period": "last_7_days", "group_by": "category", "compare_to_previous": True}),
    ("sales_summary", {"period": "last_90_days", "group_by": "week"}),
    ("product_performance", {"period": "last_7_days", "limit": 54, "compare_to_previous": True}),
    ("product_performance", {"period": "last_30_days", "sort": "revenue_asc", "category": "dresses"}),
    ("inventory_status", {"limit": 54}),
    ("inventory_status", {"category": "shoes", "target_days": 30}),
    ("reviews", {"max_rating": 2, "limit": 50}),
    ("reviews", {"product_id": "FS180"}),
    ("product", {"product_id": "FS180"}),
]


@pytest.mark.parametrize(("method", "kwargs"), CASES)
def test_service_matches_simulator(service, method, kwargs) -> None:
    assert getattr(service.analytics(), method)(**kwargs) == getattr(load_store(), method)(**kwargs)


def test_seller_package_imports_no_eval_harness_or_simulator() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "sellerops" / "seller"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                name = node.module.lstrip(".").removeprefix("sellerops.")
                assert not name.startswith(("harness", "evaluation", "simulator")), path
