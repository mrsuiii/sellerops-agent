"""Production seller tools against a fake client (no simulator)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sellerops.harness.tools import SELLER_TOOL_NAMES, make_seller_tools


class FakeClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.response = response or {"ok": True, "result": {"x": 1}}

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, arguments))
        return self.response


def tools_by_name(client: FakeClient) -> dict[str, Any]:
    return {t.name: t for t in make_seller_tools(client)}


def test_six_tools_with_object_schemas() -> None:
    tools = make_seller_tools(FakeClient())
    assert {t.name for t in tools} == SELLER_TOOL_NAMES and len(tools) == 6
    for tool in tools:
        schema = tool.args_schema.model_json_schema()
        assert tool.description and schema["type"] == "object"
        assert schema.get("additionalProperties") is False


def test_client_gets_arguments_with_defaults_and_no_nones() -> None:
    client = FakeClient()
    output = tools_by_name(client)["get_reviews"].invoke({"product_id": "FS188", "max_rating": 2})
    assert json.loads(output) == {"ok": True, "result": {"x": 1}}
    assert client.calls == [("get_reviews", {"product_id": "FS188", "max_rating": 2,
                                             "period": "last_90_days", "limit": 20})]


def test_business_error_is_returned_as_observation() -> None:
    error = {"ok": False, "error": {"code": "product_not_found", "message": "no product with id 'FS999'"}}
    output = tools_by_name(FakeClient(error))["get_product"].invoke({"product_id": "FS999"})
    assert json.loads(output) == error


@pytest.mark.parametrize(("name", "args"), [
    ("get_sales_summary", {"period": "yesterday"}),
    ("get_sales_summary", {"period": "last_7_days", "store_id": "other"}),
    ("get_product_performance", {"period": "last_7_days", "limit": 0}),
    ("propose_caption_update", {"product_id": "FS1"}),
])
def test_invalid_arguments_never_reach_the_client(name, args) -> None:
    client = FakeClient()
    with pytest.raises(ValidationError):
        tools_by_name(client)[name].invoke(args)
    assert client.calls == []


def test_harness_does_not_import_eval_or_simulator() -> None:
    root = Path(__file__).parents[2] / "src" / "sellerops" / "harness"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = ("." * node.level) + (node.module or "")
                assert "evaluation" not in module and "simulator" not in module, path
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("sellerops.evaluation", "sellerops.simulator")), path
