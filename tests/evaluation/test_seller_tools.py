"""Seller tools as LangChain tools: schemas, recording, validation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sellerops.evaluation.seller_tools import SELLER_TOOL_NAMES, make_seller_tools
from sellerops.simulator import SellerEpisode


@pytest.fixture
def episode():
    return SellerEpisode()


def test_tool_names_match_the_episode(episode) -> None:
    tools = make_seller_tools(episode)
    assert [t.name for t in tools] == list(episode.tool_names)
    assert set(SELLER_TOOL_NAMES) == set(episode.tool_names)
    for tool in tools:
        assert tool.description and tool.args_schema.model_json_schema()["type"] == "object"


def test_call_goes_through_the_episode(episode) -> None:
    tools = {t.name: t for t in make_seller_tools(episode)}
    output = json.loads(tools["get_sales_summary"].invoke({"period": "last_7_days"}))
    assert output["ok"] and output["result"]["period"]["name"] == "last_7_days"
    step = episode.snapshot()["history"][0]
    assert step["tool"] == "get_sales_summary"
    assert step["arguments"] == {"period": "last_7_days", "compare_to_previous": False}


def test_business_error_is_an_observation(episode) -> None:
    tool = {t.name: t for t in make_seller_tools(episode)}["get_product"]
    output = json.loads(tool.invoke({"product_id": "FS999"}))
    assert output == {"ok": False, "error": {"code": "product_not_found", "message": "no product with id 'FS999'"}}


@pytest.mark.parametrize(("name", "args"), [
    ("get_sales_summary", {"period": "yesterday"}),
    ("get_sales_summary", {"period": "last_7_days", "extra": 1}),
    ("get_inventory_status", {"target_days": 0}),
    ("get_reviews", {"category": "hats"}),
])
def test_invalid_arguments_raise_before_recording(episode, name, args) -> None:
    tool = {t.name: t for t in make_seller_tools(episode)}[name]
    with pytest.raises(ValidationError):
        tool.invoke(args)
    assert episode.snapshot()["history"] == []
