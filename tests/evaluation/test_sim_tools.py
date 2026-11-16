"""Eval simulator tools run through EpisodeService (no network)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sellerops.evaluation.sim_tools import SIM_TOOL_NAMES, make_sim_tools
from sellerops.simulator.episode_service import EpisodeService


@pytest.fixture
def episode():
    service = EpisodeService()
    state = service.create_episode("test-task")
    tools = {tool.name: tool for tool in make_sim_tools(service, state.session_id)}
    return service, state, tools


def test_sim_tools_run_through_episode_service(episode) -> None:
    service, state, tools = episode
    assert set(tools) == SIM_TOOL_NAMES

    observation = json.loads(tools["search_products"].invoke(
        {"query": "backpack", "filters": {"brand": "Comet", "in_stock": True}}
    ))
    assert observation["ok"] is True
    assert observation["result"]["products"][0]["product_id"] == "P1008"
    # Unset filters are dropped before reaching the simulator; the call is recorded.
    assert state.history[0]["action"] == {
        "type": "tool_call", "tool": "search_products",
        "arguments": {"query": "backpack", "filters": {"brand": "Comet", "in_stock": True}},
    }
    assert state.step_count == 1


def test_business_failure_is_an_observation(episode) -> None:
    _, _, tools = episode
    observation = json.loads(tools["get_inventory"].invoke({"product_id": "P9999"}))
    assert observation["ok"] is False and observation["error"]["code"] == "product_not_found"


def test_invalid_arguments_raise_before_execution(episode) -> None:
    _, state, tools = episode
    with pytest.raises(ValidationError):
        tools["calculate_offer"].invoke({"product_id": "P1008", "discount_percent": 150, "quantity": 1})
    assert state.history == []
