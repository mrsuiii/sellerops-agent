"""The production seller tools: harness tools → InProcessSellerClient → SellerService."""

from __future__ import annotations

import json

import pytest

from sellerops.harness.tools import SELLER_TOOL_NAMES, make_seller_tools
from sellerops.seller import InProcessSellerClient
from sellerops.simulator import SellerEpisode


@pytest.fixture
def tools(service):
    return {t.name: t for t in make_seller_tools(InProcessSellerClient(service))}


def test_client_serves_exactly_the_seller_tools(tools) -> None:
    assert set(tools) == SELLER_TOOL_NAMES
    assert not {"approve", "approve_caption", "revert"} & set(tools)  # the agent cannot apply captions


@pytest.mark.parametrize(("name", "args"), [
    ("get_sales_summary", {"period": "last_30_days", "compare_to_previous": True}),
    ("get_product_performance", {"period": "last_7_days", "sort": "units_asc", "category": "bags"}),
    ("get_inventory_status", {"limit": 3}),
    ("get_reviews", {"max_rating": 2, "limit": 5}),
    ("get_product", {"product_id": "FS180"}),
    ("get_product", {"product_id": "NOPE"}),
    ("propose_caption_update", {"product_id": "FS180", "reason": "clearer", "description": "Clearer."}),
    ("propose_caption_update", {"product_id": "FS180", "reason": "same", "name": "Dress Pea"}),
])
def test_production_observations_match_the_simulator(tools, name, args) -> None:
    episode_tools = {t.name: t for t in make_seller_tools(SellerEpisode())}
    assert json.loads(tools[name].invoke(args)) == json.loads(episode_tools[name].invoke(args))


def test_agent_proposal_is_pending_and_recorded_as_the_agent(service, tools) -> None:
    before = service.analytics().product("FS180")
    observation = json.loads(tools["propose_caption_update"].invoke(
        {"product_id": "FS180", "reason": "clearer", "tags": ["dress", "women"]}))
    assert observation["ok"] and observation["result"]["status"] == "pending"
    assert service.analytics().product("FS180") == before
    entry = service.audit_log(proposal_id=observation["result"]["proposal_id"])[0]
    assert (entry["action"], entry["actor"]) == ("proposed", "agent")


def test_unknown_tool_is_a_bug_not_an_observation(service) -> None:
    with pytest.raises(KeyError):
        InProcessSellerClient(service).call("approve_caption", {})
