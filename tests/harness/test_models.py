"""Chat model factory: request payload built from settings (no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_anthropic")

from langchain_core.messages import HumanMessage  # noqa: E402

from sellerops.evaluation.seller_tools import make_seller_tools  # noqa: E402
from sellerops.evaluation.terminal_tools import TERMINAL_TOOLS  # noqa: E402
from sellerops.harness.models import CACHE_CONTROL, cached_system_message, get_chat_model  # noqa: E402
from sellerops.settings import get_settings  # noqa: E402
from sellerops.simulator import SellerEpisode  # noqa: E402


def test_chat_model_request_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELLEROPS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SELLEROPS_LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("SELLEROPS_LLM_EFFORT", "high")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    get_settings.cache_clear()
    try:
        model = get_chat_model()
    finally:
        get_settings.cache_clear()

    seller_tools = make_seller_tools(SellerEpisode())
    bound = model.bind_tools([*seller_tools, *TERMINAL_TOOLS])
    payload = model._get_request_payload(
        [cached_system_message("SYSTEM"), HumanMessage("hi")],
        **bound.kwargs, cache_control=CACHE_CONTROL,
    )
    assert payload["model"] == "claude-haiku-5-5"
    assert payload["output_config"] == {"effort": "high"}
    assert "thinking" not in payload or payload["thinking"].get("type") == "adaptive"
    assert payload["system"][0]["cache_control"] == CACHE_CONTROL
    names = [tool["name"] for tool in payload["tools"]]
    assert names == ["get_sales_summary", "get_product_performance", "get_inventory_status", "get_reviews",
                     "get_product", "propose_caption_update", "final", "ask_user"]
    assert "tool_choice" not in payload  # forced tool choice is rejected by Opus 5.5
