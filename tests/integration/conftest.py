"""Integration tests against a real LLM: opt-in, because every run costs money.

    make test-live            # = SELLEROPS_LIVE_TESTS=1 pytest -m live -s tests/integration

The model comes from the usual settings (``SELLEROPS_LLM_*``, keys from
``.env``). Assertions check properties (which fields changed, length, that
a version differs), never exact wording, so a different but valid answer
still passes. Traces are tagged ``live-integration`` in LangSmith.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("SELLEROPS_LIVE_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    if LIVE:
        return
    skip = pytest.mark.skip(reason="live LLM test: run with SELLEROPS_LIVE_TESTS=1 (make test-live)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def live_model():
    from sellerops.harness.models import get_chat_model
    from sellerops.settings import get_settings

    settings = get_settings()
    if settings.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set (.env)")
    model = get_chat_model(settings).with_config(tags=["live-integration"])
    print(f"\n[live] provider={settings.llm_provider} model={settings.llm_model}")
    return model
