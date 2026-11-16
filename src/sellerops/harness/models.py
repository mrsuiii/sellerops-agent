"""Chat model for the agent graph, built from ``get_settings()``.

What the framework does here:

- ``init_chat_model(..., model_provider="anthropic")`` returns a
  ``ChatAnthropic``. It converts LangChain messages to the Messages API,
  parses ``tool_use`` blocks into ``AIMessage.tool_calls``, and fills
  ``AIMessage.usage_metadata`` (``input_token_details.cache_read`` holds
  ``cache_read_input_tokens``).
- ``output_config={"effort": ...}`` is sent as-is. Claude Opus 5.5 always uses
  adaptive thinking, so ``thinking`` is left unset; effort is the only depth
  control. Thinking blocks stay in ``AIMessage.content`` and are replayed on
  the next call because the graph keeps the whole AIMessage in state.
- ``max_retries`` is the SDK's own retry (408/409/429/5xx, connection errors).
  Behavioral retries (the model producing a bad answer) belong in the graph.
- Prompt caching: pass ``cache_control=CACHE_CONTROL`` when invoking the
  model (``model.invoke(messages, cache_control=CACHE_CONTROL)`` or
  ``model.bind_tools(tools).bind(cache_control=CACHE_CONTROL)``). ChatAnthropic
  places the breakpoint on the last eligible block, so each call reuses the
  cached prefix (tools → system → earlier turns). ``cached_system_message``
  adds a second breakpoint on the system prompt itself.

Server-side refusal fallback (``fallbacks="default"``) is not wired up yet:
the fallback content blocks have not been checked against ChatAnthropic's
message conversion. Client-side ``with_fallbacks`` is planned for Session A2.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from ..settings import Settings, get_settings

CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def get_chat_model(settings: Settings | None = None, **overrides: Any) -> BaseChatModel:
    """Return the chat model selected by SELLEROPS_LLM_* settings.

    ``overrides`` are passed to the model constructor (e.g. ``max_tokens``,
    ``timeout``, ``max_retries``) and win over the defaults below.
    """
    settings = settings or get_settings()
    if settings.llm_provider != "anthropic":
        raise NotImplementedError(
            f"provider {settings.llm_provider!r} needs langchain-openai (not installed yet)"
        )
    kwargs: dict[str, Any] = {
        "max_tokens": 16000,
        "timeout": 120.0,
        "max_retries": 2,
        "output_config": {"effort": settings.llm_effort},
    }
    kwargs.update(overrides)
    return init_chat_model(settings.llm_model, model_provider="anthropic", **kwargs)


def cached_system_message(text: str) -> SystemMessage:
    """System prompt as a content block with a cache breakpoint."""
    return SystemMessage(content=[{"type": "text", "text": text, "cache_control": CACHE_CONTROL}])
