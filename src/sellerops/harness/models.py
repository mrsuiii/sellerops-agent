"""Chat model for the agent graph, built from ``get_settings()``.

What the framework does here:

- ``init_chat_model(..., model_provider="anthropic")`` returns a
  ``ChatAnthropic``. It converts LangChain messages to the Messages API,
  parses ``tool_use`` blocks into ``AIMessage.tool_calls``, and fills
  ``AIMessage.usage_metadata`` (``input_token_details.cache_read`` holds
  ``cache_read_input_tokens``).
- ``output_config={"effort": ...}`` is sent as-is. The default model, Claude
  Haiku 5.5 (and Opus 5.5), runs adaptive thinking by default, so
  ``thinking`` is left unset; effort is the only depth control. Do not send
  ``temperature`` / ``top_p`` / ``top_k``: Haiku 5.5 rejects non-default values. Thinking blocks stay in ``AIMessage.content`` and are replayed on
  the next call because the graph keeps the whole AIMessage in state.
- ``max_retries`` is the SDK's own retry (408/409/429/5xx, connection errors).
  Behavioral retries (the model producing a bad answer) belong in the graph.
- Prompt caching: pass ``cache_control=CACHE_CONTROL`` when invoking the
  model (``model.invoke(messages, cache_control=CACHE_CONTROL)`` or
  ``model.bind_tools(tools).bind(cache_control=CACHE_CONTROL)``). ChatAnthropic
  places the breakpoint on the last eligible block, so each call reuses the
  cached prefix (tools → system → earlier turns). ``cached_system_message``
  adds a second breakpoint on the system prompt itself.

- Providers: ``anthropic`` (default) → ``ChatAnthropic``;
  ``openai_compatible`` → ``ChatOpenAI`` on ``LLM_BASE_URL`` (NVIDIA by
  default) with ``LLM_API_KEY`` / ``LLM_MODEL``. Prompt caching
  (``cache_control``) is Anthropic-only: ``supports_prompt_caching(model)``
  tells the graph whether to send it.
- Fallbacks (Session A2): ``get_fallback_models()`` builds the models listed
  in ``SELLEROPS_LLM_FALLBACK_MODELS`` (same provider as the primary, e.g.
  Sonnet 5.5 behind Haiku 5.5). The graph wraps them with ``with_fallbacks``;
  see ``harness/errors.py`` for how retries and fallbacks combine.

Server-side refusal fallback (``fallbacks="default"``) is not wired up: the
fallback content blocks have not been checked against ChatAnthropic's message
conversion.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from ..settings import Settings, get_settings

CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def get_chat_model(settings: Settings | None = None, model: str | None = None,
                   **overrides: Any) -> BaseChatModel:
    """Return the chat model selected by SELLEROPS_LLM_* settings.

    ``model`` replaces the configured model id (same provider); ``overrides``
    are passed to the model constructor (e.g. ``max_tokens``, ``timeout``,
    ``max_retries``) and win over the defaults below.
    """
    settings = settings or get_settings()
    model_id = model or settings.llm_model
    kwargs: dict[str, Any] = {"max_tokens": 16000, "timeout": 120.0, "max_retries": 2}
    if settings.llm_provider == "anthropic":
        kwargs["output_config"] = {"effort": settings.llm_effort}
        kwargs.update(overrides)
        return init_chat_model(model_id, model_provider="anthropic", **kwargs)
    if not model_id:
        raise ValueError("Set LLM_MODEL to a model id for the openai_compatible provider")
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set LLM_API_KEY for the openai_compatible provider")
    kwargs.update(base_url=settings.llm_base_url, api_key=api_key, use_responses_api=False)
    kwargs.update(overrides)
    return init_chat_model(model_id, model_provider="openai", **kwargs)


def get_fallback_models(settings: Settings | None = None, **overrides: Any) -> list[BaseChatModel]:
    """Models from SELLEROPS_LLM_FALLBACK_MODELS, in order (empty when unset)."""
    settings = settings or get_settings()
    return [get_chat_model(settings, model=name, **overrides) for name in settings.llm_fallback_models]


def supports_prompt_caching(model: Any) -> bool:
    """Only ChatAnthropic understands ``cache_control`` (blocks and invoke kwarg)."""
    return getattr(model, "_llm_type", "") == "anthropic-chat"


def supports_vision(model: Any) -> bool:
    """Whether image content blocks can go to this model (C3b); callers fall back to text only.

    Every Claude model reads images. The OpenAI-compatible endpoint serves many
    models, most text only, so it counts only when ``SELLEROPS_LLM_VISION=1``
    says the configured one does (``Settings.llm_vision``). Wrapped models (``with_config``,
    ``with_fallbacks``) are unwrapped; a fallback chain counts only if every
    model in it does.
    """
    bound = getattr(model, "bound", None)
    if bound is not None:                      # RunnableBinding (with_config / bind)
        return supports_vision(bound)
    runnables = [getattr(model, "runnable", None), *getattr(model, "fallbacks", ())]
    if runnables[0] is not None:               # RunnableWithFallbacks
        return all(supports_vision(item) for item in runnables)
    kind = getattr(model, "_llm_type", "")
    if kind == "anthropic-chat":
        return True
    if kind == "openai-chat":
        return get_settings().llm_vision
    return bool(getattr(model, "supports_vision", False))   # fakes in tests opt in


def cached_system_message(text: str) -> SystemMessage:
    """System prompt as a content block with a cache breakpoint."""
    return SystemMessage(content=[{"type": "text", "text": text, "cache_control": CACHE_CONTROL}])
