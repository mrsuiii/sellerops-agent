"""Free stand-in for the chat model (``SELLEROPS_APPROVAL_FAKE=1``, ``make store-fake``).

It never calls a tool; it echoes the question, so the UI and the thread can be
tried without an API key or cost.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        question = str(messages[-1].content) if messages else ""
        text = f"(fake agent) Kamu bertanya: “{question[:200]}”. Model asli tidak dipakai di mode fake."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])
