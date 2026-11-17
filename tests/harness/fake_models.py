"""Fake chat models for harness tests (imported by test modules; not a test file)."""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ids = itertools.count(1)


class APIError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeModel(BaseChatModel):
    """Plays a script of AIMessages / exceptions; records prompts and kwargs."""

    script: list[Any]
    calls: list[list[BaseMessage]] = []
    kwargs: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        self.kwargs.append(kwargs)
        item = self.script[len(self.calls) - 1] if len(self.calls) <= len(self.script) else AIMessage("(end)")
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def fake(*script: Any) -> FakeModel:
    return FakeModel(script=list(script), calls=[], kwargs=[])


def call(name: str, args: dict[str, Any], *, tokens: int = 0) -> AIMessage:
    usage = {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens} if tokens else None
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{next(_ids)}"}], usage_metadata=usage)
