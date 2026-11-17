"""Fixtures for harness tests: a scripted chat model and seller dev tasks (no network)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_anthropic")

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted AIMessages in order, then a plain-text reply.

    ``calls`` records the messages of every invocation; ``bound_tools`` and
    ``call_kwargs`` record what the graph bound and passed.
    """

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    call_kwargs: list[dict[str, Any]] = []
    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        self.call_kwargs.append(kwargs)
        index = len(self.calls) - 1
        message = self.responses[index] if index < len(self.responses) else AIMessage("(no more script)")
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def scripted_model():
    def make(*responses: AIMessage) -> ScriptedChatModel:
        return ScriptedChatModel(responses=list(responses), calls=[], call_kwargs=[], bound_tools=[])

    return make


@pytest.fixture(scope="session")
def dev_tasks() -> dict[str, dict[str, Any]]:
    from sellerops.evaluation.seller_tasks import load_tasks

    return {task["task_id"]: task for task in load_tasks("dev")}
