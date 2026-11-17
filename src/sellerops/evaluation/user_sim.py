"""User simulators: the seller on the other side of a multi-turn eval conversation.

Both implement ``reply(task, question, transcript) -> UserReply``:

- ``ScriptedUser``: deterministic, no network (tests, smoke runs). It reveals
  the hidden facts whose keywords appear in the agent's question; if none
  match, it reveals the next unrevealed fact (a cooperative seller), and once
  everything is revealed it says so.
- ``LLMUser``: a chat model role-playing the seller from the task's persona,
  goal and hidden info (τ-bench pattern). The model is injected, so tests can
  pass a scripted model and eval passes ``get_chat_model()``.

``transcript`` is the visible conversation so far as ``{"role": "user" |
"agent", "text": ...}`` items (no tool calls). The simulator never sees the
answer key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

STOP_TOKEN = "###STOP###"


@dataclass(frozen=True)
class UserReply:
    text: str
    stop: bool = False  # the simulated seller ends the conversation


class UserSimulator(Protocol):
    def reply(self, task: dict[str, Any], question: str, transcript: list[dict[str, str]]) -> UserReply: ...


@dataclass
class ScriptedUser:
    no_more_info: str = "Itu saja infonya, silakan lanjut."
    revealed: dict[str, set[int]] = field(default_factory=dict)

    def reply(self, task: dict[str, Any], question: str, transcript: list[dict[str, str]]) -> UserReply:
        facts = task["hidden_info"]
        seen = self.revealed.setdefault(task["task_id"], set())
        text = question.lower()
        matched = [i for i, fact in enumerate(facts)
                   if i not in seen and any(word in text for word in fact["keywords"])]
        if not matched:
            matched = [i for i in range(len(facts)) if i not in seen][:1]
        if not matched:
            return UserReply(self.no_more_info)
        seen.update(matched)
        return UserReply(" ".join(facts[i]["fact"] for i in matched))


USER_SYSTEM_PROMPT = """\
You are role-playing a seller who is chatting with an analytics assistant \
about your online store. Stay in character; you are the human, not the assistant.

Persona: {persona}
What you want from this conversation: {goal}
Details you know but only share when the assistant asks about them:
{hidden}

Rules:
- Reply in one or two short sentences, in the language of your persona.
- Share a detail only when the assistant's question is about it. Never invent \
details that are not listed; if asked something you do not know, say you do \
not know.
- Do not solve the task, do not mention numbers or product ids yourself.
- If the assistant has nothing left to ask and your goal is met, or the \
conversation goes in circles, reply with {stop} only."""


class LLMUser:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def system_prompt(self, task: dict[str, Any]) -> str:
        hidden = "\n".join(f"- {fact['fact']}" for fact in task["hidden_info"]) or "- (none)"
        return USER_SYSTEM_PROMPT.format(persona=task["persona"], goal=task["goal"], hidden=hidden,
                                         stop=STOP_TOKEN)

    def messages(self, task: dict[str, Any], question: str,
                 transcript: list[dict[str, str]]) -> list[BaseMessage]:
        # Roles are flipped: the simulated seller is the model ("assistant"),
        # the agent's words arrive as "user" messages.
        messages: list[BaseMessage] = [SystemMessage(self.system_prompt(task))]
        for item in transcript:
            messages.append(AIMessage(item["text"]) if item["role"] == "user" else HumanMessage(item["text"]))
        messages.append(HumanMessage(question))
        return messages

    def reply(self, task: dict[str, Any], question: str, transcript: list[dict[str, str]]) -> UserReply:
        response = self.model.invoke(self.messages(task, question, transcript))
        text = response.text if isinstance(response.text, str) else str(response.content)
        text = text.strip()
        if STOP_TOKEN in text:
            return UserReply(text.replace(STOP_TOKEN, "").strip(), stop=True)
        return UserReply(text)
