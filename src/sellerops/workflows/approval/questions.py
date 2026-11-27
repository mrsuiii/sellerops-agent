"""Asking the seller (C3b): the contract between the graph, the runner and the UI.

When the product data and the photos do not hold a fact the caption needs
(material, size, care…), the graph may stop and ask the seller instead of
guessing. It does so with an ``interrupt`` whose value is built by
``question_request``::

    answer = interrupt(question_request([SellerQuestion(key="material", question="Bahannya apa?")],
                                        product_id=..., request_id=..., proposal_id=...))
    # the runner resumes with Answer: {"answers": {"material": "katun" | None}, "user": "seller-1"}

``kind`` tells the runner which step is waiting: ``"question"`` here,
``"decision"`` for ``await_decision`` (a payload without ``kind`` counts as a
decision, so the C3 graph keeps working unchanged).

The runner, not the graph, stores the answers as product facts
(``SellerService.remember_facts``, source = the request or proposal id)
before it resumes: the seller's words are kept even if the next step fails,
and the graph reads them back from the service like any other fact.
``None`` means the seller does not know; the caption should then say that
more product data is needed rather than invent it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sellerops.seller.service import FACT_KEY, MAX_FACT_LENGTH

MAX_QUESTIONS = 3
KEY_PATTERN = FACT_KEY.pattern


class SellerQuestion(BaseModel):
    """One question; ``key`` names the fact the answer is stored under (snake_case)."""

    key: str = Field(pattern=KEY_PATTERN, description='Fact name, e.g. "material", "fit", "care".')
    question: str = Field(min_length=1, max_length=300, description="The question, in the seller's language.")


def question_request(questions: list[SellerQuestion], *, product_id: str, request_id: str | None = None,
                     proposal_id: str | None = None, error: Any = None) -> dict[str, Any]:
    """The ``interrupt`` value for a question step (JSON-serialisable)."""
    payload: dict[str, Any] = {
        "kind": "question",
        "product_id": product_id,
        "request_id": request_id,
        "proposal_id": proposal_id,
        "questions": [q.model_dump() for q in questions[:MAX_QUESTIONS]],
    }
    if error is not None:
        payload["error"] = error
    return payload


class AnswerIn(BaseModel):
    """The seller's answers (API body). Every question asked must be answered; "" or null = does not know."""

    model_config = ConfigDict(extra="forbid")

    answers: dict[str, str | None] = Field(min_length=1)

    @field_validator("answers")
    @classmethod
    def _clean(cls, answers: dict[str, str | None]) -> dict[str, str | None]:
        cleaned: dict[str, str | None] = {}
        for key, value in answers.items():
            if not FACT_KEY.match(key):
                raise ValueError(f"answer key {key!r} is not a fact key")
            text = (value or "").strip()
            if len(text) > MAX_FACT_LENGTH:
                raise ValueError(f"answer to {key!r} is longer than {MAX_FACT_LENGTH} characters")
            cleaned[key] = text or None
        return cleaned


class Answer(AnswerIn):
    """What the graph receives in ``Command(resume=...)``; check it again there (it crosses a boundary)."""

    user: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@-]+$")
