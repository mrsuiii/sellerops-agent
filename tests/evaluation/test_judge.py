"""Message judge: input, parsing and failure handling with a scripted model (no network)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sellerops.evaluation import judge
from sellerops.settings import get_settings


class ReplyModel(BaseChatModel):
    reply: str = ""
    fail: bool = False
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "reply"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        if self.fail:
            raise RuntimeError("401 Unauthorized: Bearer nvapi-secret")
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.reply))])


TASK = {"task_id": "demo", "expected": {"revenue_idr": 999}}
SNAPSHOT = {"history": [{"tool": "get_sales_summary", "arguments": {"period": "last_30_days"},
                         "observation": {"ok": True, "result": {"revenue_idr": 1872943000, "orders": 775}}}]}
RESULT = {"transcript": [{"role": "user", "text": "Berapa omzet 30 hari?"}],
          "final_action": {"type": "final", "message": "Omzet Rp1.872.943.000 dari 775 order.",
                           "data": {"revenue_idr": 1872943000, "orders": 775}}}


def decision(**overrides: Any) -> str:
    return json.dumps({"faithful": True, "answers_request": True, "reason": "Angka cocok.", **overrides})


def test_scored_pass_and_judge_sees_evidence_not_answer_key() -> None:
    model = ReplyModel(reply=decision(), seen=[])
    result = judge.MessageJudge(model).assess(TASK, SNAPSHOT, RESULT)
    assert result["status"] == "scored" and result["pass"] is True
    system, human = model.seen[0]
    assert system.content == judge.RUBRIC
    sent = json.loads(human.content)
    assert sent["final_message"] == RESULT["final_action"]["message"]
    assert sent["tool_evidence"][0]["result"]["result"]["orders"] == 775
    assert "expected" not in human.content and "999" not in human.content


@pytest.mark.parametrize("field", judge.CRITERIA)
def test_any_failed_criterion_fails(field: str) -> None:
    model = ReplyModel(reply=decision(**{field: False}), seen=[])
    assert judge.MessageJudge(model).assess(TASK, SNAPSHOT, RESULT)["pass"] is False


def test_reasoning_block_and_surrounding_text_are_ignored() -> None:
    reply = '<think>{"faithful": false} hitung dulu</think>\nHasil: ' + decision(faithful=True) + "\n"
    assert judge.parse_decision(reply).faithful is True


@pytest.mark.parametrize("reply", ["tidak ada json", '{"faithful": "yes", "answers_request": true, "reason": "x"}',
                                   decision(extra=1), decision(reason="")])
def test_bad_reply_is_judge_error(reply: str) -> None:
    with pytest.raises(judge.JudgeError):
        judge.MessageJudge(ReplyModel(reply=reply, seen=[])).assess(TASK, SNAPSHOT, RESULT)


def test_transport_error_does_not_leak_details() -> None:
    with pytest.raises(judge.JudgeError) as info:
        judge.MessageJudge(ReplyModel(fail=True, seen=[])).assess(TASK, SNAPSHOT, RESULT)
    assert "nvapi-secret" not in str(info.value) and info.value.__cause__ is None


def test_no_final_is_skipped_without_calling_the_model() -> None:
    model = ReplyModel(reply=decision(), seen=[])
    result = judge.MessageJudge(model).assess(TASK, SNAPSHOT, {"transcript": [], "final_action": None})
    assert result["status"] == "skipped" and model.seen == []


def test_nvidia_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "nvapi-test")
    monkeypatch.setenv("SELLEROPS_JUDGE_MODEL", "vendor/some-model")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        model = judge.nvidia_judge_model()
    finally:
        get_settings.cache_clear()
    assert model.model_name == "vendor/some-model" and model.temperature == 0
    assert model.openai_api_base == "https://integrate.api.nvidia.com/v1"
    assert model.use_responses_api is False
    assert judge.MessageJudge(model).metadata["model"] == "vendor/some-model"


def test_judge_model_falls_back_to_llm_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "nvapi-test")
    monkeypatch.setenv("SELLEROPS_JUDGE_MODEL", "")
    monkeypatch.setenv("LLM_MODEL", "vendor/fallback")
    get_settings.cache_clear()
    try:
        assert judge.nvidia_judge_model().model_name == "vendor/fallback"
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(("missing", "message"), [("LLM_API_KEY", "LLM_API_KEY"),
                                                  ("SELLEROPS_JUDGE_MODEL", "SELLEROPS_JUDGE_MODEL")])
def test_nvidia_model_requires_key_and_model(monkeypatch: pytest.MonkeyPatch, missing: str, message: str) -> None:
    monkeypatch.setenv("LLM_API_KEY", "nvapi-test")
    monkeypatch.setenv("SELLEROPS_JUDGE_MODEL", "vendor/some-model")
    monkeypatch.setenv(missing, "")
    monkeypatch.setenv("LLM_MODEL", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match=message):
            judge.nvidia_judge_model()
    finally:
        get_settings.cache_clear()
