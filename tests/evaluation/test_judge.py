"""Judge transport/protocol tests; no external API calls or model-quality claims."""

from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from sellerops.evaluation import judge


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    def reject(*args, **kwargs):
        raise AssertionError("Unexpected network request")
    monkeypatch.setattr(judge, "urlopen", reject)


def payload(decision=None, finish="STOP"):
    decision = decision or {"correct": True, "complete": True, "grounded": True, "reason": "Facts agree."}
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": json.dumps(decision)}]}}],
            "modelVersion": "test-version", "usageMetadata": {"totalTokenCount": 123}}


def assess():
    return judge.GeminiJudge().assess(
        {"prompt": "Check stock"},
        {"response": "Seven units remain.", "final_action": {"data": {"available_stock": 7}},
         "snapshot": {"history": []}, "config": {"must_not_send": "private-config"}},
        {"task": {"task_id": "demo"}, "expected_answer": "Stock is 7.",
         "expected_final_action": {"data": {"available_stock": 7}}, "acceptance_notes": "Wording may differ."})


def test_request_schema_credentials_and_provenance(monkeypatch):
    sent = []
    def send(request, timeout):
        sent.append((request, timeout))
        return BytesIO(json.dumps(payload()).encode())
    monkeypatch.setattr(judge, "urlopen", send)
    result = assess()
    request, timeout = sent[0]
    body = json.loads(request.data)
    assert timeout == 30 and request.get_header("X-goog-api-key") == "test-secret"
    assert "test-secret" not in request.full_url and "test-secret" not in request.data.decode()
    assert "private-config" not in request.data.decode()
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["systemInstruction"]["parts"][0]["text"] == judge.RUBRIC
    data = json.loads(body["contents"][0]["parts"][0]["text"])
    assert data["actual_answer"] != data["expected_answer"]
    assert result["pass"] is True and result["model_version"] == "test-version"
    assert "test-secret" not in json.dumps(judge.GeminiJudge().metadata)


@pytest.mark.parametrize("decision", [
    {"correct": False, "complete": True, "grounded": True, "reason": "Incorrect stock."},
    {"correct": True, "complete": False, "grounded": True, "reason": "Missing total."},
    {"correct": True, "complete": True, "grounded": False, "reason": "Unsupported reservation."},
])
def test_all_three_criteria_required(monkeypatch, decision):
    monkeypatch.setattr(judge, "urlopen", lambda *a, **kw: BytesIO(json.dumps(payload(decision)).encode()))
    assert assess()["pass"] is False


@pytest.mark.parametrize("response", [
    b"not json", b"{}", json.dumps(payload(finish="MAX_TOKENS")).encode(),
    json.dumps(payload({"correct": 1, "complete": True, "grounded": True, "reason": "bad type"})).encode(),
    json.dumps(payload({"correct": True, "complete": True, "grounded": True, "reason": ""})).encode(),
    json.dumps(payload({"correct": True, "complete": True, "grounded": True, "reason": "ok", "pass": True})).encode(),
])
def test_bad_responses_are_evaluation_errors(monkeypatch, response):
    monkeypatch.setattr(judge, "urlopen", lambda *a, **kw: BytesIO(response))
    with pytest.raises(judge.JudgeError):
        assess()


@pytest.mark.parametrize("error", [HTTPError("secret-url", 429, "test-secret", {}, None),
                                  URLError("test-secret"), TimeoutError("test-secret")])
def test_transport_errors_are_sanitized_without_retry(monkeypatch, error):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(judge, "urlopen", fail)
    with pytest.raises(judge.JudgeError) as caught:
        assess()
    assert "test-secret" not in str(caught.value) and len(calls) == 1


def test_key_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        judge.GeminiJudge()
