"""Use the real graph, simulator, grader and LangSmith SDK without network/weights."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

# Agent nodes import torch; skip when the `local-model` extra is not installed (e.g. CI).
pytest.importorskip("torch")

import requests
from langsmith import Client, traceable
from langsmith.run_helpers import tracing_context
from langsmith.schemas import Example
from langsmith.utils import LangSmithNotFoundError

from sellerops.agent import nodes
from sellerops.evaluation import dataset, evaluators, runner
from sellerops.evaluation.metrics import pass_metrics
from sellerops.evaluation.target import make_target

ACT = {"decision": "act", "text": ""}
STOCK = {"type": "tool_call", "tool": "get_inventory", "arguments": {"product_id": "P1001"}}
FINAL = {"type": "final", "status": "success", "message": "Stok tersedia 7 unit.",
         "data": {"product_id": "P1001", "available_stock": 7}}
REPLY = {"decision": "reply", "text": "Stok tersedia 7 unit."}
TASK = {"task_id": "dev-fixture", "family": "inventory_offer",
        "context": {"product_id": "P1001", "quantity": 1, "mode": "stock_only"}}
INPUTS = {"task_id": TASK["task_id"], "prompt": "Cek stok P1001"}


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    attempts = []

    def reject_network(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("Test must not access the network")

    monkeypatch.setattr(requests.Session, "request", reject_network)
    # Even inherited tracing/key settings must not override explicit offline mode.
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-never-send")
    with tracing_context(enabled=False):
        yield
    assert attempts == []


def script(monkeypatch, completions):
    sequence = iter(completions)
    calls = []

    def generate(messages, config):
        calls.append((deepcopy(messages), config))
        result = next(sequence)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, str) else json.dumps(result)

    monkeypatch.setattr(nodes, "_generate_text", generate)
    return calls


def dev_record(task_id="dev-fixture"):
    task = {**deepcopy(TASK), "task_id": task_id}
    return {"inputs": {**INPUTS, "task_id": task_id}, "outputs": {"task": task},
            "metadata": {"family": "inventory_offer", "source_sha256": "a" * 64,
                         "dataset_version": "test-fixture", "environment_version": "sellerops-sim-v0.1"}}


def test_success_real_graph_and_episode_isolation(monkeypatch):
    calls = script(monkeypatch, [ACT, STOCK, ACT, FINAL, REPLY] * 2)
    target = make_target()
    first, second = target(INPUTS), target(INPUTS)
    assert first["snapshot"]["session_id"] != second["snapshot"]["session_id"]
    for output in (first, second):
        assert output["model_calls"] == output["committed_model_calls"] == 5
        assert output["tool_calls"] == 1
        assert output["final_action"] == FINAL
        assert all(a["schema_valid"] for a in output["attempts"])
        assert evaluators.grade_episode(TASK, output)["decision"] == "pass"
        json.dumps(output)
    assert first["attempts"] is not second["attempts"]
    assert "context" not in calls[0][0][-1]["content"]


@pytest.mark.parametrize("completions,reason,calls", [
    ([ACT, STOCK, REPLY], "missing_structured_outcome", 3),
    (["invalid JSON"], "retry_exhausted", 1),
    ([ACT, "invalid JSON"], "retry_exhausted", 2),
    ([ACT, RuntimeError("generation failed")], "runtime_error", 2),
    ([ACT, FINAL, {"decision": "ask", "text": "Produk apa?"}], "terminal_status_mismatch", 3),
    ([ACT, FINAL, ACT, STOCK, REPLY], "missing_structured_outcome", 5),
])
def test_failure_reasons_and_attempt_counts(monkeypatch, completions, reason, calls):
    script(monkeypatch, completions)
    output = make_target(max_retries=0)(INPUTS)
    assert output["model_calls"] == calls
    assert evaluators.grade_episode(TASK, output)["reason_code"] == reason
    if "invalid JSON" in completions:
        assert output["attempts"][-1]["raw_output"] == "invalid JSON"
        assert output["attempts"][-1]["schema_valid"] is False
        assert output["committed_model_calls"] == calls


def test_budget_failure_cannot_pass_old_candidate(monkeypatch):
    script(monkeypatch, [ACT, FINAL])
    output = make_target(max_model_calls=2)(INPUTS)
    assert output["model_calls"] == 2
    assert evaluators.grade_episode(TASK, output)["reason_code"] == "budget_exhausted"


def test_schema_retries_recover_on_same_node_and_count_every_attempt(monkeypatch):
    calls = script(monkeypatch, ["invalid router", ACT, "invalid action", STOCK, ACT, FINAL, REPLY])
    output = make_target()(INPUTS)
    assert output["model_calls"] == output["committed_model_calls"] == len(calls) == 7
    assert output["retry_count"] == 2 and output["tool_calls"] == 1
    assert [a["is_retry"] for a in output["attempts"]] == [False, True, False, True, False, False, False]
    assert [a["node"] for a in output["attempts"][:4]] == ["router", "router", "action", "action"]
    assert "failed schema validation" in calls[1][0][-1]["content"]
    assert "failed schema validation" in calls[3][0][-1]["content"]
    assert all("failed schema validation" not in m["content"] for m in calls[4][0])
    assert len(output["messages"]) == 5  # input, action, observation, final, reply
    assert evaluators.grade_episode(TASK, output)["decision"] == "pass"
    scores = {f["key"]: f["score"] for f in evaluators.episode_evaluator(INPUTS, output, {"task": TASK})["results"]}
    assert scores["retry_count"] == 2
    assert scores["router_schema_validity"] == .75
    assert scores["action_schema_validity"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("completions,options,calls,retries,reason", [
    (["bad"] * 3, {}, 3, 2, "retry_exhausted"),
    (["bad", ACT, "bad"], {"max_retries": 1}, 3, 1, "retry_exhausted"),
    (["bad"], {"max_model_calls": 1}, 1, 0, "budget_exhausted"),
    (["bad"] * 2, {"max_model_calls": 2}, 2, 1, "budget_exhausted"),
    (["bad", RuntimeError("device failed")], {}, 2, 1, "runtime_error"),
])
def test_retry_limits_and_runtime_errors(monkeypatch, completions, options, calls, retries, reason):
    actual = script(monkeypatch, completions)
    output = make_target(**options)(INPUTS)
    assert output["model_calls"] == output["committed_model_calls"] == len(actual) == calls
    assert output["retry_count"] == retries
    assert output["tool_calls"] == 0 and output["final_action"] is None
    assert evaluators.grade_episode(TASK, output)["reason_code"] == reason


def test_trial_seeds_reproduce_independently_and_restore_cpu_rng(monkeypatch):
    import torch
    draws = []

    def generate(*args):
        draws.append(torch.rand(3).tolist())
        return json.dumps(REPLY)

    monkeypatch.setattr(nodes, "_generate_text", generate)
    before = torch.random.get_rng_state().clone()
    target = make_target(temperature=.7, seed=11)
    first, second = target(INPUTS), target(INPUTS)
    assert torch.equal(before, torch.random.get_rng_state())
    assert first["trial_index"] == 0 and second["trial_index"] == 1
    assert first["generation_seed"] != second["generation_seed"] and draws[0] != draws[1]
    repeated = make_target(temperature=.7, seed=11)(INPUTS)
    assert repeated["generation_seed"] == first["generation_seed"] and draws[0] == draws[2]


def test_pass_estimators_use_combinations_not_any_success():
    samples = [("a", s) for s in [1, 0, 1, 0]] + [("b", 0)] * 4
    result = pass_metrics(samples, repetitions=4, k=2, task_ids=["a", "b"])
    assert result["pass@1"] == .25
    assert result["pass@2"] == pytest.approx(5 / 12)
    assert set(result) == {"pass@1", "pass@2"}
    assert pass_metrics([("a", 1.0)] * 3, repetitions=3, k=3, task_ids=["a"])["pass@3"] == 1


@pytest.mark.parametrize("samples", [[], [("a", 1)], [("a", 1), ("a", None)],
                                         [("a", 1), ("b", 0)], [("a", 1)] * 3])
def test_incomplete_or_unscored_trials_invalidate_pass_metrics(samples):
    assert pass_metrics(samples, repetitions=2, k=2, task_ids=["a"]) == {"pass@1": None, "pass@2": None}


@pytest.mark.parametrize("options", [
    {"repetitions": 0}, {"repetitions": True}, {"pass_k": 2}, {"pass_k": 0},
    {"max_retries": -1}, {"max_retries": True}, {"temperature": float("nan")},
    {"temperature": -1}, {"seed": -1},
])
def test_invalid_eval_configuration_before_work(options, tmp_path):
    with pytest.raises(ValueError):
        runner.run_evaluation(output_root=tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_sdk_repetitions_and_summary_feedback_match_local_report(monkeypatch, tmp_path):
    from collections import Counter
    from pathlib import Path
    monkeypatch.setattr(runner, "load_dev_records", lambda: [dev_record("a"), dev_record("b")])
    successful = [ACT, STOCK, ACT, FINAL, REPLY]
    script(monkeypatch, successful + ["bad"] + successful + ["bad"] * 3)
    real_evaluate, sdk_results = runner.evaluate, []

    def evaluate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        sdk_results.append(result)
        return result

    monkeypatch.setattr(runner, "evaluate", evaluate)
    result = runner.run_evaluation(output_root=tmp_path, repetitions=3, pass_k=3, max_retries=0)
    summary = result["summary"]
    rows = [json.loads(line) for line in (Path(result["output_dir"]) / "episodes.jsonl").read_text().splitlines()]
    assert summary["complete"] and summary["episodes"] == summary["expected_episodes"] == 6
    assert Counter(r["task_id"] for r in rows) == {"a": 3, "b": 3}
    assert len({r["output"]["snapshot"]["session_id"] for r in rows}) == 6
    assert len({r["output"]["generation_seed"] for r in rows}) == 6
    assert summary["pass@1"] == pytest.approx(1 / 3)
    assert summary["pass@3"] == len({r["task_id"] for r in rows if r["scores"]["structured_outcome_pass"]}) / 2
    # SDK summary evaluators also execute offline; these same scores publish to
    # the experiment in cloud mode. No API key/network is needed for this check.
    feedback = sdk_results[0]._summary_results["results"]
    assert {f.key: f.score for f in feedback} == {k: summary[k] for k in ("pass@1", "pass@3")}


def test_blocked_policy_attempt_remains_visible(monkeypatch):
    blocked = {"type": "tool_call", "tool": "calculate_offer",
               "arguments": {"product_id": "P1001", "quantity": 1, "discount_percent": 99}}
    script(monkeypatch, [ACT, blocked, ACT, FINAL, REPLY])
    output = make_target()(INPUTS)
    feedback = evaluators.episode_evaluator(INPUTS, output, {"task": TASK})["results"]
    scores = {f["key"]: f["score"] for f in feedback}
    assert scores["structured_outcome_pass"] == 0
    assert scores["blocked_policy_attempt"] == 1


def test_reference_context_never_in_target_inputs():
    records = dataset.load_dev_records()
    assert len(records) == 30
    for record in records:
        assert set(record["inputs"]) == {"task_id", "prompt"}
        assert set(record["outputs"]["task"]) == {"task_id", "family", "context"}
        assert "trajectory" not in record and "verifier" not in record
    with pytest.raises(ValueError, match="only task_id and prompt"):
        make_target()({**INPUTS, "context": TASK["context"]})


def test_real_answer_key_is_exported_by_sdk_without_entering_prompt(monkeypatch, tmp_path):
    from pathlib import Path
    record = dataset.load_dev_records()[0]
    reference = record["outputs"]
    task = reference["task"]
    search = {"type": "tool_call", "tool": "search_products", "arguments": task["context"]}
    calls = script(monkeypatch, [ACT, search, ACT, reference["expected_final_action"],
                                {"decision": "reply", "text": reference["expected_answer"]}])
    monkeypatch.setattr(runner, "load_dev_records", lambda: [record])
    result = runner.run_evaluation(output_root=tmp_path)
    assert result["summary"]["pass@1"] == 1
    folder = Path(result["output_dir"])
    saved = json.loads((folder / "episodes.jsonl").read_text().strip())
    metadata = json.loads((folder / "metadata.json").read_text())
    assert saved["reference_outputs"] == reference
    assert metadata["answer_key_author_model"] == "gpt-6-astra"
    assert metadata["answer_key_sha256"] == record["metadata"]["answer_key_sha256"]
    assert json.loads(calls[0][0][-1]["content"]) == [{"role": "human", "content": record["inputs"]["prompt"]}]
    assert calls[1][0][-1] == {"role": "user", "content": record["inputs"]["prompt"]}


@pytest.mark.parametrize("verdict", [True, False, "error", "skip"])
def test_judge_sdk_aggregates_once_per_eligible_episode(monkeypatch, tmp_path, verdict):
    from pathlib import Path

    from sellerops.evaluation.judge import GeminiJudge, JudgeError
    record = dataset.load_dev_records()[0]
    ref = record["outputs"]
    monkeypatch.setattr(runner, "load_dev_records", lambda: [record])
    monkeypatch.setenv("GEMINI_API_KEY", "test-judge-secret")
    completions = [ACT, {"type": "tool_call", "tool": "search_products", "arguments": ref["task"]["context"]},
                   ACT, ref["expected_final_action"], {"decision": "reply", "text": ref["expected_answer"]}]
    script(monkeypatch, ["broken"] * 2 if verdict == "skip" else completions * 2)
    calls, sdk_results = [], []
    def assess(self, inputs, outputs, reference):
        calls.append(inputs)
        if verdict == "error":
            raise JudgeError("Gemini HTTP 429; quota")
        return {"status": "scored", "pass": verdict, "correct": verdict, "complete": True,
                "grounded": verdict, "reason": "Test verdict", "error": None}
    monkeypatch.setattr(GeminiJudge, "assess", assess)
    real_evaluate = runner.evaluate
    def evaluate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        sdk_results.append(result)
        return result
    monkeypatch.setattr(runner, "evaluate", evaluate)
    result = runner.run_evaluation(output_root=tmp_path, judge="gemini", repetitions=2, pass_k=2, max_retries=0)
    summary = result["summary"]
    expected = None if verdict == "error" else int(verdict is True)
    assert summary["end_to_end"] == {"pass@1": expected, "pass@2": expected}
    assert summary["pass@1"] == (0 if verdict == "skip" else 1)
    assert summary["judge_errors"] == (2 if verdict == "error" else 0)
    assert summary["evaluation_errors"] == 0
    assert len(calls) == (0 if verdict == "skip" else 2)  # summary must not re-call API
    feedback = {f.key: f.score for f in sdk_results[0]._summary_results["results"]}
    assert feedback["end_to_end_pass@2"] == expected
    exported = (Path(result["output_dir"]) / "episodes.jsonl").read_text()
    assert "test-judge-secret" not in exported


def test_sdk_offline_batch_continues_after_agent_failure(monkeypatch, tmp_path):
    records = [dev_record("dev-bad"), dev_record("dev-good")]
    monkeypatch.setattr(runner, "load_dev_records", lambda: records)
    script(monkeypatch, ["broken", ACT, STOCK, ACT, FINAL, REPLY])
    result = runner.run_evaluation(output_root=tmp_path, max_retries=0)
    summary = result["summary"]
    assert summary["complete"] and summary["episodes"] == 2
    assert summary["passed"] == 1 and summary["structured_outcome_pass_rate"] == .5
    assert summary["runtime_failures"] == 1 and summary["evaluation_errors"] == 0
    from pathlib import Path
    saved = [json.loads(line) for line in (Path(result["output_dir"]) / "episodes.jsonl").read_text().splitlines()]
    assert {r["reason_code"] for r in saved} == {"retry_exhausted", "outcome_valid"}
    assert all(r["run_id"] and r["output"]["snapshot"] for r in saved)


def test_evaluator_bug_is_not_an_agent_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_dev_records", lambda: [dev_record()])
    script(monkeypatch, [ACT, STOCK, ACT, FINAL, REPLY])

    def broken_verifier(*args):
        raise RuntimeError("grader bug")

    monkeypatch.setattr(evaluators, "verify_outcome", broken_verifier)
    result = runner.run_evaluation(output_root=tmp_path)
    assert result["summary"]["evaluation_errors"] == 1
    assert result["summary"]["structured_outcome_pass_rate"] is None
    assert result["summary"]["runtime_failures"] == 0


def test_cloud_requires_key_before_model_or_upload(monkeypatch, tmp_path):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        runner.run_evaluation(cloud=True, output_root=tmp_path)
    assert not list(tmp_path.iterdir())


class MemoryClient:
    def __init__(self):
        self.dataset = None
        self.examples = {}
        self.writes = 0

    def read_dataset(self, **kwargs):
        if self.dataset is None:
            raise LangSmithNotFoundError("not found")
        return self.dataset

    def create_dataset(self, **kwargs):
        self.dataset = SimpleNamespace(id=uuid4(), metadata=kwargs["metadata"])
        return self.dataset

    def list_examples(self, **kwargs):
        return list(self.examples.values())

    def create_examples(self, *, dataset_id, examples):
        self.writes += 1
        for example in examples:
            self.examples[example["id"]] = Example(dataset_id=dataset_id, **deepcopy(example))


def test_cloud_dataset_reuse_and_interrupted_population():
    client = MemoryClient()
    examples = dataset.as_examples([dev_record("one"), dev_record("two")])
    first = dataset.prepare_cloud_dataset(client, examples)
    second = dataset.prepare_cloud_dataset(client, examples)
    assert client.writes == 1 and [e.id for e in first] == [e.id for e in second]
    del client.examples[first[0].id]
    dataset.prepare_cloud_dataset(client, examples)
    assert client.writes == 2 and len(client.examples) == 2
    client.examples[first[0].id].outputs["task"]["context"]["quantity"] = 999
    with pytest.raises(ValueError, match="differs from local"):
        dataset.prepare_cloud_dataset(client, examples)
    assert client.writes == 2


def test_local_model_trace_and_real_token_counts(monkeypatch):
    import torch
    from langchain_core.messages import HumanMessage

    from sellerops.evaluation.telemetry import capture_attempts

    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = {"input_ids": torch.tensor([[10, 11]])}
    tokenizer.decode.return_value = json.dumps(FINAL)
    model = Mock(device=torch.device("cpu"))
    model.config._commit_hash = "revision-for-test"
    model.generate.return_value = torch.tensor([[10, 11, 12, 13, 14]])
    monkeypatch.setattr(nodes, "_load_model", lambda **kwargs: SimpleNamespace(model=model, tokenizer=tokenizer))
    finished = []

    @traceable
    def parent():
        with capture_attempts() as attempts:
            nodes.call_model({"messages": [HumanMessage(content="Cek stok P1001")], "model_calls": 0},
                             {"configurable": {"model_name": "qwen-base"}})
        return attempts

    client = Client(api_url="http://localhost:1984", api_key="offline", auto_batch_tracing=False, info={})
    with tracing_context(enabled="local", client=client):
        attempts = parent(langsmith_extra={"on_end": finished.append})
    client.close()
    assert attempts[0]["usage"] == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    child = finished[0].child_runs[0]
    assert child.run_type == "llm" and child.metadata["usage_metadata"]["total_tokens"] == 5
    assert child.outputs["choices"][0]["message"]["content"] == json.dumps(FINAL)
    assert set(child.inputs) == {"messages"}
    assert child.metadata["model_revision"] == "revision-for-test"
