"""Answer keys stay complete, grounded, versioned, and outside agent inputs."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from sellerops.evaluation import answer_keys, dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sources(tmp_path, monkeypatch):
    # Copy only development references; frozen and training data are untouched.
    for relative in ("data/tasks/dev.jsonl", "data/evaluation/dev_expected_answers.jsonl",
                     "data/evaluation/dev_expected_answers.manifest.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())
    monkeypatch.setattr(dataset, "PROJECT_ROOT", tmp_path)
    return tmp_path / "data/evaluation"


def rewrite_keys(directory, change):
    path = directory / "dev_expected_answers.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    manifest_path = directory / "dev_expected_answers.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["answer_key_sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_all_dev_keys_are_grounded_and_reference_only():
    # Loader verifies every structured answer with fresh tools + real verifier.
    records = dataset.load_dev_records()
    assert len(records) == 30
    assert {r["metadata"]["answer_key_author_model"] for r in records} == {"gpt-6-astra"}
    for record in records:
        assert set(record["inputs"]) == {"task_id", "prompt"}
        assert set(record["outputs"]) == {"task", "expected_answer", "expected_final_action", "acceptance_notes"}
        assert record["outputs"]["expected_answer"].strip()
        assert record["outputs"]["acceptance_notes"].strip()
        assert "evidence_actions" not in record["outputs"]
    negative = [r for r in records if r["outputs"]["task"]["context"].get("proposed_discount_percent", 0) < 0]
    assert negative
    assert all(r["outputs"]["expected_final_action"]["data"]["proposal_allowed"] is False for r in negative)


@pytest.mark.parametrize("source", ["answers", "tasks", "environment"])
def test_stale_source_rejected(sources, monkeypatch, source):
    if source == "environment":
        fixtures = answer_keys.environment_fixtures()
        fixtures["inventory"][0]["stock"] += 1
        monkeypatch.setattr(answer_keys, "environment_fixtures", lambda: fixtures)
    else:
        path = sources / "dev_expected_answers.jsonl" if source == "answers" else sources.parent / "tasks/dev.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source mismatch"):
        dataset.load_dev_records()


@pytest.mark.parametrize("case,match", [("missing", "cover exactly"), ("duplicate", "Duplicate answer key"),
                                       ("wrong_fact", "Invalid answer key for"), ("empty_prose", "non-empty expected_answer")])
def test_reviewed_hash_does_not_bypass_key_validation(sources, case, match):
    def change(records):
        if case == "missing":
            records.pop()
        elif case == "duplicate":
            records.append(records[0])
        elif case == "wrong_fact":
            records[0]["expected_final_action"]["data"]["products"][0]["price_idr"] += 1
        else:
            records[0]["expected_answer"] = ""

    rewrite_keys(sources, change)
    with pytest.raises(ValueError, match=match):
        dataset.load_dev_records()


def test_answer_revision_changes_dataset_identity_without_changing_prompts(sources):
    before = dataset.load_dev_records()
    rewrite_keys(sources, lambda records: records[0].update(
        expected_answer=records[0]["expected_answer"] + " "
    ))
    after = dataset.load_dev_records()
    assert [r["inputs"] for r in before] == [r["inputs"] for r in after]
    assert before[0]["metadata"]["task_source_sha256"] == after[0]["metadata"]["task_source_sha256"]
    assert before[0]["metadata"]["source_sha256"] != after[0]["metadata"]["source_sha256"]
    assert dataset.as_examples(before)[0].dataset_id != dataset.as_examples(after)[0].dataset_id
