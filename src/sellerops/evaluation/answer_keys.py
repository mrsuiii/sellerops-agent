"""Load reviewed dev references separately from training/oracle task records."""

import json
from hashlib import sha256

from ..environment import environment_fixtures
from ..schemas import canonical_json, validate_action
from ..services.episode_service.episode_service import EpisodeService
from .verifier import verify_outcome


def load_answer_keys(project_root, tasks, task_checksum):
    """Reject stale sources, incomplete coverage, and unsupported answer facts.

    Evidence actions validate the key against a fresh simulator. They are not
    forwarded to the evaluated agent or used to require a single tool sequence.
    Prose is model-authored reference text, not an automatic prose grader.
    """
    directory = project_root / "data/evaluation"
    content = (directory / "dev_expected_answers.jsonl").read_bytes()
    manifest_content = (directory / "dev_expected_answers.manifest.json").read_bytes()
    manifest = json.loads(manifest_content)
    checksum = sha256(content).hexdigest()
    environment_checksum = sha256(canonical_json(environment_fixtures()).encode()).hexdigest()
    expected_hashes = {"task_source_sha256": task_checksum,
                       "answer_key_sha256": checksum,
                       "environment_sha256": environment_checksum}
    for key, expected in expected_hashes.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Answer key source mismatch: {key}; review and re-version the keys")
    for key in ("answer_key_version", "author_model", "created_on"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Answer key manifest requires {key}")
    if manifest.get("split") != "dev" or manifest.get("task_count") != len(tasks):
        raise ValueError("Answer key manifest must describe the complete dev split")
    keys = {}
    fields = {"task_id", "expected_answer", "expected_final_action", "evidence_actions", "acceptance_notes"}
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("Invalid answer key fields")
        for field in ("task_id", "expected_answer", "acceptance_notes"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Answer key requires non-empty {field}")
        task_id = record["task_id"]
        if task_id in keys:
            raise ValueError(f"Duplicate answer key: {task_id}")
        validate_action(record["expected_final_action"])
        if record["expected_final_action"]["type"] not in {"final", "ask_user"}:
            raise ValueError(f"Answer key must be terminal: {task_id}")
        if not isinstance(record["evidence_actions"], list):
            raise ValueError("Answer key evidence_actions must be a list")
        for action in record["evidence_actions"]:
            validate_action(action)
            if action["type"] != "tool_call":
                raise ValueError("Answer key evidence must contain only tool calls")
        keys[task_id] = record
    if set(keys) != {task["task_id"] for task in tasks}:
        raise ValueError("Answer keys must cover exactly the dev task IDs")
    for task in tasks:
        key = keys[task["task_id"]]
        service = EpisodeService()
        episode = service.create_episode(task["task_id"])
        for action in key["evidence_actions"]:
            service.execute_tool(episode.session_id, action)
        report = verify_outcome(task, service.snapshot(episode.session_id), key["expected_final_action"])
        if report["decision"] != "pass":
            raise ValueError(f"Invalid answer key for {task['task_id']}: {report['reason_code']}")
    metadata = {**expected_hashes,
                "answer_key_manifest_sha256": sha256(manifest_content).hexdigest(),
                "answer_key_version": manifest["answer_key_version"],
                "answer_key_author_model": manifest["author_model"]}
    return keys, metadata
